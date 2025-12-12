#!/usr/bin/env python3
"""Unified TCP server for the VR + Myo bridge.


This rebuilt server keeps Unity and the Python backend in sync by combining:
- Myo BLE management (scan/connect/stream/inference)
- Real-time inference driven by models trained with Core.session_pipeline
- Guided recording/training sessions that stream prompts to Unity via PIPELINE events
- AprilTag tracking with multi-joint support and JSON status payloads


The protocol matches the expectations of NetworkBridgeClient.cs / ServerEMGInterpreter.cs.
"""
from __future__ import annotations


import argparse
import asyncio
from collections import deque
import json
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple


import cv2 as cv
import joblib
import numpy as np
from pupil_apriltags import Detector


from Core.session_pipeline import (
    PREP_SECONDS,
    GESTURE_BLEND_MAP,
    ORIENTATION_VALUE_MAP,
    RecordingStep,
    SessionRecorder,
    WINDOW_SIZE,
    STEP_SIZE,
    build_feature_table,
    build_guided_sequence,
    build_manifest,
    compute_feature_vector,
    save_artifacts,
    train_models,
)
from myo_ble_client import MyoManager


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RECORDINGS_DIR = BASE_DIR / "recordings"
MODELS_DIR.mkdir(exist_ok=True)
RECORDINGS_DIR.mkdir(exist_ok=True)


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5005




def make_json_safe(value: Any) -> Any:
    """Recursively convert numpy/scalar types to native Python for json.dumps."""
    if isinstance(value, dict):
        return {key: make_json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [make_json_safe(val) for val in value]
    if isinstance(value, tuple):
        return [make_json_safe(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value




class ClientConnection:
    """Lightweight wrapper around (reader, writer) with helper send util."""


    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer
        self.peer = writer.get_extra_info("peername")


    async def send_line(self, text: str) -> None:
        data = text if text.endswith("\n") else f"{text}\n"
        try:
            self.writer.write(data.encode("utf-8"))
            await self.writer.drain()
        except ConnectionError:
            pass


    async def close(self) -> None:
        try:
            self.writer.close()
            await self.writer.wait_closed()
        except Exception:
            pass


    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"ClientConnection({self.peer})"




class MyoService:
    """Async helper that wraps MyoManager and exposes EMG callbacks."""


    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.manager = MyoManager()
        self.active_address: Optional[str] = None
        self.listeners: Set[Callable[[int, List[int]], None]] = set()
        self._stream_refcount = 0
        self._stream_lock = asyncio.Lock()


    async def scan(self) -> List[Dict[str, Any]]:
        return await self.manager.discover(timeout=5.0)


    async def connect(self, address: Optional[str] = None) -> Any:
        if address is None:
            devices = await self.scan()
            if not devices:
                raise RuntimeError("No Myo devices discovered")
            target = devices[0]
            address = target["address"]
            name = target.get("name", "Myo")
        else:
            name = "Myo"


        client = await self.manager.connect(address, name)
        if not client:
            raise RuntimeError(f"Failed to connect to {address}")


        client.set_emg_callback(self._handle_emg_notification)
        client.set_disconnect_callback(lambda _: self.loop.call_soon_threadsafe(self._handle_disconnect))
        self.active_address = address
        return client


    async def disconnect(self, address: Optional[str] = None) -> Optional[str]:
        target = address or self.active_address
        if target:
            await self.manager.disconnect(target)
            if target == self.active_address:
                self.active_address = None
            return target
        return None


    async def acquire_stream(self) -> None:
        async with self._stream_lock:
            self._stream_refcount += 1
            if self._stream_refcount == 1:
                client = self._active_client
                if not client:
                    raise RuntimeError("Myo not connected")
                await client.start_streaming(emg=True, imu=False)


    async def release_stream(self) -> None:
        async with self._stream_lock:
            self._stream_refcount = max(0, self._stream_refcount - 1)
            if self._stream_refcount == 0:
                client = self._active_client
                if client and client.is_streaming:
                    await client.stop_streaming()


    def register_listener(self, callback: Callable[[int, List[int]], None]) -> Callable[[], None]:
        self.listeners.add(callback)
        return lambda: self.listeners.discard(callback)


    def status(self) -> Dict[str, Any]:
        return self.manager.get_status_all()


    @property
    def _active_client(self):
        if self.active_address:
            return self.manager.get_client(self.active_address)
        return None


    def _handle_disconnect(self) -> None:
        self.active_address = None
        self._stream_refcount = 0


    def _handle_emg_notification(self, sample: Iterable[int], client: Any) -> None:
        values = [int(v) for v in sample]
        if len(values) < 16:
            values.extend([0] * (16 - len(values)))
        timestamp_ms = int(time.time() * 1000)
        for listener in list(self.listeners):
            self.loop.call_soon_threadsafe(listener, timestamp_ms, list(values))




class AprilTagTracker:
    """Background AprilTag tracker that mirrors the previous implementation."""


    def __init__(self, camera_id: int = 0, tag_size: float = 0.05, family: str = "tag36h11") -> None:
        self.camera_id = camera_id
        self.tag_size = tag_size
        self.tag_family = family
        self.tracked_joints: Dict[int, float] = {}
        self.joints_lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.position_queue: "queue.Queue[Dict[int, Any]]" = queue.Queue(maxsize=5)
        self.last_positions: Dict[int, Any] = {}
        self.cap = None
        self.detector: Optional[Detector] = None
        self.camera_matrix = None


    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.thread.start()
        print(f"[APRILTAG] Tracker started (camera {self.camera_id})")


    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        print("[APRILTAG] Tracker stopped")


    def add_joint(self, tag_id: int, offset: float) -> None:
        with self.joints_lock:
            self.tracked_joints[tag_id] = offset


    def remove_joint(self, tag_id: int) -> None:
        with self.joints_lock:
            self.tracked_joints.pop(tag_id, None)


    def clear_joints(self) -> None:
        with self.joints_lock:
            self.tracked_joints.clear()
            self.last_positions.clear()


    def get_tracked_joints(self) -> List[int]:
        with self.joints_lock:
            return list(self.tracked_joints.keys())


    def get_latest_positions(self) -> Dict[int, Any]:
        updates: Dict[int, Any] = {}
        while True:
            try:
                # latest = self.position_queue.get_nowait()
                payload = self.position_queue.get_nowait()
            except queue.Empty:
                break
        # if latest:
        #    self.last_positions = latest
            else:
                updates.update(payload)

        if updates:
            for tag_id, data in updates.items():
                self.last_positions[tag_id] = data
                
        return {k: dict(v) for k, v in self.last_positions.items()}


    def status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "tracked_joints": self.get_tracked_joints(),
            "latest_positions": self.get_latest_positions(),
        }


    def _tracking_loop(self) -> None:
        try:
            self.cap = cv.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"[APRILTAG] Unable to open camera {self.camera_id}")
                self.running = False
                return
            width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)) or 480
            fx = fy = (width / 2.0) / np.tan(np.deg2rad(60.0) / 2.0)
            cx, cy = width / 2.0, height / 2.0
            self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            self.detector = Detector(families=self.tag_family)
        except Exception as exc:
            print(f"[APRILTAG] Failed to init tracker: {exc}")
            self.running = False
            return


        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            detections = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(self.camera_matrix[0, 0], self.camera_matrix[1, 1], self.camera_matrix[0, 2], self.camera_matrix[1, 2]),
                tag_size=self.tag_size,
            )
            with self.joints_lock:
                tracked = dict(self.tracked_joints)
            joint_payload: Dict[int, Any] = {}
            timestamp = time.time()
            for det in detections:
                tag_id = int(det.tag_id)
                if tracked and tag_id not in tracked:
                    continue
                t = det.pose_t.reshape(-1)
                R = det.pose_R
                roll, pitch, yaw = rotation_to_euler_xyz(R)
                offset = tracked.get(tag_id, 0.0)
                joint_center = (t[0], t[1], t[2] - offset)
                joint_payload[tag_id] = {
                    "timestamp": timestamp,
                    "tag_id": tag_id,
                    "tag_position": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
                    "joint_center": {"x": float(joint_center[0]), "y": float(joint_center[1]), "z": float(joint_center[2])},
                    "offset_applied": offset,
                    "rotation": {"roll": float(roll), "pitch": float(pitch), "yaw": float(yaw)},
                    "rotation_degrees": {
                        "roll": float(np.degrees(roll)),
                        "pitch": float(np.degrees(pitch)),
                        "yaw": float(np.degrees(yaw)),
                    },
                }
            if joint_payload:
                try:
                    self.position_queue.put_nowait(joint_payload)
                except queue.Full:
                    pass
        if self.cap:
            self.cap.release()




def rotation_to_euler_xyz(R: np.ndarray) -> Tuple[float, float, float]:
    sy = -R[2, 0]
    cy = np.sqrt(max(0.0, 1 - sy**2))
    singular = cy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arcsin(sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arcsin(sy)
        yaw = 0.0
    return roll, pitch, yaw




class InferenceEngine:
    """Consumes EMG samples, runs predictions, and emits JSON payloads."""


    def __init__(
        self,
        myo: MyoService,
        tracker: AprilTagTracker,
        publisher: Callable[[Dict[str, Any]], None],
    ) -> None:
        self.myo = myo
        self.tracker = tracker
        self.publisher = publisher
        self.bundle: Optional[Dict[str, Any]] = None
        self.model_name: Optional[str] = None
        self.window = deque()
        self.window_size = WINDOW_SIZE
        self.step_size = STEP_SIZE
        self.running = False
        self._listener: Optional[Callable[[], None]] = None


    @staticmethod
    def _display_name(stem: str) -> str:
        return stem[8:] if stem.startswith("session_") else stem


    def list_models(self) -> List[str]:
        names: List[str] = []
        MODELS_DIR.mkdir(exist_ok=True)
        for path in MODELS_DIR.glob("*.pkl"):
            names.append(self._display_name(path.stem))
        return sorted(set(names))


    async def load_model(self, name: str) -> str:
        MODELS_DIR.mkdir(exist_ok=True)
        candidates = [
            MODELS_DIR / name,
            MODELS_DIR / f"{name}.pkl",
            MODELS_DIR / f"session_{name}.pkl",
        ]
        for path in candidates:
            if path.exists():
                bundle = joblib.load(path)
                self.bundle = bundle
                self.model_name = path.stem
                self.window_size = int(bundle.get("window_size", 200))
                self.step_size = int(bundle.get("step_size", 100))
                self.window = deque(maxlen=self.window_size * 2)
                print(f"[INFERENCE] Loaded model {path}")
                return self.model_name
        normalized = name.strip()
        # Fallback: resolve names that were shown without the leading prefix
        for path in MODELS_DIR.glob("*.pkl"):
            if self._display_name(path.stem) == normalized:
                bundle = joblib.load(path)
                self.bundle = bundle
                self.model_name = path.stem
                self.window_size = int(bundle.get("window_size", 200))
                self.step_size = int(bundle.get("step_size", 100))
                self.window = deque(maxlen=self.window_size * 2)
                print(f"[INFERENCE] Loaded model {path}")
                return self.model_name
        raise FileNotFoundError(name)


    async def start(self) -> None:
        if not self.bundle:
            raise RuntimeError("No model loaded")
        if self.running:
            return
        await self.myo.acquire_stream()
        self._listener = self.myo.register_listener(self._on_sample)
        self.running = True
        print("[INFERENCE] Streaming started")


    async def stop(self) -> None:
        if not self.running:
            return
        if self._listener:
            self._listener()
            self._listener = None
        await self.myo.release_stream()
        self.window.clear()
        self.running = False
        print("[INFERENCE] Streaming stopped")


    def _on_sample(self, timestamp_ms: int, sample: List[int]) -> None:
        if not self.running or not self.bundle:
            return
        self.window.append(sample)
        if len(self.window) < self.window_size:
            return
        window_arr = np.array(list(self.window)[-self.window_size:])
        features = compute_feature_vector(window_arr)
        scaler = self.bundle["scaler"]
        feats_scaled = scaler.transform([features])
        gesture_classifier = self.bundle["gesture_classifier"]
        gesture_encoder = self.bundle["gesture_encoder"]
        gesture_proba = gesture_classifier.predict_proba(feats_scaled)[0]
        gesture_idx = int(np.argmax(gesture_proba))
        gesture_label = gesture_encoder.inverse_transform([gesture_idx])[0]
        orientation_classifier = self.bundle["orientation_classifier"]
        orientation_encoder = self.bundle["orientation_encoder"]
        orientation_proba = orientation_classifier.predict_proba(feats_scaled)[0]
        orientation_idx = int(np.argmax(orientation_proba))
        orientation_label = orientation_encoder.inverse_transform([orientation_idx])[0]


        gesture_reg = self.bundle.get("gesture_reg")
        if gesture_reg is not None:
            gesture_blend = float(gesture_reg.predict(feats_scaled)[0])
        else:
            gesture_blend = float(GESTURE_BLEND_MAP.get(gesture_label, 0.0))
        orientation_reg = self.bundle.get("orientation_reg")
        if orientation_reg is not None:
            orientation_value = float(orientation_reg.predict(feats_scaled)[0])
        else:
            orientation_value = float(ORIENTATION_VALUE_MAP.get(orientation_label, 0.0))


        gesture_blend = float(np.clip(gesture_blend, -1.0, 1.0))
        orientation_value = float(np.clip(orientation_value, -1.0, 1.0))
        gesture_conf = float(gesture_proba[gesture_idx])
        orientation_conf = float(orientation_proba[orientation_idx])
        payload = {
            "type": "prediction",
            "timestamp": timestamp_ms / 1000.0,
            "pose": f"{gesture_label}:{orientation_label}",
            "gesture": gesture_label,
            "gesture_blend": gesture_blend,
            "gesture_confidence": gesture_conf,
            "orientation": orientation_label,
            "orientation_confidence": orientation_conf,
            "orientation_value": orientation_value,
            "confidence": gesture_conf,
            "joints": self.tracker.get_latest_positions(),
        }
        self.publisher(payload)
        if self.step_size > 0:
            while len(self.window) > self.step_size:
                self.window.popleft()




class GuidedSessionManager:
    """Runs guided capture/training and emits PIPELINE events."""


    def __init__(
        self,
        myo: MyoService,
        event_cb: Callable[[Dict[str, Any]], None],
        status_cb: Callable[[Dict[str, Any]], None],
        models_dir: Path,
        recordings_dir: Path,
    ) -> None:
        self.myo = myo
        self.event_cb = event_cb
        self.status_cb = status_cb
        self.models_dir = models_dir
        self.recordings_dir = recordings_dir
        self.recorder = SessionRecorder()
        self._listener = self.myo.register_listener(self._on_sample)
        self._recording = False
        self._task: Optional[asyncio.Task] = None
        self.session_id: Optional[str] = None
        self.state = "idle"
        self.current_step = ""


    async def start(self, session_label: Optional[str]) -> str:
        if self._task and not self._task.done():
            raise RuntimeError("Guided session already running")
        self.session_id = session_label or f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        self.recorder.start(self.session_id)
        self._task = asyncio.create_task(self._run_sequence())
        self._push_status()
        return self.session_id


    async def abort(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


    def status(self) -> Dict[str, Any]:
        return {
            "session": self.session_id or "",
            "state": self.state,
            "step": self.current_step,
        }


    def _push_status(self) -> None:
        if self.status_cb:
            self.status_cb(self.status())


    async def _run_sequence(self) -> None:
        try:
            await self.myo.acquire_stream()
            steps = build_guided_sequence()
            self.state = "countdown"
            self._push_status()
            self.event_cb({"type": "session_started", "session": self.session_id, "total": len(steps)})
            for idx, step in enumerate(steps, start=1):
                self.current_step = step.name
                self._push_status()
                self.event_cb(
                    {
                        "type": "step_prompt",
                        "session": self.session_id,
                        "index": idx,
                        "total": len(steps),
                        "step": step.name,
                        "prompt": step.prompt,
                        "duration": step.duration,
                    }
                )
                await asyncio.sleep(PREP_SECONDS)
                self._begin_step(step)
                await self._run_segment(step)
                self._end_step()
                if step.rest_after > 0:
                    self.state = "rest"
                    self._push_status()
                    await asyncio.sleep(step.rest_after)
            self.state = "processing"
            self._push_status()
            self.event_cb({"type": "recording_complete", "session": self.session_id})
            await self._run_training()
            self.event_cb({"type": "session_complete", "session": self.session_id})
            self.recorder.stop()
        except asyncio.CancelledError:
            self._recording = False
            self.recorder.stop()
            self.event_cb({"type": "session_cancelled", "session": self.session_id})
            raise
        except Exception as exc:
            self._recording = False
            self.recorder.stop()
            self.event_cb({"type": "session_error", "session": self.session_id, "error": str(exc)})
        finally:
            await self.myo.release_stream()
            self._task = None
            self.state = "idle"
            self.current_step = ""
            self._push_status()


    def _begin_step(self, step: RecordingStep) -> None:
        self.state = "recording"
        self._push_status()
        self.recorder.begin_step(step.metadata)
        self._recording = True
        self.event_cb({"type": "segment_started", "session": self.session_id, "step": step.name, "duration": step.duration})


    async def _run_segment(self, step: RecordingStep) -> None:
        start = time.perf_counter()
        while True:
            elapsed = time.perf_counter() - start
            self.event_cb(
                {
                    "type": "segment_tick",
                    "session": self.session_id,
                    "step": step.name,
                    "elapsed": elapsed,
                    "duration": step.duration,
                }
            )
            if elapsed >= step.duration:
                break
            await asyncio.sleep(0.5)


    def _end_step(self) -> None:
        self._recording = False
        self.recorder.end_step()
        self.event_cb({"type": "segment_completed", "session": self.session_id, "step": self.current_step})


    async def _run_training(self) -> None:
        df = self.recorder.to_dataframe()
        if df.empty:
            raise RuntimeError("No EMG samples captured")
        csv_path = self.recordings_dir / f"{self.session_id}_samples.csv"
        df.to_csv(csv_path, index=False)
        features = build_feature_table(df)
        if features.empty:
            raise RuntimeError("Feature extraction produced empty table")
        bundle, metrics = train_models(features)
        manifest = build_manifest(self.session_id, bundle)
        artifacts = save_artifacts(self.session_id, bundle, manifest, metrics, self.models_dir)
        self.event_cb(
            {
                "type": "training_complete",
                "session": self.session_id,
                "model": artifacts.bundle_path.stem,
                "metrics": make_json_safe(metrics),
            }
        )


    def _on_sample(self, timestamp_ms: int, sample: List[int]) -> None:
        if self._recording:
            self.recorder.add_sample(timestamp_ms, sample)




class BridgeServer:
    """Central coordinator for TCP clients, Myo BLE, guided sessions, and AprilTags."""


    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.clients: Set[ClientConnection] = set()
        self.myo = MyoService(loop)
        self.tracker = AprilTagTracker()
        self.inference = InferenceEngine(self.myo, self.tracker, self.broadcast_prediction)
        self.guided = GuidedSessionManager(
            self.myo,
            self.broadcast_pipeline_event,
            self.broadcast_pipeline_status,
            MODELS_DIR,
            RECORDINGS_DIR,
        )


    async def start(self, host: str, port: int) -> None:
        server = await asyncio.start_server(self._accept_client, host, port)
        sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets)
        print("=" * 60)
        print(f"VR Bridge Server listening on {sockets}")
        print("Ready: BLE · Guided pipeline · AprilTag · Inference")
        print("=" * 60)
        async with server:
            await server.serve_forever()


    async def _accept_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        conn = ClientConnection(reader, writer)
        self.clients.add(conn)
        print(f"[CLIENT] Connected: {conn.peer}")
        try:
            while True:
                data = await reader.readline()
                if not data:
                    break
                line = data.decode("utf-8").strip()
                if not line:
                    continue
                await self._handle_command(conn, line)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            self.clients.discard(conn)
            await conn.close()
            print(f"[CLIENT] Disconnected: {conn.peer}")


    async def _handle_command(self, conn: ClientConnection, line: str) -> None:
        cmd, arg = (line.split(",", 1) + [""])[:2]
        cmd = cmd.strip().upper()
        arg = arg.strip()
        try:
            if cmd == "PING":
                await conn.send_line("ACK PING")
            elif cmd == "LIST_MODELS":
                models = self.inference.list_models()
                payload = ",".join(models) if models else "NONE"
                await conn.send_line(f"MODELS {payload}")
            elif cmd == "SET_MODEL":
                name = arg or ""
                if not name:
                    raise RuntimeError("MODEL_NAME_REQUIRED")
                try:
                    model_name = await self.inference.load_model(name)
                    await conn.send_line(f"ACK MODEL_SET {model_name}")
                except FileNotFoundError:
                    await conn.send_line(f"ERROR MODEL_NOT_FOUND {name}")
            elif cmd == "START_INFERENCE":
                await self.inference.start()
                await conn.send_line("ACK INFERENCE_STARTED")
            elif cmd == "STOP_INFERENCE":
                await self.inference.stop()
                await conn.send_line("ACK INFERENCE_STOPPED")
            elif cmd == "SCAN_MYO":
                devices = await self.myo.scan()
                summary = ",".join(
                    f"{d['address']}:{d.get('name','Myo')}:{d.get('rssi','')}" for d in devices
                ) or "NONE"
                await conn.send_line(f"MYO_DEVICES {summary}")
                await conn.send_line(f"MYO_DEVICES_LIST {json.dumps(devices)}")
            elif cmd == "CONNECT_MYO":
                client = await self.myo.connect(arg or None)
                battery = await client.read_battery_level()
                battery_str = battery if battery is not None else -1
                await conn.send_line(f"MYO_CONNECTED {client.address}:{client.name}:{battery_str}")
                await conn.send_line(f"MYO_STATUS {json.dumps(self.myo.status())}")
            elif cmd == "DISCONNECT_MYO":
                target = await self.myo.disconnect(arg or None)
                target_str = target or ""
                await conn.send_line(f"MYO_DISCONNECTED {target_str}")
            elif cmd == "LIST_MYO_DEVICES":
                await conn.send_line(f"MYO_DEVICES_LIST {json.dumps(self.myo.status())}")
            elif cmd == "GET_MYO_STATUS":
                await conn.send_line(f"MYO_STATUS {json.dumps(self.myo.status())}")
            elif cmd == "VIBRATE_MYO":
                if not arg:
                    await conn.send_line("ERROR MYO_ADDRESS_REQUIRED")
                else:
                    client = self.myo._active_client
                    if client and client.address == arg:
                        await client.vibrate()
                        await conn.send_line("ACK MYO_VIBRATE")
                    else:
                        await conn.send_line("ERROR MYO_NOT_CONNECTED")
            elif cmd == "GUIDED_START":
                session_id = await self.guided.start(arg or None)
                await conn.send_line(f"ACK GUIDED_START {session_id}")
            elif cmd == "GUIDED_ABORT":
                await self.guided.abort()
                await conn.send_line("ACK GUIDED_ABORT")
            elif cmd == "GUIDED_STATUS":
                await conn.send_line(f"PIPELINE_STATUS {json.dumps(self.guided.status())}")
            elif cmd == "START_APRILTAG":
                self.tracker.start()
                await conn.send_line("ACK APRILTAG_STARTED")
            elif cmd == "STOP_APRILTAG":
                self.tracker.stop()
                await conn.send_line("ACK APRILTAG_STOPPED")
            elif cmd == "GET_APRILTAG_STATUS":
                await conn.send_line(f"APRILTAG_STATUS {json.dumps(self.tracker.status())}")
            elif cmd == "ADD_APRILTAG_JOINT":
                tag_id, offset = (arg.split(",") + ["0"])[:2]
                self.tracker.add_joint(int(tag_id), float(offset))
                await conn.send_line(f"ACK APRILTAG_JOINT_ADDED {tag_id}")
            elif cmd == "REMOVE_APRILTAG_JOINT":
                self.tracker.remove_joint(int(arg))
                await conn.send_line(f"ACK APRILTAG_JOINT_REMOVED {arg}")
            elif cmd == "CLEAR_APRILTAG_JOINTS":
                self.tracker.clear_joints()
                await conn.send_line("ACK APRILTAG_JOINTS_CLEARED")
            elif cmd == "LIST_APRILTAG_JOINTS":
                await conn.send_line(f"APRILTAG_JOINTS {json.dumps(self.tracker.get_tracked_joints())}")
            elif cmd == "GET_JOINT_POSITIONS":
                payload = {
                    "type": "joint_positions",
                    "timestamp": time.time(),
                    "joints": self.tracker.get_latest_positions(),
                }
                await conn.send_line(f"JOINT_POSITIONS {json.dumps(payload)}")
            else:
                await conn.send_line(f"ERROR UNKNOWN_COMMAND {cmd}")
        except Exception as exc:
            await conn.send_line(f"ERROR {cmd}_FAILED {exc}")


    def broadcast_line(self, line: str) -> None:
        for conn in list(self.clients):
            asyncio.create_task(conn.send_line(line))


    def broadcast_prediction(self, payload: Dict[str, Any]) -> None:
        self.broadcast_line(f"PREDICTION_DATA {json.dumps(payload)}")


    def broadcast_pipeline_event(self, payload: Dict[str, Any]) -> None:
        self.broadcast_line(f"PIPELINE {json.dumps(payload)}")


    def broadcast_pipeline_status(self, payload: Dict[str, Any]) -> None:
        self.broadcast_line(f"PIPELINE_STATUS {json.dumps(payload)}")




async def run_server(host: str, port: int) -> None:
    loop = asyncio.get_running_loop()
    server = BridgeServer(loop)
    await server.start(host, port)




def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VR + Myo bridge server")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args(argv)




def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_server(args.host, args.port))
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped")


if __name__ == "__main__":
    main()



