#!/usr/bin/env python3
"""Manual real-time inference runner for Myo EMG streams."""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from pathlib import Path
import signal
import sys
import time
from typing import Any, Dict, Tuple

import joblib
import numpy as np

from Core.session_pipeline import (
    GESTURE_BLEND_MAP,
    ORIENTATION_VALUE_MAP,
    STEP_SIZE,
    WINDOW_SIZE,
    compute_feature_vector,
)
from myo_ble_client import MyoBleClient


DEFAULT_MODELS_DIR = Path("models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream EMG data from a Myo armband and run live inference using a trained model bundle.",
    )
    parser.add_argument(
        "-p",
        "--path",
        required=True,
        help="Path or name of the trained model bundle (.pkl). Accepts bare session names from models/.",
    )
    parser.add_argument(
        "--address",
        help="Specific Myo BLE address/UUID. If omitted the script auto-discovers the first device.",
    )
    parser.add_argument(
        "--name",
        help="Friendly name for the armband (used when address is provided).",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        help="Override the sliding step size (defaults to bundle metadata or Core.session_pipeline.STEP_SIZE).",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.2,
        help="Minimum seconds between console updates (default: 0.2).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        help="Optional maximum duration before auto-stopping (default: run until Ctrl+C).",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Seconds to count down before inference starts (default: 3).",
    )
    return parser.parse_args()


def resolve_model_path(token: str) -> Path:
    candidate = Path(token).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    models_dir = DEFAULT_MODELS_DIR
    candidates = [models_dir / token]
    if not token.endswith(".pkl"):
        candidates.append(models_dir / f"{token}.pkl")
        candidates.append(models_dir / f"session_{token}.pkl")
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Could not resolve model path for '{token}'.")


def load_bundle(path_str: str) -> Tuple[Path, Dict[str, Any]]:
    model_path = resolve_model_path(path_str)
    bundle = joblib.load(model_path)
    return model_path, bundle


def format_prediction(payload: Dict[str, Any]) -> str:
    gesture = payload["gesture"]
    orientation = payload["orientation"]
    g_blend = payload["gesture_blend"]
    o_val = payload["orientation_value"]
    g_conf = payload["gesture_confidence"]
    o_conf = payload["orientation_confidence"]
    return (
        f"{gesture:<8} ({g_blend:+.2f}, conf={g_conf:.2f})  |  "
        f"{orientation:<10} ({o_val:+.2f}, conf={o_conf:.2f})"
    )


def predict_from_window(bundle: Dict[str, Any], window: np.ndarray) -> Dict[str, Any]:
    features = compute_feature_vector(window)
    scaler = bundle["scaler"]
    feats_scaled = scaler.transform([features])

    gesture_classifier = bundle["gesture_classifier"]
    gesture_encoder = bundle["gesture_encoder"]
    gesture_proba = gesture_classifier.predict_proba(feats_scaled)[0]
    gesture_idx = int(np.argmax(gesture_proba))
    gesture_label = gesture_encoder.inverse_transform([gesture_idx])[0]

    orientation_classifier = bundle["orientation_classifier"]
    orientation_encoder = bundle["orientation_encoder"]
    orientation_proba = orientation_classifier.predict_proba(feats_scaled)[0]
    orientation_idx = int(np.argmax(orientation_proba))
    orientation_label = orientation_encoder.inverse_transform([orientation_idx])[0]

    gesture_reg = bundle.get("gesture_reg")
    orientation_reg = bundle.get("orientation_reg")
    gesture_blend = (
        float(gesture_reg.predict(feats_scaled)[0])
        if gesture_reg is not None
        else float(GESTURE_BLEND_MAP.get(gesture_label, 0.0))
    )
    orientation_value = (
        float(orientation_reg.predict(feats_scaled)[0])
        if orientation_reg is not None
        else float(ORIENTATION_VALUE_MAP.get(orientation_label, 0.0))
    )

    return {
        "gesture": gesture_label,
        "gesture_blend": float(np.clip(gesture_blend, -1.0, 1.0)),
        "gesture_confidence": float(gesture_proba[gesture_idx]),
        "orientation": orientation_label,
        "orientation_value": float(np.clip(orientation_value, -1.0, 1.0)),
        "orientation_confidence": float(orientation_proba[orientation_idx]),
    }


async def run_inference(args: argparse.Namespace) -> None:
    model_path, bundle = load_bundle(args.path)
    window_size = int(bundle.get("window_size", WINDOW_SIZE))
    step_size = args.step_size or int(bundle.get("step_size", STEP_SIZE))

    client = MyoBleClient(address=args.address, name=args.name)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[Tuple[int, Tuple[int, ...]]] = asyncio.Queue(maxsize=window_size * 4)
    stop_event = asyncio.Event()

    def handle_sample(sample, _client):
        timestamp_ms = int(time.time() * 1000)
        try:
            loop.call_soon_threadsafe(queue.put_nowait, (timestamp_ms, tuple(int(v) for v in sample)))
        except asyncio.QueueFull:
            pass

    client.set_emg_callback(handle_sample)

    def request_stop() -> None:
        if not stop_event.is_set():
            print("\nStopping inference...", flush=True)
            stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError:  # pragma: no cover - Windows fallback
            signal.signal(sig, lambda *_: request_stop())

    print(f"Loading model: {model_path}")
    print(f"Window size: {window_size} | Step size: {step_size}")

    print(f"Connecting to Myo ({args.address or 'auto-discover'})...")
    if not await client.connect():
        raise RuntimeError("Failed to connect to Myo armband.")

    await client.start_streaming(emg=True, imu=False)

    if args.countdown > 0:
        for remaining in range(args.countdown, 0, -1):
            print(f"Starting in {remaining}s", end="\r", flush=True)
            await asyncio.sleep(1)
        print(" " * 40, end="\r")

    print("Inference running. Press Ctrl+C to stop.")
    window = deque(maxlen=window_size * 2)
    last_print = 0.0
    start_time = time.time()

    while not stop_event.is_set():
        if args.max_seconds and (time.time() - start_time) >= args.max_seconds:
            request_stop()
            break
        try:
            timestamp_ms, sample = await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        window.append(sample)
        if len(window) < window_size:
            continue
        window_arr = np.array(list(window)[-window_size:])
        payload = predict_from_window(bundle, window_arr)
        now = time.perf_counter()
        if now - last_print >= args.print_interval:
            print(
                f"t={timestamp_ms}  |  {format_prediction(payload)}",
                end="\r",
                flush=True,
            )
            last_print = now
        if step_size > 0:
            while len(window) > step_size:
                window.popleft()

    print("\nCleaning up...")
    if client.is_streaming:
        await client.stop_streaming()
    if client.is_connected:
        await client.disconnect()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_inference(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error during inference: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
