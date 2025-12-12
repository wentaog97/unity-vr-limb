#!/usr/bin/env python3
"""Guided Myo EMG recorder + trainer.

Running this script walks you through the same guided capture flow that the
network server uses (rest/open/clench × pronation/neutral/supination) and then
trains a model bundle from the freshly recorded samples. You can optionally
augment the session with existing CSV recordings via ``-p/--path`` flags.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from Core.session_pipeline import (
    EMG_COLUMNS,
    PREP_SECONDS,
    RecordingStep,
    SessionRecorder,
    build_feature_table,
    build_guided_sequence,
    build_manifest,
    save_artifacts,
    train_models,
)
from myo_ble_client import MyoBleClient


REQUIRED_COLUMNS = {
    "timestamp_ms",
    "gesture_label",
    "orientation_label",
    "gesture_target",
    "orientation_target",
    "segment_type",
    "emg_hex",
    *EMG_COLUMNS,
}

NUMERIC_COLUMNS = [
    "timestamp_ms",
    "gesture_target",
    "orientation_target",
    *EMG_COLUMNS,
]

DEFAULT_MODELS_DIR = Path("models")
DEFAULT_RECORDINGS_DIR = Path("recordings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a guided Myo capture session and train a model bundle.",
    )
    parser.add_argument(
        "recordings",
        nargs="*",
        help="Optional positional paths/globs to merge with the guided session.",
    )
    parser.add_argument(
        "-p",
        "--path",
        dest="path_args",
        action="append",
        help="Additional recording path or glob (repeat as needed).",
    )
    parser.add_argument("--session-id", help="Override the session identifier.")
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help="Directory for trained artifacts (default: models/).",
    )
    parser.add_argument(
        "--recordings-dir",
        default=str(DEFAULT_RECORDINGS_DIR),
        help="Where to store the captured CSV (default: recordings/).",
    )
    parser.add_argument(
        "--feature-csv",
        help="Optional path to dump the engineered feature table.",
    )
    parser.add_argument(
        "--address",
        help="Myo BLE address/UUID (auto-discover when omitted).",
    )
    parser.add_argument(
        "--name",
        help="Friendly name for the armband (used when address provided).",
    )
    parser.add_argument(
        "--prep-seconds",
        type=float,
        default=PREP_SECONDS,
        help="Seconds to count down before each segment (default matches server).",
    )
    parser.add_argument(
        "--skip-capture",
        action="store_true",
        help="Do not run the guided session (requires --path/recordings inputs).",
    )
    args = parser.parse_args()
    paths: List[str] = []
    if args.path_args:
        paths.extend(args.path_args)
    if args.recordings:
        paths.extend(args.recordings)
    args.recordings = paths
    if args.skip_capture and not args.recordings:
        parser.error("--skip-capture requires at least one --path or positional recording.")
    return args


def expand_recording_paths(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        matched = list(Path().glob(pattern))
        if matched:
            paths.extend(matched)
        else:
            candidate = Path(pattern)
            if candidate.exists():
                paths.append(candidate)
    unique = sorted({p.resolve() for p in paths})
    if not unique:
        raise FileNotFoundError("No recording files matched the provided inputs.")
    return list(unique)


def derive_session_id(recording_paths: List[Path]) -> str:
    stem = recording_paths[0].stem
    if stem.endswith("_samples"):
        stem = stem[:-len("_samples")]
    return stem


def load_recordings(recording_paths: List[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in recording_paths:
        df = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Recording {path} missing columns: {sorted(missing)}")
        df = df.copy()
        for col in NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=NUMERIC_COLUMNS)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise ValueError("No valid samples remained after cleaning the recordings.")
    return combined.sort_values("timestamp_ms").reset_index(drop=True)


def summarize_recordings(df: pd.DataFrame) -> str:
    gestures = df["gesture_label"].value_counts().to_dict()
    orientations = df["orientation_label"].value_counts().to_dict()
    return (
        f"Loaded {len(df):,} samples across {len(gestures)} gestures and "
        f"{len(orientations)} orientations."
    )


async def countdown(label: str, seconds: float) -> None:
    if seconds <= 0:
        return
    remaining = int(round(seconds))
    for tick in range(remaining, 0, -1):
        print(f"{label}: {tick:02d}s", end="\r", flush=True)
        await asyncio.sleep(1)
    print(" " * 40, end="\r")


async def run_segment(step: RecordingStep, stop_event: asyncio.Event) -> None:
    start = time.perf_counter()
    while not stop_event.is_set():
        elapsed = time.perf_counter() - start
        if elapsed >= step.duration:
            break
            remaining = max(0.0, step.duration - elapsed)
            print(f"Recording... {remaining:4.1f}s left", end="\r", flush=True)
            await asyncio.sleep(0.5)
    print(" " * 40, end="\r")


async def rest_period(seconds: float, stop_event: asyncio.Event) -> None:
    if seconds <= 0 or stop_event.is_set():
        return
    for tick in range(int(round(seconds)), 0, -1):
        if stop_event.is_set():
            break
        print(f"Rest... {tick:02d}s", end="\r", flush=True)
        await asyncio.sleep(1)
    print(" " * 40, end="\r")


async def run_guided_capture(args: argparse.Namespace) -> Tuple[pd.DataFrame, Path, str]:
    recordings_dir = Path(args.recordings_dir).expanduser().resolve()
    recordings_dir.mkdir(parents=True, exist_ok=True)
    session_id = args.session_id or f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    recorder = SessionRecorder()
    recorder.start(session_id)
    client = MyoBleClient(address=args.address, name=args.name)
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def handle_sample(sample, _client):
        timestamp_ms = int(time.time() * 1000)
        recorder.add_sample(timestamp_ms, sample)

    client.set_emg_callback(handle_sample)

    def request_stop() -> None:
        if not stop_event.is_set():
            print("\nStopping guided session...", file=sys.stderr)
            stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError:  # pragma: no cover (Windows)
            signal.signal(sig, lambda *_: request_stop())

    print(f"Connecting to Myo ({args.address or 'auto-discover'})...")
    if not await client.connect():
        raise RuntimeError("Failed to connect to Myo armband.")
    await client.start_streaming(emg=True, imu=False)

    steps = build_guided_sequence()
    print(f"Capturing guided session '{session_id}' ({len(steps)} segments)...")
    for idx, step in enumerate(steps, start=1):
        if stop_event.is_set():
            break
        print(
            f"\n[{idx}/{len(steps)}] {step.name}\n{step.prompt}"
        )
        await countdown("Prep", args.prep_seconds)
        if stop_event.is_set():
            break
        recorder.begin_step(step.metadata)
        await run_segment(step, stop_event)
        recorder.end_step()
        if step.rest_after > 0:
            await rest_period(step.rest_after, stop_event)

    recorder.stop()
    print("\nCapture complete. Cleaning up...")
    if client.is_streaming:
        await client.stop_streaming()
    if client.is_connected:
        await client.disconnect()

    df = recorder.to_dataframe()
    if df.empty:
        raise RuntimeError("No EMG samples were captured. Try rerunning the session.")
    print(summarize_recordings(df))
    csv_path = recordings_dir / f"{session_id}_samples.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved raw samples to {csv_path}")
    return df, csv_path, session_id


def train_bundle(df: pd.DataFrame, session_id: str, models_dir: Path, feature_csv: Optional[str]) -> None:
    feature_df = build_feature_table(df)
    if feature_df.empty:
        raise RuntimeError("Feature extraction produced no windows. Collect more data.")
    print(f"Generated {len(feature_df):,} feature windows.")
    bundle, metrics = train_models(feature_df)
    artifacts = save_artifacts(
        session_id=session_id,
        bundle=bundle,
        manifest=build_manifest(session_id, bundle),
        metrics=metrics,
        models_dir=models_dir,
    )
    if feature_csv:
        feature_path = Path(feature_csv)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(feature_path, index=False)
        print(f"Feature table exported to {feature_path}")
    print("Training complete!")
    print(f"Session ID: {session_id}")
    print(f"Model bundle: {artifacts.bundle_path}")
    print(f"Manifest: {artifacts.manifest_path}")
    print(f"Metrics: {artifacts.metrics_path}")
    for key in ("gesture_classifier", "orientation_classifier"):
        stats = metrics.get(key)
        if stats and stats.get("accuracy") is not None:
            print(f"{key.replace('_', ' ').title()} accuracy: {stats['accuracy']:.3f}")


def main() -> None:
    args = parse_args()
    guided_df: Optional[pd.DataFrame] = None
    guided_session_id: Optional[str] = None
    additional_paths: List[Path] = []
    if not args.skip_capture:
        try:
            guided_df, _, guided_session_id = asyncio.run(run_guided_capture(args))
        except Exception as exc:
            print(f"Guided session failed: {exc}", file=sys.stderr)
            sys.exit(1)
    if args.recordings:
        try:
            additional_paths = expand_recording_paths(args.recordings)
        except FileNotFoundError as exc:
            print(exc)
            sys.exit(1)

    frames: List[pd.DataFrame] = []
    session_hint: Optional[str] = guided_session_id
    if guided_df is not None:
        frames.append(guided_df)
    if additional_paths:
        try:
            extra_df = load_recordings(additional_paths)
        except ValueError as exc:
            print(f"Error while loading extra recordings: {exc}")
            sys.exit(1)
        frames.append(extra_df)
        if not session_hint:
            session_hint = derive_session_id(additional_paths)
    if not frames:
        print("No data was captured or provided. Nothing to train.", file=sys.stderr)
        sys.exit(1)

    combined_df = pd.concat(frames, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)
    print(summarize_recordings(combined_df))
    session_id = args.session_id or session_hint or f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    models_dir = Path(args.models_dir).expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    try:
        train_bundle(combined_df, session_id, models_dir, args.feature_csv)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
