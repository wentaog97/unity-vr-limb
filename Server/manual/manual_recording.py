#!/usr/bin/env python3
"""Guided Myo EMG recorder.

Runs the same 5-second prompt schedule (rest/open/clench × pronation/neutral/
supination) that the network server uses, inserts short rest gaps between
segments, and saves the captured session to ``recordings/``.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from Core.session_pipeline import (
    PREP_SECONDS,
    REST_SECONDS,
    RecordingStep,
    SessionRecorder,
    build_guided_sequence,
)
from myo_ble_client import MyoBleClient


DEFAULT_RECORDINGS_DIR = Path("recordings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Guide the user through rest/open/clench across all orientations and record EMG samples.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_RECORDINGS_DIR),
        help="Directory for the resulting *_samples.csv file (default: recordings/).",
    )
    parser.add_argument("--session-id", help="Optional session identifier (default: timestamped).")
    parser.add_argument(
        "--address",
        help="Specific Myo BLE address/UUID. Auto-discover when omitted.",
    )
    parser.add_argument(
        "--name",
        help="Friendly name shown when connecting to the armband.",
    )
    parser.add_argument(
        "--prep-seconds",
        type=float,
        default=PREP_SECONDS,
        help="Countdown duration before each segment (default mirrors server).",
    )
    parser.add_argument(
        "--rest-seconds",
        type=float,
        default=REST_SECONDS,
        help="Rest duration between segments (default mirrors server).",
    )
    return parser.parse_args()


def next_session_id(custom: Optional[str]) -> str:
    if custom:
        return custom
    return f"myo_raw_session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"


async def countdown(label: str, seconds: float) -> None:
    if seconds <= 0:
        return
    for tick in range(int(round(seconds)), 0, -1):
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
    if seconds <= 0:
        return
    for tick in range(int(round(seconds)), 0, -1):
        if stop_event.is_set():
            break
        print(f"Rest... {tick:02d}s", end="\r", flush=True)
        await asyncio.sleep(1)
    print(" " * 40, end="\r")


async def record_guided_session(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = next_session_id(args.session_id)
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
            print("\nStopping guided recording...", file=sys.stderr)
            stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except NotImplementedError:  # pragma: no cover (Windows fallback)
            signal.signal(sig, lambda *_: request_stop())

    print(f"Connecting to Myo ({args.address or 'auto-discover'})...")
    if not await client.connect():
        raise RuntimeError("Failed to connect to Myo armband.")
    await client.start_streaming(emg=True, imu=False)

    steps: List[RecordingStep] = build_guided_sequence()
    for step in steps:
        step.rest_after = max(0.0, args.rest_seconds)
    print(
        f"Guided capture '{session_id}': {len(steps)} segments · 5s each + {args.rest_seconds}s rest"
    )

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

    df: pd.DataFrame = recorder.to_dataframe()
    if df.empty:
        raise RuntimeError("No EMG samples recorded. Try rerunning the session.")
    csv_path = output_dir / f"{session_id}_samples.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df):,} samples to {csv_path}")
    return csv_path


def main() -> None:
    args = parse_args()
    try:
        csv_path = asyncio.run(record_guided_session(args))
        print(f"Recording complete: {csv_path}")
    except KeyboardInterrupt:  # pragma: no cover - CLI convenience
        print("\nInterrupted. Cleaning up...", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
