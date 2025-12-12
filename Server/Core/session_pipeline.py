#!/usr/bin/env python3
"""Reusable utilities for guided Myo EMG recording and training.


This module now mirrors the simplified demo workflow (rest/open/clench ×
pronation/neutral/supination). It exposes thread-safe helpers so the network
server can orchestrate guided sessions, extract features, and train models that
emit a 1D gesture blend plus a forearm rotation value alongside readable
labels.
"""


from __future__ import annotations


from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import threading


import joblib
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ------------------------------- Constants -------------------------------- #


WINDOW_SIZE = 20
STEP_SIZE = 5
SAMPLING_RATE = 100  # Hz
PREP_SECONDS = 3
REST_SECONDS = 2
RECORD_SECONDS = 5
REPETITIONS = 5


EMG_CHANNELS = 16
EMG_COLUMNS = [f"emg_{i+1:02d}" for i in range(EMG_CHANNELS)]
FEATURE_SUFFIXES = [
    "rms",
    "mav",
    "wl",
    "zc",
    "ssc",
    "fft_mean",
    "fft_max",
    "mean_freq",
    "median_freq",
]
FEATURE_NAMES = [
    f"{channel}_{suffix}"
    for channel in EMG_COLUMNS
    for suffix in FEATURE_SUFFIXES
]


GESTURE_LABELS = ["rest", "open", "clench"]
ORIENTATION_LABELS = ["pronation", "neutral", "supination"]
GESTURE_PROMPTS = {
    "rest": "Relax your hand",
    "open": "Open your palm wide",
    "clench": "Make a tight fist",
}
ORIENTATION_PHRASES = {
    "pronation": "pronation (palm down)",
    "neutral": "neutral",
    "supination": "supination (palm up)",
}
GESTURE_BLEND_MAP: Dict[str, float] = {
    "rest": 0.0,
    "open": 1.0,
    "clench": -1.0,
}
ORIENTATION_VALUE_MAP: Dict[str, float] = {
    "pronation": -1.0,
    "neutral": 0.0,
    "supination": 1.0,
}


FEATURES_SUMMARY = FEATURE_NAMES


DEFAULT_METADATA = {
    "gesture_label": "rest",
    "orientation_label": "neutral",
    "gesture_target": 0.0,
    "orientation_target": 0.0,
    "segment_type": "gesture",
}


# ---------------------------- Data Structures ----------------------------- #


@dataclass
class RecordingStep:
    name: str
    prompt: str
    duration: float
    metadata: Dict[str, Any]
    rest_after: float = REST_SECONDS




@dataclass
class TrainingArtifacts:
    session_id: str
    bundle_path: Path
    manifest_path: Path
    metrics_path: Path
    metrics: Dict[str, Any]




class SessionRecorder:
    """Thread-safe buffer that collects EMG samples with metadata."""


    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: List[Dict[str, Any]] = []
        self._metadata: Optional[Dict[str, Any]] = None
        self._active = False
        self.session_id: Optional[str] = None


    # -------- State management -------- #


    def start(self, session_id: str) -> None:
        with self._lock:
            self._rows.clear()
            self._metadata = None
            self._active = True
            self.session_id = session_id


    def begin_step(self, metadata: Dict[str, Any]) -> None:
        with self._lock:
            self._metadata = dict(metadata)


    def end_step(self) -> None:
        with self._lock:
            self._metadata = None


    def stop(self) -> None:
        with self._lock:
            self._active = False
            self._metadata = None


    # -------- Data capture -------- #


    def add_sample(self, timestamp_ms: int, sample: Iterable[int]) -> None:
        with self._lock:
            if not self._active or self._metadata is None:
                return
            sample_list = list(sample)
            if len(sample_list) != EMG_CHANNELS:
                return
            row = {"timestamp_ms": int(timestamp_ms), **self._metadata}
            for idx, value in enumerate(sample_list):
                row[EMG_COLUMNS[idx]] = int(value)
            row["emg_hex"] = "".join(f"{(val & 0xFF):02x}" for val in sample_list)
            self._rows.append(row)


    def to_dataframe(self) -> pd.DataFrame:
        with self._lock:
            rows = list(self._rows)
        return pd.DataFrame(rows)


    def has_data(self) -> bool:
        with self._lock:
            return bool(self._rows)




# ----------------------------- Guided Steps ------------------------------- #


def build_guided_sequence() -> List[RecordingStep]:
    steps: List[RecordingStep] = []
    for rep in range(1, REPETITIONS + 1):
        for orientation in ORIENTATION_LABELS:
            orientation_phrase = ORIENTATION_PHRASES.get(orientation, orientation)
            for gesture in GESTURE_LABELS:
                gesture_prompt = GESTURE_PROMPTS.get(gesture, f"Perform {gesture}")
                if gesture == "rest":
                    prompt = f"Relax in {orientation_phrase} for {RECORD_SECONDS}s."
                else:
                    prompt = f"{gesture_prompt} while holding {orientation_phrase} for {RECORD_SECONDS}s."
                metadata = {
                    **DEFAULT_METADATA,
                    "gesture_label": gesture,
                    "orientation_label": orientation,
                    "gesture_target": GESTURE_BLEND_MAP.get(gesture, 0.0),
                    "orientation_target": ORIENTATION_VALUE_MAP.get(orientation, 0.0),
                }
                steps.append(
                    RecordingStep(
                        name=f"{gesture}_{orientation}_rep{rep}",
                        prompt=prompt,
                        duration=RECORD_SECONDS,
                        metadata=metadata,
                    )
                )
    return steps




# --------------------------- Feature Extraction --------------------------- #


def compute_channel_features(signal: np.ndarray) -> List[float]:
    """Compute handcrafted features for a single EMG channel."""
    sig = signal.astype(np.float32)
    if sig.size == 0:
        return [0.0] * len(FEATURE_SUFFIXES)


    rms = float(np.sqrt(np.mean(sig ** 2)))
    mav = float(np.mean(np.abs(sig)))
    wl = float(np.sum(np.abs(np.diff(sig))))
    zc = float(np.sum((sig[:-1] * sig[1:]) < 0)) if sig.size > 1 else 0.0
    if sig.size >= 3:
        diff1 = sig[1:-1] - sig[:-2]
        diff2 = sig[1:-1] - sig[2:]
        ssc = float(np.sum((diff1 * diff2) > 0))
    else:
        ssc = 0.0


    fft_vals = np.abs(np.fft.rfft(sig))
    fft_mean = float(np.mean(fft_vals)) if fft_vals.size else 0.0
    fft_max = float(np.max(fft_vals)) if fft_vals.size else 0.0


    try:
        freqs, power = welch(sig, fs=SAMPLING_RATE, nperseg=min(len(sig), 64))
        power_sum = float(np.sum(power))
        if power_sum > 1e-9:
            mean_freq = float(np.sum(freqs * power) / power_sum)
            cumulative = np.cumsum(power)
            median_idx = int(np.searchsorted(cumulative, power_sum / 2.0))
            median_freq = float(freqs[min(median_idx, len(freqs) - 1)])
        else:
            mean_freq = 0.0
            median_freq = 0.0
    except Exception:
        mean_freq = 0.0
        median_freq = 0.0


    return [
        rms,
        mav,
        wl,
        zc,
        ssc,
        fft_mean,
        fft_max,
        mean_freq,
        median_freq,
    ]




def compute_feature_vector(window: np.ndarray) -> List[float]:
    features: List[float] = []
    for ch in range(EMG_CHANNELS):
        features.extend(compute_channel_features(window[:, ch]))
    return features




def most_common(series: pd.Series) -> Any:
    counts = series.value_counts()
    if counts.empty:
        return None
    return counts.idxmax()




def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()


    feature_rows: List[List[float]] = []
    meta_rows: List[Dict[str, Any]] = []
    for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        window_df = df.iloc[start:end]
        window = window_df[EMG_COLUMNS].values.astype(np.float32)
        features = compute_feature_vector(window)
        feature_rows.append(features)
        meta_rows.append(
            {
                "gesture_label": most_common(window_df["gesture_label"]),
                "orientation_label": most_common(window_df["orientation_label"]),
                "segment_type": most_common(window_df["segment_type"]),
                "gesture_target": float(window_df["gesture_target"].mean()),
                "orientation_target": float(window_df["orientation_target"].mean()),
            }
        )


    if not feature_rows:
        return pd.DataFrame()


    feature_df = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    meta_df = pd.DataFrame(meta_rows)
    return pd.concat([feature_df, meta_df], axis=1)




# ------------------------------- Training -------------------------------- #


def train_models(feature_df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    feature_cols = FEATURE_NAMES
    X_full = feature_df[feature_cols].values
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)


    metrics: Dict[str, Any] = {
        "window_count": int(len(feature_df)),
    }


    def train_classifier(y: np.ndarray, label_names: List[str], name: str) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        stats: Dict[str, Any] = {}
        unique = np.unique(y)
        if len(unique) > 1 and len(y) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X_full_scaled,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            stats["accuracy"] = float(acc)
            stats["report"] = classification_report(
                y_test,
                y_pred,
                target_names=label_names,
                zero_division=0,
                output_dict=True,
            )
        else:
            clf.fit(X_full_scaled, y)
            stats["accuracy"] = float(clf.score(X_full_scaled, y))
            stats["note"] = "Trained on entire dataset (insufficient samples for split)."
        stats["samples"] = int(len(y))
        stats["classes"] = int(len(unique))
        metrics[f"{name}_classifier"] = stats
        return clf, stats


    gesture_encoder = LabelEncoder()
    gesture_labels = feature_df["gesture_label"].fillna("rest").astype(str)
    gesture_y = gesture_encoder.fit_transform(gesture_labels)
    gesture_clf, _ = train_classifier(gesture_y, list(gesture_encoder.classes_), "gesture")


    orientation_encoder = LabelEncoder()
    orientation_labels = feature_df["orientation_label"].fillna("neutral").astype(str)
    orientation_y = orientation_encoder.fit_transform(orientation_labels)
    orientation_clf, _ = train_classifier(orientation_y, list(orientation_encoder.classes_), "orientation")


    gesture_reg = RandomForestRegressor(n_estimators=200, random_state=42)
    orientation_reg = RandomForestRegressor(n_estimators=200, random_state=42)
    metrics["regressors"] = {
        "gesture_blend": fit_regressor(gesture_reg, X_full_scaled, feature_df["gesture_target"].values),
        "orientation_value": fit_regressor(orientation_reg, X_full_scaled, feature_df["orientation_target"].values),
    }


    bundle = {
        "gesture_classifier": gesture_clf,
        "gesture_encoder": gesture_encoder,
        "orientation_classifier": orientation_clf,
        "orientation_encoder": orientation_encoder,
        "gesture_reg": gesture_reg,
        "orientation_reg": orientation_reg,
        "gesture_blend_map": dict(GESTURE_BLEND_MAP),
        "orientation_value_map": dict(ORIENTATION_VALUE_MAP),
        "scaler": scaler,
        "feature_names": feature_cols,
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
    }


    return bundle, metrics




def fit_regressor(
    regressor: RandomForestRegressor,
    features: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if len(np.unique(targets)) <= 1:
        regressor.fit(features, targets)
        result["mae"] = 0.0
        result["note"] = "Target constant; trained without validation."
        return result


    X_train, X_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.2,
        random_state=42,
    )
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    result["mae"] = float(mae)
    return result




# ------------------------------- Manifest -------------------------------- #


def build_manifest(session_id: str, bundle: Dict[str, Any]) -> Dict[str, Any]:
    gesture_encoder: LabelEncoder = bundle["gesture_encoder"]
    orientation_encoder: LabelEncoder = bundle["orientation_encoder"]
    gestures = [
        {
            "label": label,
            "blend": float(bundle["gesture_blend_map"].get(label, 0.0)),
        }
        for label in gesture_encoder.classes_
    ]
    orientations = [
        {
            "label": label,
            "value": float(bundle["orientation_value_map"].get(label, 0.0)),
        }
        for label in orientation_encoder.classes_
    ]
    return {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "sampling_rate": SAMPLING_RATE,
        "gestures": gestures,
        "orientations": orientations,
        "feature_names": FEATURES_SUMMARY,
    }




# ------------------------------- Persistence ------------------------------ #


def save_artifacts(
    session_id: str,
    bundle: Dict[str, Any],
    manifest: Dict[str, Any],
    metrics: Dict[str, Any],
    models_dir: Path,
) -> TrainingArtifacts:
    models_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = models_dir / f"session_{session_id}.pkl"
    manifest_path = models_dir / f"session_{session_id}.json"
    metrics_path = models_dir / f"session_{session_id}_metrics.json"


    joblib.dump(bundle, bundle_path)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    metrics_path.write_text(json.dumps(metrics, indent=2))


    return TrainingArtifacts(
        session_id=session_id,
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        metrics=metrics,
    )




__all__ = [
    "WINDOW_SIZE",
    "STEP_SIZE",
    "SAMPLING_RATE",
    "PREP_SECONDS",
    "REST_SECONDS",
    "RECORD_SECONDS",
    "REPETITIONS",
    "EMG_CHANNELS",
    "EMG_COLUMNS",
    "GESTURE_LABELS",
    "ORIENTATION_LABELS",
    "GESTURE_PROMPTS",
    "ORIENTATION_PHRASES",
    "GESTURE_BLEND_MAP",
    "ORIENTATION_VALUE_MAP",
    "RecordingStep",
    "TrainingArtifacts",
    "SessionRecorder",
    "build_guided_sequence",
    "compute_feature_vector",
    "build_feature_table",
    "train_models",
    "build_manifest",
    "save_artifacts",
]



