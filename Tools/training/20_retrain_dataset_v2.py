from __future__ import annotations

"""
Dataset v2 complete experiment, retraining and documentation pipeline.

Run from the repository root:

    python -m Tools.training.20_retrain_dataset_v2

Optional custom configuration:

    python -m Tools.training.20_retrain_dataset_v2 \
        --config Tools/training/dataset_v2_training_config.json

The script performs, in order:

1. Dataset integrity and acquisition-rate diagnostics.
2. Window/step search using file-based splits.
3. Decision-tree parameter search.
4. Classifier comparison using identical file splits.
5. Fisher Top-K feature-budget comparison.
6. Repeated file-holdout stability evaluation.
7. Binary-task diagnostics and direct versus hierarchical comparison.
8. Session-holdout and quadrant-holdout validation.
9. Final fit using the complete labeled Dataset v2.
10. Export of a JSON configuration and a serialized model for realtime use.
11. Generation of CSV tables, figures, an experiment index and a Markdown report.

Important methodological rule:

    Z-score parameters, correlation filtering and Fisher ranking are fitted
    only with the training files of each validation split. All windows from
    one acquisition file remain together in either training or testing.
"""

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.ndimage import median_filter, uniform_filter1d
    from sklearn.base import BaseEstimator
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, label_binarize
    from sklearn.svm import LinearSVC, SVC
    from sklearn.tree import DecisionTreeClassifier
except ImportError as exc:  # pragma: no cover - dependency message for user PC.
    raise SystemExit(
        "Missing training dependency. Install the project requirements and "
        "ensure numpy, pandas, scipy, matplotlib, scikit-learn and joblib "
        f"are available. Original error: {exc}"
    ) from exc


# ---------------------------------------------------------------------------
# Paths and imports from this repository
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
TOOLS_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = TOOLS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Tools.csi.csi_binary_io import read_packets  # noqa: E402


DEFAULT_CONFIG_PATH = THIS_FILE.with_name("dataset_v2_training_config.json")
CLASS_ORDER = ["empty", "static_presence", "movement"]
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_ORDER)}
FEATURE_NAMES_DEFAULT = [
    "mean",
    "std",
    "min",
    "max",
    "peak_to_peak",
    "energy",
    "mean_abs_diff",
    "max_abs_diff",
    "diff_energy",
    "std_diff",
    "slope",
]

LOGGER = logging.getLogger("dataset_v2_retraining")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FileRecord:
    path: Path
    label: str
    session: str
    quadrant: str
    file_name: str
    matrix: np.ndarray
    packet_count: int
    valid_packet_count: int
    subcarrier_count: int
    duration_seconds: float
    sampling_rate_hz: float
    sequence_gaps: int
    mean_rssi: float

    @property
    def group_id(self) -> str:
        return str(self.path.resolve())


@dataclass
class PreprocessingState:
    means: np.ndarray
    stds: np.ndarray
    informative_indices: np.ndarray
    selected_relative_indices: np.ndarray
    selected_subcarriers: np.ndarray


@dataclass
class FeatureSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    train_metadata: pd.DataFrame
    x_test: np.ndarray
    y_test: np.ndarray
    test_metadata: pd.DataFrame
    preprocessing: PreprocessingState
    window_packets: int
    step_packets: int


@dataclass
class EvaluationResult:
    metrics: dict[str, Any]
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray
    class_metrics: pd.DataFrame
    confusion: np.ndarray


# ---------------------------------------------------------------------------
# Configuration and logging
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain and document the complete Dataset v2 pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the JSON training configuration.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run a reduced smoke test using fewer seeds and candidates. "
            "This mode must not be used for final TCC metrics."
        ),
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Training configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported training configuration schema version.")

    return config


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_directories(config: dict[str, Any]) -> dict[str, Path]:
    output_config = config["outputs"]
    results_root = resolve_project_path(output_config["results_root"])
    processed_root = resolve_project_path(output_config["processed_root"])

    if results_root.exists() and config["execution"].get(
        "overwrite_results", True
    ):
        shutil.rmtree(results_root)

    paths = {
        "results": results_root,
        "tables": results_root / "tables",
        "figures": results_root / "figures",
        "reports": results_root / "reports",
        "logs": results_root / "logs",
        "processed": processed_root,
    }

    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths


def configure_logging(log_path: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    LOGGER.addHandler(stream_handler)
    LOGGER.addHandler(file_handler)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_dataframe(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False, encoding="utf-8")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Dataset discovery and signal preparation
# ---------------------------------------------------------------------------

def normalize_label(value: str) -> str | None:
    normalized = value.lower().strip()
    aliases = {
        "empty": "empty",
        "static": "static_presence",
        "static_presence": "static_presence",
        "presence": "static_presence",
        "movement": "movement",
    }
    return aliases.get(normalized)


def find_path_component(path: Path, prefix: str) -> str | None:
    for part in path.parts:
        lowered = part.lower()
        if lowered.startswith(prefix):
            return lowered
    return None


def infer_label(path: Path) -> str | None:
    parent_label = normalize_label(path.parent.name)
    if parent_label:
        return parent_label

    lower_name = path.stem.lower()
    for candidate in CLASS_ORDER:
        if lower_name.startswith(candidate):
            return candidate
    if lower_name.startswith("static"):
        return "static_presence"
    return None


def sequence_gap_count(sequences: Sequence[int]) -> int:
    if len(sequences) < 2:
        return 0

    gaps = 0
    previous = int(sequences[0])

    for raw_value in sequences[1:]:
        current = int(raw_value)
        difference = (current - previous) & 0xFFFFFFFF
        if 1 < difference < 0x80000000:
            gaps += difference - 1
        previous = current

    return int(gaps)


def calculate_sampling_rate(timestamps: Sequence[float]) -> tuple[float, float]:
    valid = [float(value) for value in timestamps if float(value) > 0]
    if len(valid) < 2:
        return 0.0, 0.0

    duration = valid[-1] - valid[0]
    if duration <= 0:
        return 0.0, 0.0

    return float((len(valid) - 1) / duration), float(duration)


def hampel_filter_matrix(
    matrix: np.ndarray,
    radius: int,
    n_sigmas: float,
) -> np.ndarray:
    if radius <= 0 or matrix.shape[0] < 3:
        return matrix.astype(np.float32, copy=True)

    temporal_size = 2 * radius + 1
    local_median = median_filter(
        matrix,
        size=(temporal_size, 1),
        mode="nearest",
    )
    absolute_deviation = np.abs(matrix - local_median)
    local_mad = median_filter(
        absolute_deviation,
        size=(temporal_size, 1),
        mode="nearest",
    )
    scaled_mad = 1.4826 * local_mad
    threshold = n_sigmas * scaled_mad
    replace_mask = (scaled_mad > 0) & (absolute_deviation > threshold)

    filtered = matrix.astype(np.float32, copy=True)
    filtered[replace_mask] = local_median[replace_mask]
    return filtered


def smooth_matrix(matrix: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return matrix.astype(np.float32, copy=True)

    return uniform_filter1d(
        matrix,
        size=window_size,
        axis=0,
        mode="nearest",
    ).astype(np.float32, copy=False)


def read_and_prepare_file(
    path: Path,
    config: dict[str, Any],
) -> tuple[FileRecord | None, dict[str, Any]]:
    dataset_config = config["dataset"]
    preprocessing_config = config["preprocessing"]

    session = find_path_component(path, "session")
    quadrant = find_path_component(path, "quad")
    label = infer_label(path)

    diagnostic: dict[str, Any] = {
        "path": str(path),
        "file_name": path.name,
        "session": session or "",
        "quadrant": quadrant or "",
        "label": label or "",
        "accepted": False,
        "rejection_reason": "",
    }

    if label not in set(dataset_config["labels"]):
        diagnostic["rejection_reason"] = "invalid_or_missing_label"
        return None, diagnostic

    if dataset_config.get("require_session_and_quadrant", True):
        if session is None or quadrant is None:
            diagnostic["rejection_reason"] = "missing_session_or_quadrant"
            return None, diagnostic

    try:
        packets = read_packets(path)
    except Exception as exc:  # noqa: BLE001 - preserve diagnostics per file.
        diagnostic["rejection_reason"] = f"read_error:{exc}"
        return None, diagnostic

    if not packets:
        diagnostic["rejection_reason"] = "empty_file"
        return None, diagnostic

    lengths = [
        min(len(packet.get("imag", [])), len(packet.get("real", [])))
        for packet in packets
    ]
    positive_lengths = [length for length in lengths if length > 0]

    if not positive_lengths:
        diagnostic["rejection_reason"] = "no_valid_csi_vectors"
        return None, diagnostic

    modal_length = Counter(positive_lengths).most_common(1)[0][0]
    expected_subcarriers = int(dataset_config.get("expected_subcarriers", 0))

    if expected_subcarriers > 0 and modal_length != expected_subcarriers:
        diagnostic["rejection_reason"] = (
            f"unexpected_subcarrier_count:{modal_length}"
        )
        return None, diagnostic

    rows: list[np.ndarray] = []
    accepted_packets: list[dict[str, Any]] = []

    for packet, length in zip(packets, lengths):
        if length != modal_length:
            continue

        imag = np.asarray(packet.get("imag", []), dtype=np.float32)
        real = np.asarray(packet.get("real", []), dtype=np.float32)
        amplitude = np.sqrt(imag[:modal_length] ** 2 + real[:modal_length] ** 2)

        if not np.all(np.isfinite(amplitude)):
            continue

        rows.append(amplitude)
        accepted_packets.append(packet)

    minimum_packets = int(dataset_config.get("minimum_packets_per_file", 1))
    if len(rows) < minimum_packets:
        diagnostic["rejection_reason"] = (
            f"insufficient_valid_packets:{len(rows)}"
        )
        return None, diagnostic

    matrix = np.vstack(rows).astype(np.float32, copy=False)
    matrix = hampel_filter_matrix(
        matrix,
        radius=int(preprocessing_config["hampel_radius"]),
        n_sigmas=float(preprocessing_config["hampel_n_sigmas"]),
    )
    matrix = smooth_matrix(
        matrix,
        window_size=int(preprocessing_config["moving_average_window"]),
    )

    timestamps = [
        float(
            packet.get(
                "capture_timestamp",
                packet.get("pc_timestamp", 0.0),
            )
            or 0.0
        )
        for packet in accepted_packets
    ]
    rate_hz, duration_seconds = calculate_sampling_rate(timestamps)
    sequences = [int(packet.get("sequence", 0) or 0) for packet in accepted_packets]
    rssis = [float(packet.get("rssi", 0) or 0) for packet in accepted_packets]

    record = FileRecord(
        path=path,
        label=label,
        session=session or "unknown_session",
        quadrant=quadrant or "unknown_quadrant",
        file_name=path.name,
        matrix=matrix,
        packet_count=len(packets),
        valid_packet_count=matrix.shape[0],
        subcarrier_count=matrix.shape[1],
        duration_seconds=duration_seconds,
        sampling_rate_hz=rate_hz,
        sequence_gaps=sequence_gap_count(sequences),
        mean_rssi=float(np.mean(rssis)) if rssis else 0.0,
    )

    diagnostic.update(
        {
            "accepted": True,
            "packet_count": record.packet_count,
            "valid_packet_count": record.valid_packet_count,
            "subcarrier_count": record.subcarrier_count,
            "duration_seconds": record.duration_seconds,
            "sampling_rate_hz": record.sampling_rate_hz,
            "sequence_gaps": record.sequence_gaps,
            "mean_rssi": record.mean_rssi,
        }
    )
    return record, diagnostic


def discover_and_load_dataset(
    config: dict[str, Any],
) -> tuple[list[FileRecord], pd.DataFrame]:
    dataset_root = resolve_project_path(config["dataset"]["root"])
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    files = sorted(dataset_root.rglob("*.bin"))
    if not files:
        raise ValueError(f"No .bin files found under: {dataset_root}")

    LOGGER.info("Discovered %d binary files.", len(files))
    records: list[FileRecord] = []
    diagnostics: list[dict[str, Any]] = []

    for file_index, path in enumerate(files, start=1):
        record, diagnostic = read_and_prepare_file(path, config)
        diagnostics.append(diagnostic)
        if record is not None:
            records.append(record)

        if file_index % 25 == 0 or file_index == len(files):
            LOGGER.info(
                "Prepared %d/%d files; accepted=%d.",
                file_index,
                len(files),
                len(records),
            )

    if not records:
        raise ValueError("No valid Dataset v2 files remained after validation.")

    present_labels = {record.label for record in records}
    missing_labels = set(config["dataset"]["labels"]) - present_labels
    if missing_labels:
        raise ValueError(f"Dataset is missing required labels: {sorted(missing_labels)}")

    return records, pd.DataFrame(diagnostics)


def determine_sampling_rate(
    records: Sequence[FileRecord],
    config: dict[str, Any],
) -> float:
    execution_config = config["execution"]
    mode = execution_config.get("sampling_rate_mode", "measured_median")
    fallback = float(execution_config.get("fallback_sampling_rate_hz", 50.0))

    measured = np.asarray(
        [record.sampling_rate_hz for record in records if record.sampling_rate_hz > 0],
        dtype=float,
    )

    if mode == "fixed" or measured.size == 0:
        return fallback

    sampling_rate = float(np.median(measured))
    if not math.isfinite(sampling_rate) or sampling_rate <= 0:
        return fallback
    return sampling_rate


def dataset_summary_table(records: Sequence[FileRecord]) -> pd.DataFrame:
    rows = [
        {
            "source_file": str(record.path),
            "file_name": record.file_name,
            "session": record.session,
            "quadrant": record.quadrant,
            "label": record.label,
            "packet_count": record.packet_count,
            "valid_packet_count": record.valid_packet_count,
            "subcarrier_count": record.subcarrier_count,
            "duration_seconds": record.duration_seconds,
            "sampling_rate_hz": record.sampling_rate_hz,
            "sequence_gaps": record.sequence_gaps,
            "mean_rssi": record.mean_rssi,
        }
        for record in records
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# File-based splitting and preprocessing fitting
# ---------------------------------------------------------------------------

def make_stratum(record: FileRecord, fields: Sequence[str]) -> str:
    values: list[str] = []
    for field in fields:
        if field == "label":
            values.append(record.label)
        elif field == "session":
            values.append(record.session)
        elif field == "quadrant":
            values.append(record.quadrant)
        else:
            raise ValueError(f"Unsupported stratification field: {field}")
    return "|".join(values)


def file_stratified_split(
    records: Sequence[FileRecord],
    test_size: float,
    seed: int,
    stratify_fields: Sequence[str],
) -> tuple[list[FileRecord], list[FileRecord]]:
    random_generator = np.random.default_rng(seed)
    groups: dict[str, list[FileRecord]] = defaultdict(list)

    for record in records:
        groups[make_stratum(record, stratify_fields)].append(record)

    train: list[FileRecord] = []
    test: list[FileRecord] = []

    for stratum in sorted(groups):
        stratum_records = list(groups[stratum])
        indices = random_generator.permutation(len(stratum_records))
        shuffled = [stratum_records[int(index)] for index in indices]

        if len(shuffled) < 2:
            raise ValueError(
                "Each stratum must contain at least two acquisition files. "
                f"Problematic stratum: {stratum}"
            )

        test_count = int(round(len(shuffled) * test_size))
        test_count = max(1, min(test_count, len(shuffled) - 1))
        test.extend(shuffled[:test_count])
        train.extend(shuffled[test_count:])

    random_generator.shuffle(train)
    random_generator.shuffle(test)
    return train, test


def fit_zscore_parameters(
    records: Sequence[FileRecord],
    minimum_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("Cannot fit preprocessing with an empty training set.")

    width = records[0].matrix.shape[1]
    total_count = 0
    sum_values = np.zeros(width, dtype=np.float64)
    sum_squares = np.zeros(width, dtype=np.float64)

    for record in records:
        if record.matrix.shape[1] != width:
            raise ValueError("Training files have inconsistent subcarrier counts.")
        matrix64 = record.matrix.astype(np.float64, copy=False)
        total_count += matrix64.shape[0]
        sum_values += np.sum(matrix64, axis=0)
        sum_squares += np.sum(matrix64 * matrix64, axis=0)

    means = sum_values / total_count
    variances = np.maximum(sum_squares / total_count - means * means, 0.0)
    stds = np.sqrt(variances)
    informative = np.flatnonzero(stds > minimum_std)

    if informative.size == 0:
        raise ValueError("All subcarriers were removed as non-informative.")

    return means, stds, informative


def sample_normalized_training_rows(
    records: Sequence[FileRecord],
    means: np.ndarray,
    stds: np.ndarray,
    informative_indices: np.ndarray,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    random_generator = np.random.default_rng(seed)
    total_rows = sum(record.matrix.shape[0] for record in records)

    if total_rows <= max_rows:
        sampled_parts = [
            (record.matrix[:, informative_indices] - means[informative_indices])
            / stds[informative_indices]
            for record in records
        ]
        return np.vstack(sampled_parts).astype(np.float32, copy=False)

    quotas = [
        max(1, int(round(max_rows * record.matrix.shape[0] / total_rows)))
        for record in records
    ]
    sampled_parts: list[np.ndarray] = []

    for record, quota in zip(records, quotas):
        quota = min(quota, record.matrix.shape[0])
        indices = random_generator.choice(
            record.matrix.shape[0],
            size=quota,
            replace=False,
        )
        normalized = (
            record.matrix[indices][:, informative_indices]
            - means[informative_indices]
        ) / stds[informative_indices]
        sampled_parts.append(normalized.astype(np.float32, copy=False))

    sampled = np.vstack(sampled_parts)
    if sampled.shape[0] > max_rows:
        chosen = random_generator.choice(
            sampled.shape[0],
            size=max_rows,
            replace=False,
        )
        sampled = sampled[chosen]
    return sampled


def greedy_non_redundant_subcarriers(
    normalized_rows: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if normalized_rows.ndim != 2 or normalized_rows.shape[1] == 0:
        raise ValueError("Correlation input matrix is empty.")

    if normalized_rows.shape[1] == 1 or threshold >= 1.0:
        return np.arange(normalized_rows.shape[1], dtype=int)

    correlation = np.corrcoef(normalized_rows, rowvar=False)
    correlation = np.nan_to_num(np.abs(correlation), nan=0.0, posinf=1.0, neginf=1.0)

    selected: list[int] = []
    for candidate in range(correlation.shape[0]):
        if not selected:
            selected.append(candidate)
            continue
        if np.all(correlation[candidate, selected] < threshold):
            selected.append(candidate)

    if not selected:
        selected = [0]
    return np.asarray(selected, dtype=int)


def fit_preprocessing_state(
    train_records: Sequence[FileRecord],
    config: dict[str, Any],
    seed: int,
) -> PreprocessingState:
    preprocessing_config = config["preprocessing"]
    means, stds, informative_indices = fit_zscore_parameters(
        train_records,
        minimum_std=float(preprocessing_config["minimum_informative_std"]),
    )
    sampled_rows = sample_normalized_training_rows(
        train_records,
        means,
        stds,
        informative_indices,
        max_rows=int(preprocessing_config["correlation_max_rows"]),
        seed=seed,
    )
    selected_relative = greedy_non_redundant_subcarriers(
        sampled_rows,
        threshold=float(preprocessing_config["correlation_threshold"]),
    )
    selected_original = informative_indices[selected_relative]

    return PreprocessingState(
        means=means,
        stds=stds,
        informative_indices=informative_indices,
        selected_relative_indices=selected_relative,
        selected_subcarriers=selected_original,
    )


def transform_record(
    record: FileRecord,
    state: PreprocessingState,
) -> np.ndarray:
    informative = state.informative_indices
    normalized = (
        record.matrix[:, informative] - state.means[informative]
    ) / state.stds[informative]
    reduced = normalized[:, state.selected_relative_indices]
    return reduced.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Windowing and feature extraction
# ---------------------------------------------------------------------------

def seconds_to_packets(seconds: float, sampling_rate_hz: float) -> int:
    return max(2, int(round(float(seconds) * sampling_rate_hz)))


def extract_window_features(window: np.ndarray) -> np.ndarray:
    if window.ndim != 2 or window.shape[0] < 2:
        raise ValueError("Feature extraction requires a 2-D window with >=2 rows.")

    minimum = np.min(window, axis=0)
    maximum = np.max(window, axis=0)
    differences = np.diff(window, axis=0)

    sample_positions = np.arange(window.shape[0], dtype=np.float64)
    centered_positions = sample_positions - np.mean(sample_positions)
    denominator = float(np.sum(centered_positions * centered_positions))
    centered_signal = window - np.mean(window, axis=0, keepdims=True)
    slopes = (
        np.sum(centered_positions[:, None] * centered_signal, axis=0)
        / denominator
        if denominator > 0
        else np.zeros(window.shape[1], dtype=float)
    )

    descriptors = np.stack(
        [
            np.mean(window, axis=0),
            np.std(window, axis=0),
            minimum,
            maximum,
            maximum - minimum,
            np.sum(window * window, axis=0),
            np.mean(np.abs(differences), axis=0),
            np.max(np.abs(differences), axis=0),
            np.sum(differences * differences, axis=0),
            np.std(differences, axis=0),
            slopes,
        ],
        axis=1,
    )
    return descriptors.reshape(-1).astype(np.float32, copy=False)


def create_feature_matrix(
    records: Sequence[FileRecord],
    state: PreprocessingState,
    window_packets: int,
    step_packets: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    metadata_rows: list[dict[str, Any]] = []

    for record in records:
        transformed = transform_record(record, state)
        if transformed.shape[0] < window_packets:
            continue

        window_index = 0
        for start in range(
            0,
            transformed.shape[0] - window_packets + 1,
            step_packets,
        ):
            end = start + window_packets
            window = transformed[start:end]
            features.append(extract_window_features(window))
            labels.append(CLASS_TO_INDEX[record.label])
            metadata_rows.append(
                {
                    "source_file": str(record.path),
                    "file_name": record.file_name,
                    "session": record.session,
                    "quadrant": record.quadrant,
                    "label": record.label,
                    "window_index": window_index,
                    "start_packet": start,
                    "end_packet_exclusive": end,
                }
            )
            window_index += 1

    if not features:
        raise ValueError(
            "No windows were generated. Reduce the window size or inspect files."
        )

    return (
        np.vstack(features).astype(np.float32, copy=False),
        np.asarray(labels, dtype=int),
        pd.DataFrame(metadata_rows),
    )


def prepare_feature_split(
    train_records: Sequence[FileRecord],
    test_records: Sequence[FileRecord],
    config: dict[str, Any],
    window_packets: int,
    step_packets: int,
    seed: int,
) -> FeatureSplit:
    state = fit_preprocessing_state(train_records, config, seed=seed)
    x_train, y_train, train_metadata = create_feature_matrix(
        train_records,
        state,
        window_packets,
        step_packets,
    )
    x_test, y_test, test_metadata = create_feature_matrix(
        test_records,
        state,
        window_packets,
        step_packets,
    )

    return FeatureSplit(
        x_train=x_train,
        y_train=y_train,
        train_metadata=train_metadata,
        x_test=x_test,
        y_test=y_test,
        test_metadata=test_metadata,
        preprocessing=state,
        window_packets=window_packets,
        step_packets=step_packets,
    )


def get_cached_file_feature_split(
    records: Sequence[FileRecord],
    config: dict[str, Any],
    window_packets: int,
    step_packets: int,
    seed: int,
    cache: dict[tuple[int, int, int], FeatureSplit],
) -> FeatureSplit:
    key = (int(seed), int(window_packets), int(step_packets))
    if key not in cache:
        train_records, test_records = file_stratified_split(
            records,
            float(config["validation"]["test_size"]),
            int(seed),
            config["validation"]["stratify_by"],
        )
        cache[key] = prepare_feature_split(
            train_records,
            test_records,
            config,
            int(window_packets),
            int(step_packets),
            seed=int(seed),
        )
    return cache[key]


def fisher_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    global_mean = np.mean(x, axis=0)
    numerator = np.zeros(x.shape[1], dtype=np.float64)
    denominator = np.zeros(x.shape[1], dtype=np.float64)

    for class_index in np.unique(y):
        class_values = x[y == class_index]
        class_count = class_values.shape[0]
        class_mean = np.mean(class_values, axis=0)
        class_variance = np.var(class_values, axis=0)
        numerator += class_count * (class_mean - global_mean) ** 2
        denominator += class_count * class_variance

    return numerator / (denominator + 1e-8)


def select_feature_indices(
    x_train: np.ndarray,
    y_train: np.ndarray,
    top_k: int | str,
) -> tuple[np.ndarray, np.ndarray]:
    scores = fisher_scores(x_train, y_train)
    ranking = np.argsort(scores)[::-1]

    if top_k == "all":
        selected = ranking
    else:
        selected = ranking[: min(int(top_k), x_train.shape[1])]

    return selected.astype(int), scores


def feature_descriptor(
    feature_index: int,
    selected_subcarriers: Sequence[int],
    feature_names: Sequence[str],
) -> dict[str, Any]:
    features_per_subcarrier = len(feature_names)
    reduced_subcarrier_index = feature_index // features_per_subcarrier
    feature_type_index = feature_index % features_per_subcarrier

    if reduced_subcarrier_index >= len(selected_subcarriers):
        raise IndexError("Feature index exceeds selected subcarrier vector.")

    return {
        "feature_index": int(feature_index),
        "reduced_subcarrier_index": int(reduced_subcarrier_index),
        "original_subcarrier": int(selected_subcarriers[reduced_subcarrier_index]),
        "feature_type": str(feature_names[feature_type_index]),
    }


# ---------------------------------------------------------------------------
# Models and metrics
# ---------------------------------------------------------------------------

def optional_xgboost_classifier(parameters: dict[str, Any]) -> BaseEstimator | None:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=0,
        n_jobs=1,
        verbosity=0,
        **parameters,
    )


def create_model(
    name: str,
    parameters: dict[str, Any],
    seed: int,
    num_classes: int = 3,
) -> BaseEstimator | None:
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, **parameters)

    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **parameters)

    if name == "extra_trees":
        return ExtraTreesClassifier(random_state=seed, n_jobs=1, **parameters)

    if name == "knn":
        return Pipeline([("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(**parameters))])

    if name == "linear_svm":
        return Pipeline([("scaler", StandardScaler()), ("classifier", LinearSVC(random_state=seed, **parameters))])

    if name == "rbf_svm":
        return Pipeline([("scaler", StandardScaler()), ("classifier", SVC(random_state=seed, **parameters))])

    if name == "logistic_regression":
        return Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(random_state=seed, **parameters))])

    if name.startswith("gradient_boosting") or name == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed, **parameters)

    if name.startswith("xgboost") or name == "xgboost":
        model = optional_xgboost_classifier(parameters)
        if model is not None:
            model.set_params(random_state=seed, num_class=num_classes)
        return model

    raise ValueError(f"Unsupported classifier: {name}")


def normalize_probabilities(
    model: BaseEstimator,
    x: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(x), dtype=float)
    elif hasattr(model, "decision_function"):
        decision = np.asarray(model.decision_function(x), dtype=float)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        shifted = decision - np.max(decision, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        raw = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    else:
        predictions = np.asarray(model.predict(x), dtype=int)
        raw = np.zeros((len(predictions), num_classes), dtype=float)
        raw[np.arange(len(predictions)), predictions] = 1.0

    model_classes = np.asarray(getattr(model, "classes_", np.arange(raw.shape[1])))
    full = np.zeros((raw.shape[0], num_classes), dtype=float)
    for column_index, class_value in enumerate(model_classes):
        full[:, int(class_value)] = raw[:, column_index]

    row_sums = np.sum(full, axis=1, keepdims=True)
    zero_rows = row_sums[:, 0] <= 0
    if np.any(zero_rows):
        full[zero_rows] = 1.0 / num_classes
        row_sums = np.sum(full, axis=1, keepdims=True)
    return full / row_sums


def evaluate_model(
    model: BaseEstimator,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_order: Sequence[str] = CLASS_ORDER,
) -> EvaluationResult:
    model.fit(x_train, y_train)
    y_pred = np.asarray(model.predict(x_test), dtype=int)
    probabilities = normalize_probabilities(model, x_test, len(class_order))

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    try:
        roc_auc = roc_auc_score(
            y_test,
            probabilities,
            labels=np.arange(len(class_order)),
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        roc_auc = float("nan")

    y_binary = label_binarize(y_test, classes=np.arange(len(class_order)))
    try:
        average_precision = average_precision_score(
            y_binary,
            probabilities,
            average="macro",
        )
    except ValueError:
        average_precision = float("nan")

    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=np.arange(len(class_order)),
        zero_division=0,
    )
    class_metrics = pd.DataFrame(
        {
            "class_index": np.arange(len(class_order)),
            "class": list(class_order),
            "precision": precision,
            "recall": recall,
            "f1": class_f1,
            "support": support,
        }
    )
    confusion = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(class_order)),
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "roc_auc_ovr_macro": float(roc_auc),
        "average_precision_macro": float(average_precision),
        "empty_f1": float(class_f1[CLASS_TO_INDEX["empty"]]),
        "static_presence_f1": float(
            class_f1[CLASS_TO_INDEX["static_presence"]]
        ),
        "movement_f1": float(class_f1[CLASS_TO_INDEX["movement"]]),
        "test_samples": int(len(y_test)),
    }

    return EvaluationResult(
        metrics=metrics,
        y_true=np.asarray(y_test, dtype=int),
        y_pred=y_pred,
        probabilities=probabilities,
        class_metrics=class_metrics,
        confusion=confusion,
    )


def summarize_metric_rows(
    rows: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "roc_auc_ovr_macro",
        "average_precision_macro",
        "empty_f1",
        "static_presence_f1",
        "movement_f1",
        "selected_feature_count",
        "selected_subcarrier_count",
        "train_windows",
        "test_windows",
    ]
    available = [column for column in metric_columns if column in rows.columns]

    grouped = rows.groupby(list(group_columns), dropna=False)[available]
    mean_frame = grouped.mean().add_suffix("_mean")
    std_frame = grouped.std(ddof=0).add_suffix("_std")
    return mean_frame.join(std_frame).reset_index()


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------

def evaluate_split_with_model(
    feature_split: FeatureSplit,
    model_name: str,
    model_parameters: dict[str, Any],
    top_k: int | str,
    seed: int,
) -> tuple[EvaluationResult, np.ndarray, np.ndarray, BaseEstimator]:
    selected_indices, scores = select_feature_indices(
        feature_split.x_train,
        feature_split.y_train,
        top_k,
    )
    model = create_model(model_name, model_parameters, seed=seed)
    if model is None:
        raise RuntimeError(f"Optional model unavailable: {model_name}")

    result = evaluate_model(
        model,
        feature_split.x_train[:, selected_indices],
        feature_split.y_train,
        feature_split.x_test[:, selected_indices],
        feature_split.y_test,
    )
    return result, selected_indices, scores, model


def base_experiment_row(
    seed: int,
    split: FeatureSplit,
    selected_indices: Sequence[int],
) -> dict[str, Any]:
    return {
        "seed": seed,
        "window_packets": split.window_packets,
        "step_packets": split.step_packets,
        "train_files": split.train_metadata["source_file"].nunique(),
        "test_files": split.test_metadata["source_file"].nunique(),
        "train_windows": len(split.y_train),
        "test_windows": len(split.y_test),
        "selected_subcarrier_count": len(
            split.preprocessing.selected_subcarriers
        ),
        "selected_feature_count": len(selected_indices),
    }


def window_step_search(
    records: Sequence[FileRecord],
    config: dict[str, Any],
    sampling_rate_hz: float,
    paths: dict[str, Path],
    experiment_index: list[dict[str, str]],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Stage 1/8: window and step search.")
    validation = config["validation"]
    seeds = list(validation["search_seeds"])
    candidates = list(validation["window_step_candidates_seconds"])

    if config["execution"].get("quick_mode", False):
        seeds = seeds[:1]
        candidates = candidates[:2]

    rows: list[dict[str, Any]] = []
    fixed_model_name = "gradient_boosting_20_d3"
    fixed_parameters = {
        "n_estimators": 20,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 1.0,
    }

    if config["execution"].get("quick_mode", False):
        fixed_model_name = "decision_tree"
        fixed_parameters = {
            "max_depth": 4,
            "min_samples_split": 5,
            "class_weight": "balanced",
        }

    numeric_top_k = [
        int(value)
        for value in config["features"]["top_k_candidates"]
        if value != "all"
    ]
    fixed_top_k = 126 if 126 in numeric_top_k else max(numeric_top_k)
    if config["execution"].get("quick_mode", False):
        fixed_top_k = 30

    for candidate_index, candidate in enumerate(candidates, start=1):
        window_seconds = float(candidate["window"])
        step_seconds = float(candidate["step"])
        window_packets = seconds_to_packets(window_seconds, sampling_rate_hz)
        step_packets = max(1, seconds_to_packets(step_seconds, sampling_rate_hz))
        step_packets = min(step_packets, window_packets)

        LOGGER.info(
            "Window candidate %d/%d: %.3fs/%d packets, step %.3fs/%d packets.",
            candidate_index,
            len(candidates),
            window_seconds,
            window_packets,
            step_seconds,
            step_packets,
        )

        for seed in seeds:
            train_records, test_records = file_stratified_split(
                records,
                test_size=float(validation["test_size"]),
                seed=int(seed),
                stratify_fields=validation["stratify_by"],
            )
            split = prepare_feature_split(
                train_records,
                test_records,
                config,
                window_packets,
                step_packets,
                seed=int(seed),
            )
            result, selected, _, _ = evaluate_split_with_model(
                split,
                fixed_model_name,
                fixed_parameters,
                fixed_top_k,
                seed=int(seed),
            )

            row = base_experiment_row(int(seed), split, selected)
            row.update(result.metrics)
            row.update(
                {
                    "window_seconds": window_seconds,
                    "step_seconds": step_seconds,
                    "model": fixed_model_name,
                    "top_k": fixed_top_k,
                }
            )
            rows.append(row)

    detail = pd.DataFrame(rows)
    summary = summarize_metric_rows(
        detail,
        ["window_seconds", "step_seconds", "window_packets", "step_packets"],
    )
    summary = summary.sort_values(
        ["macro_f1_mean", "movement_f1_mean", "macro_f1_std", "window_seconds"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    best = summary.iloc[0].to_dict()
    save_dataframe(paths["tables"] / "window_step_comparison_by_seed.csv", detail)
    save_dataframe(paths["tables"] / "window_step_comparison_summary.csv", summary)
    plot_window_step(summary, paths["figures"] / "window_step_macro_f1.png")

    experiment_index.append(
        {
            "stage": "window_step_search",
            "objective": "Select temporal window and update interval in seconds.",
            "main_table": "tables/window_step_comparison_summary.csv",
            "main_figure": "figures/window_step_macro_f1.png",
        }
    )
    return best, detail, summary


def tree_parameter_search(
    records: Sequence[FileRecord],
    config: dict[str, Any],
    best_window: dict[str, Any],
    paths: dict[str, Path],
    experiment_index: list[dict[str, str]],
    split_cache: dict[tuple[int, int, int], FeatureSplit],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Stage 2/8: decision-tree parameter search.")
    validation = config["validation"]
    seeds = list(validation["search_seeds"])
    candidates = list(config["models"]["decision_tree_candidates"])

    if config["execution"].get("quick_mode", False):
        seeds = seeds[:1]
        candidates = candidates[:2]

    window_packets = int(best_window["window_packets"])
    step_packets = int(best_window["step_packets"])
    numeric_top_k = [
        int(value)
        for value in config["features"]["top_k_candidates"]
        if value != "all"
    ]
    top_k = 126 if 126 in numeric_top_k else max(numeric_top_k)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        split = get_cached_file_feature_split(
            records,
            config,
            window_packets,
            step_packets,
            int(seed),
            split_cache,
        )
        selected, _ = select_feature_indices(
            split.x_train,
            split.y_train,
            top_k,
        )

        for candidate in candidates:
            model = create_model("decision_tree", candidate, seed=int(seed))
            assert model is not None
            result = evaluate_model(
                model,
                split.x_train[:, selected],
                split.y_train,
                split.x_test[:, selected],
                split.y_test,
            )
            row = base_experiment_row(int(seed), split, selected)
            row.update(result.metrics)
            row.update(
                {
                    "max_depth": candidate.get("max_depth"),
                    "min_samples_split": candidate.get("min_samples_split"),
                    "top_k": top_k,
                }
            )
            rows.append(row)

    detail = pd.DataFrame(rows)
    summary = summarize_metric_rows(
        detail,
        ["max_depth", "min_samples_split", "top_k"],
    )
    summary = summary.sort_values(
        ["macro_f1_mean", "movement_f1_mean", "macro_f1_std"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    best_row = summary.iloc[0]
    best_parameters = {
        "max_depth": int(best_row["max_depth"]),
        "min_samples_split": int(best_row["min_samples_split"]),
    }

    save_dataframe(paths["tables"] / "decision_tree_tuning_by_seed.csv", detail)
    save_dataframe(paths["tables"] / "decision_tree_tuning_summary.csv", summary)
    plot_grouped_metric(
        summary.assign(
            configuration=summary.apply(
                lambda row: (
                    f"depth={int(row['max_depth'])}, "
                    f"min_split={int(row['min_samples_split'])}"
                ),
                axis=1,
            )
        ),
        "configuration",
        "macro_f1_mean",
        "macro_f1_std",
        "Decision-tree tuning",
        "Configuration",
        "Macro F1",
        paths["figures"] / "decision_tree_tuning_macro_f1.png",
    )
    experiment_index.append(
        {
            "stage": "decision_tree_tuning",
            "objective": "Tune the compact decision-tree baseline.",
            "main_table": "tables/decision_tree_tuning_summary.csv",
            "main_figure": "figures/decision_tree_tuning_macro_f1.png",
        }
    )
    return best_parameters, detail, summary


def enabled_model_specs(
    config: dict[str, Any],
    best_tree_parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "name": "decision_tree",
            "parameters": best_tree_parameters,
            "optional": False,
        }
    ]

    for item in config["models"]["comparison"]:
        if item.get("enabled", True):
            specs.append(
                {
                    "name": item["name"],
                    "parameters": dict(item.get("parameters", {})),
                    "optional": bool(item.get("optional", False)),
                }
            )
    return specs


def model_comparison(
    records: Sequence[FileRecord],
    config: dict[str, Any],
    best_window: dict[str, Any],
    best_tree_parameters: dict[str, Any],
    paths: dict[str, Path],
    experiment_index: list[dict[str, str]],
    split_cache: dict[tuple[int, int, int], FeatureSplit],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Stage 3/8: classifier comparison.")
    validation = config["validation"]
    seeds = list(validation["evaluation_seeds"])
    if config["execution"].get("quick_mode", False):
        seeds = seeds[:2]

    window_packets = int(best_window["window_packets"])
    step_packets = int(best_window["step_packets"])
    numeric_top_k = [
        int(value)
        for value in config["features"]["top_k_candidates"]
        if value != "all"
    ]
    top_k = 126 if 126 in numeric_top_k else max(numeric_top_k)
    specs = enabled_model_specs(config, best_tree_parameters)

    rows: list[dict[str, Any]] = []
    unavailable: set[str] = set()

    for seed in seeds:
        LOGGER.info("Classifier comparison seed %s.", seed)
        split = get_cached_file_feature_split(
            records,
            config,
            window_packets,
            step_packets,
            int(seed),
            split_cache,
        )
        selected, _ = select_feature_indices(
            split.x_train,
            split.y_train,
            top_k,
        )

        for spec in specs:
            if spec["name"] in unavailable:
                continue
            model = create_model(
                spec["name"],
                spec["parameters"],
                seed=int(seed),
            )
            if model is None:
                if spec["optional"]:
                    unavailable.add(spec["name"])
                    LOGGER.warning(
                        "Optional model %s skipped because its package is unavailable.",
                        spec["name"],
                    )
                    continue
                raise RuntimeError(f"Required model unavailable: {spec['name']}")

            result = evaluate_model(
                model,
                split.x_train[:, selected],
                split.y_train,
                split.x_test[:, selected],
                split.y_test,
            )
            row = base_experiment_row(int(seed), split, selected)
            row.update(result.metrics)
            row.update(
                {
                    "model": spec["name"],
                    "parameters_json": json.dumps(spec["parameters"], sort_keys=True),
                    "top_k": top_k,
                }
            )
            rows.append(row)

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError("No classifier was evaluated.")

    summary = summarize_metric_rows(detail, ["model", "parameters_json", "top_k"])
    summary = summary.sort_values(
        ["macro_f1_mean", "movement_f1_mean", "macro_f1_std"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    best_row = summary.iloc[0]
    best_model = {
        "name": str(best_row["model"]),
        "parameters": json.loads(str(best_row["parameters_json"])),
    }

    save_dataframe(paths["tables"] / "model_comparison_by_seed.csv", detail)
    save_dataframe(paths["tables"] / "model_comparison_summary.csv", summary)
    plot_grouped_metric(
        summary,
        "model",
        "macro_f1_mean",
        "macro_f1_std",
        "Classifier comparison with identical file splits",
        "Classifier",
        "Macro F1",
        paths["figures"] / "model_comparison_macro_f1.png",
    )
    plot_grouped_metric(
        summary,
        "model",
        "movement_f1_mean",
        None,
        "Movement-class comparison",
        "Classifier",
        "Movement F1",
        paths["figures"] / "model_comparison_movement_f1.png",
    )

    experiment_index.append(
        {
            "stage": "model_comparison",
            "objective": (
                "Compare decision tree, logistic regression, Gradient "
                "Boosting and optional XGBoost on identical splits."
            ),
            "main_table": "tables/model_comparison_summary.csv",
            "main_figure": "figures/model_comparison_macro_f1.png",
        }
    )
    return best_model, detail, summary


def feature_budget_comparison(
    records: Sequence[FileRecord],
    config: dict[str, Any],
    best_window: dict[str, Any],
    best_model: dict[str, Any],
    paths: dict[str, Path],
    experiment_index: list[dict[str, str]],
    split_cache: dict[tuple[int, int, int], FeatureSplit],
) -> tuple[int | str, pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Stage 4/8: feature-budget comparison.")
    validation = config["validation"]
    seeds = list(validation["evaluation_seeds"])
    budgets = list(config["features"]["top_k_candidates"])
    if config["execution"].get("quick_mode", False):
        seeds = seeds[:2]
        budgets = budgets[:2]

    window_packets = int(best_window["window_packets"])
    step_packets = int(best_window["step_packets"])
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        LOGGER.info("Feature-budget seed %s.", seed)
        split = get_cached_file_feature_split(
            records,
            config,
            window_packets,
            step_packets,
            int(seed),
            split_cache,
        )

        for budget in budgets:
            result, selected, _, _ = evaluate_split_with_model(
                split,
                best_model["name"],
                best_model["parameters"],
                budget,
                seed=int(seed),
            )
            row = base_experiment_row(int(seed), split, selected)
            row.update(result.metrics)
            row.update(
                {
                    "model": best_model["name"],
                    "top_k": str(budget),
                    "top_k_sort": (
                        split.x_train.shape[1]
                        if budget == "all"
                        else int(budget)
                    ),
                }
            )
            rows.append(row)

    detail = pd.DataFrame(rows)
    summary = summarize_metric_rows(detail, ["model", "top_k", "top_k_sort"])
    summary = choose_feature_budget_order(summary, config)
    selected_budget_text = str(summary.iloc[0]["top_k"])
    selected_budget: int | str = (
        "all" if selected_budget_text == "all" else int(selected_budget_text)
    )

    save_dataframe(paths["tables"] / "feature_budget_by_seed.csv", detail)
    save_dataframe(paths["tables"] / "feature_budget_summary.csv", summary)
    plot_grouped_metric(
        summary.assign(top_k_label=summary["top_k"].astype(str)),
        "top_k_label",
        "macro_f1_mean",
        "macro_f1_std",
        f"Feature budget for {best_model['name']}",
        "Top-K Fisher features",
        "Macro F1",
        paths["figures"] / "feature_budget_macro_f1.png",
    )
    plot_grouped_metric(
        summary.assign(top_k_label=summary["top_k"].astype(str)),
        "top_k_label",
        "movement_f1_mean",
        None,
        "Feature budget and movement recognition",
        "Top-K Fisher features",
        "Movement F1",
        paths["figures"] / "feature_budget_movement_f1.png",
    )

    experiment_index.append(
        {
            "stage": "feature_budget_comparison",
            "objective": "Find a compact Fisher Top-K without material performance loss.",
            "main_table": "tables/feature_budget_summary.csv",
            "main_figure": "figures/feature_budget_macro_f1.png",
        }
    )
    return selected_budget, detail, summary


def choose_feature_budget_order(
    summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    policy = config["selection_policy"]
    best_macro = float(summary["macro_f1_mean"].max())
    tolerance = float(policy.get("macro_f1_tolerance", 0.005))
    eligible = summary[summary["macro_f1_mean"] >= best_macro - tolerance].copy()

    sort_columns: list[str] = []
    ascending: list[bool] = []

    if policy.get("prefer_movement_f1", True):
        sort_columns.append("movement_f1_mean")
        ascending.append(False)
    if policy.get("prefer_lower_std", True):
        sort_columns.append("macro_f1_std")
        ascending.append(True)
    if policy.get("prefer_fewer_features", True):
        sort_columns.append("selected_feature_count_mean")
        ascending.append(True)

    if sort_columns:
        eligible = eligible.sort_values(sort_columns, ascending=ascending)

    selected_index = eligible.index[0]
    remaining = summary.drop(index=selected_index).sort_values(
        ["macro_f1_mean", "movement_f1_mean", "macro_f1_std"],
        ascending=[False, False, True],
    )
    ordered = pd.concat([summary.loc[[selected_index]], remaining], ignore_index=True)
    ordered.insert(0, "selected_by_policy", [True] + [False] * (len(ordered) - 1))
    return ordered


def repeated_final_evaluation(
    records: Sequence[FileRecord],
    config: dict[str, Any],
    best_window: dict[str, Any],
    best_model: dict[str, Any],
    selected_budget: int | str,
    paths: dict[str, Path],
    experiment_index: list[dict[str, str]],
    split_cache: dict[tuple[int, int, int], FeatureSplit],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    LOGGER.info("Stage 5/8: repeated final-candidate evaluation.")
    validation = config["validation"]
    seeds = list(validation["evaluation_seeds"])
    if config["execution"].get("quick_mode", False):
        seeds = seeds[:2]

    window_packets = int(best_window["window_packets"])
    step_packets = int(best_window["step_packets"])
    rows: list[dict[str, Any]] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    prediction_rows: list[pd.DataFrame] = []

    for seed in seeds:
        LOGGER.info("Final-candidate repeated holdout seed %s.", seed)
        split = get_cached_file_feature_split(
            records,
            config,
            window_packets,
            step_packets,
            int(seed),
            split_cache,
        )
        result, selected, _, _ = evaluate_split_with_model(
            split,
            best_model["name"],
            best_model["parameters"],
            selected_budget,
            seed=int(seed),
        )
        row = base_experiment_row(int(seed), split, selected)
        row.update(result.metrics)
        row.update(
            {
                "model": best_model["name"],
                "top_k": str(selected_budget),
            }
        )
        rows.append(row)
        all_true.append(result.y_true)
        all_pred.append(result.y_pred)

        seed_predictions = split.test_metadata.copy()
        seed_predictions.insert(0, "seed", int(seed))
        seed_predictions["true_class"] = [CLASS_ORDER[index] for index in result.y_true]
        seed_predictions["predicted_class"] = [
            CLASS_ORDER[index] for index in result.y_pred
        ]
        for class_index, class_name in enumerate(CLASS_ORDER):
            seed_predictions[f"probability_{class_name}"] = result.probabilities[
                :, class_index
            ]
        prediction_rows.append(seed_predictions)

    detail = pd.DataFrame(rows)
    summary = summarize_metric_rows(detail, ["model", "top_k"])
    concatenated_true = np.concatenate(all_true)
    concatenated_pred = np.concatenate(all_pred)
    aggregate_confusion = confusion_matrix(
        concatenated_true,
        concatenated_pred,
        labels=np.arange(len(CLASS_ORDER)),
    )
    aggregate_class_metrics = class_metrics_from_predictions(
        concatenated_true,
        concatenated_pred,
    )
    predictions = pd.concat(prediction_rows, ignore_index=True)

    save_dataframe(paths["tables"] / "final_candidate_by_seed.csv", detail)
    save_dataframe(paths["tables"] / "final_candidate_summary.csv", summary)
    save_dataframe(paths["tables"] / "final_candidate_predictions.csv", predictions)
    save_dataframe(
        paths["tables"] / "final_candidate_class_metrics.csv",
        aggregate_class_metrics,
    )
    save_dataframe(
        paths["tables"] / "final_candidate_confusion_matrix.csv",
        pd.DataFrame(aggregate_confusion, index=CLASS_ORDER, columns=CLASS_ORDER)
        .rename_axis("true_class")
        .reset_index(),
    )
    plot_seed_stability(
        detail,
        paths["figures"] / "final_candidate_stability.png",
    )
    plot_confusion_matrix(
        aggregate_confusion,
        CLASS_ORDER,
        "Aggregated repeated file-holdout confusion matrix",
        paths["figures"] / "final_candidate_confusion_matrix.png",
    )
    plot_class_f1(
        aggregate_class_metrics,
        paths["figures"] / "final_candidate_class_f1.png",
    )

    experiment_index.append(
        {
            "stage": "repeated_final_evaluation",
            "objective": "Measure final-candidate stability across repeated file splits.",
            "main_table": "tables/final_candidate_summary.csv",
            "main_figure": "figures/final_candidate_stability.png",
        }
    )
    return detail, summary, aggregate_confusion, aggregate_class_metrics


def class_metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_ORDER)),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class": CLASS_ORDER,
            "precision": precision,
            "recall": recall,
            "f1": class_f1,
            "support": support,
        }
    )


# ---------------------------------------------------------------------------
# Diagnostic evaluations
# ---------------------------------------------------------------------------

def binary_task_evaluation(
    records: Sequence[FileRecord],
    config: dict[str, Any],
    best_window: dict[str, Any],
    best_model: dict[str, Any],
    selected_budget: int | str,
    paths: dict[str, Path],
    experiment_index: list[dict[str, str]],
    split_cache: dict[tuple[int, int, int], FeatureSplit],
) -> pd.DataFrame:
    if not config["validation"].get("run_binary_diagnostics", True):
        return pd.DataFrame()

    LOGGER.info("Stage 6a/8: binary-task diagnostics.")
    seeds = list(config["validation"]["evaluation_seeds"])
    if config["execution"].get("quick_mode", False):
        seeds = seeds[:1]

    tasks = {
        "empty_vs_presence": {
            "include": {0, 1, 2},
            "mapping": {0: 0, 1: 1, 2: 1},
            "class_names": ["empty", "presence"],
        },
        "static_vs_movement": {
            "include": {1, 2},
            "mapping": {1: 0, 2: 1},
            "class_names": ["static_presence", "movement"],
        },
    }
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        split = get_cached_file_feature_split(
            records,
            config,
            int(best_window["window_packets"]),
            int(best_window["step_packets"]),
            int(seed),
            split_cache,
        )

        for task_name, task in tasks.items():
            train_mask = np.isin(split.y_train, list(task["include"]))
            test_mask = np.isin(split.y_test, list(task["include"]))
            y_train = np.asarray(
                [task["mapping"][int(value)] for value in split.y_train[train_mask]],
                dtype=int,
            )
            y_test = np.asarray(
                [task["mapping"][int(value)] for value in split.y_test[test_mask]],
                dtype=int,
            )
            x_train = split.x_train[train_mask]
            x_test = split.x_test[test_mask]
            selected, _ = select_feature_indices(x_train, y_train, selected_budget)
            model = create_model(
                best_model["name"],
                best_model["parameters"],
                seed=int(seed),
                num_classes=2,
            )
            if model is None:
                continue
            model.fit(x_train[:, selected], y_train)
            y_pred = np.asarray(model.predict(x_test[:, selected]), dtype=int)
            row = {
                "task": task_name,
                "seed": int(seed),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "macro_f1": float(
                    f1_score(y_test, y_pred, average="macro", zero_division=0)
                ),
                "weighted_f1": float(
                    f1_score(y_test, y_pred, average="weighted", zero_division=0)
                ),
                "positive_class_f1": float(
                    f1_score(y_test, y_pred, labels=[1], average="macro", zero_division=0)
                ),
                "selected_feature_count": len(selected),
                "test_samples": len(y_test),
            }
            rows.append(row)

    detail = pd.DataFrame(rows)
    summary = detail.groupby("task").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        weighted_f1_mean=("weighted_f1", "mean"),
        positive_class_f1_mean=("positive_class_f1", "mean"),
    ).reset_index()
    summary = summary.fillna(0.0)

    save_dataframe(paths["tables"] / "binary_tasks_by_seed.csv", detail)
    save_dataframe(paths["tables"] / "binary_tasks_summary.csv", summary)
    plot_grouped_metric(
        summary,
        "task",
        "macro_f1_mean",
        "macro_f1_std",
        "Binary-task diagnosis",
        "Task",
        "Macro F1",
        paths["figures"] / "binary_tasks_macro_f1.png",
    )
    experiment_index.append(
        {
            "stage": "binary_task_diagnostics",
            "objective": "Identify whether presence or motion-state separation is the bottleneck.",
            "main_table": "tables/binary_tasks_summary.csv",
            "main_figure": "figures/binary_tasks_macro_f1.png",
        }
    )
    return summary


def hierarchical_comparison(
    records: Sequence[FileRecord],
    config: dict[str, Any],
    best_window: dict[str, Any],
    best_model: dict[str, Any],
    selected_budget: int | str,
    paths: dict[str, Path],
    experiment_index: list[dict[str, str]],
    split_cache: dict[tuple[int, int, int], FeatureSplit],
) -> pd.DataFrame:
    if not config["validation"].get("run_hierarchical_comparison", True):
        return pd.DataFrame()

    LOGGER.info("Stage 6b/8: direct versus hierarchical comparison.")
    seeds = list(config["validation"]["evaluation_seeds"])
    if config["execution"].get("quick_mode", False):
        seeds = seeds[:1]

    rows: list[dict[str, Any]] = []

    for seed in seeds:
        split = get_cached_file_feature_split(
            records,
            config,
            int(best_window["window_packets"]),
            int(best_window["step_packets"]),
            int(seed),
            split_cache,
        )

        # Direct multiclass model.
        direct_result, _, _, _ = evaluate_split_with_model(
            split,
            best_model["name"],
            best_model["parameters"],
            selected_budget,
            seed=int(seed),
        )
        direct_row = {
            "architecture": "direct_multiclass",
            "seed": int(seed),
            **direct_result.metrics,
        }
        rows.append(direct_row)

        # Stage A: empty versus presence.
        y_train_a = np.where(split.y_train == 0, 0, 1)
        selected_a, _ = select_feature_indices(
            split.x_train,
            y_train_a,
            selected_budget,
        )
        model_a = create_model(
            best_model["name"],
            best_model["parameters"],
            seed=int(seed),
            num_classes=2,
        )
        assert model_a is not None
        model_a.fit(split.x_train[:, selected_a], y_train_a)
        predicted_a = np.asarray(
            model_a.predict(split.x_test[:, selected_a]),
            dtype=int,
        )

        # Stage B: static versus movement, trained only on presence windows.
        train_presence = split.y_train != 0
        x_train_b = split.x_train[train_presence]
        y_train_b = split.y_train[train_presence] - 1
        selected_b, _ = select_feature_indices(
            x_train_b,
            y_train_b,
            selected_budget,
        )
        model_b = create_model(
            best_model["name"],
            best_model["parameters"],
            seed=int(seed),
            num_classes=2,
        )
        assert model_b is not None
        model_b.fit(x_train_b[:, selected_b], y_train_b)

        hierarchical_pred = np.zeros_like(split.y_test)
        predicted_presence_indices = np.flatnonzero(predicted_a == 1)
        if predicted_presence_indices.size > 0:
            stage_b_pred = np.asarray(
                model_b.predict(
                    split.x_test[predicted_presence_indices][:, selected_b]
                ),
                dtype=int,
            )
            hierarchical_pred[predicted_presence_indices] = stage_b_pred + 1

        class_f1 = f1_score(
            split.y_test,
            hierarchical_pred,
            labels=np.arange(3),
            average=None,
            zero_division=0,
        )
        rows.append(
            {
                "architecture": "hierarchical_two_stage",
                "seed": int(seed),
                "accuracy": float(
                    accuracy_score(split.y_test, hierarchical_pred)
                ),
                "macro_f1": float(
                    f1_score(
                        split.y_test,
                        hierarchical_pred,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "weighted_f1": float(
                    f1_score(
                        split.y_test,
                        hierarchical_pred,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "roc_auc_ovr_macro": float("nan"),
                "average_precision_macro": float("nan"),
                "empty_f1": float(class_f1[0]),
                "static_presence_f1": float(class_f1[1]),
                "movement_f1": float(class_f1[2]),
                "test_samples": len(split.y_test),
            }
        )

    detail = pd.DataFrame(rows)
    summary = summarize_metric_rows(detail, ["architecture"])
    summary = summary.sort_values("macro_f1_mean", ascending=False)
    save_dataframe(paths["tables"] / "hierarchical_comparison_by_seed.csv", detail)
    save_dataframe(paths["tables"] / "hierarchical_comparison_summary.csv", summary)
    plot_grouped_metric(
        summary,
        "architecture",
        "macro_f1_mean",
        "macro_f1_std",
        "Direct versus hierarchical classification",
        "Architecture",
        "Macro F1",
        paths["figures"] / "hierarchical_comparison_macro_f1.png",
    )
    experiment_index.append(
        {
            "stage": "hierarchical_comparison",
            "objective": "Test whether a two-stage classifier improves the direct model.",
            "main_table": "tables/hierarchical_comparison_summary.csv",
            "main_figure": "figures/hierarchical_comparison_macro_f1.png",
        }
    )
    return summary


# ---------------------------------------------------------------------------
# Session and quadrant holdouts
# ---------------------------------------------------------------------------

def grouped_holdout_evaluation(
    records: Sequence[FileRecord],
    group_field: str,
    config: dict[str, Any],
    best_window: dict[str, Any],
    best_model: dict[str, Any],
    selected_budget: int | str,
) -> pd.DataFrame:
    group_values = sorted({getattr(record, group_field) for record in records})
    if config["execution"].get("quick_mode", False):
        # One held-out group is enough to exercise the code path in smoke mode.
        # The official execution still evaluates every session and quadrant.
        group_values = group_values[:1]
    rows: list[dict[str, Any]] = []

    for group_index, held_out in enumerate(group_values):
        train_records = [
            record for record in records if getattr(record, group_field) != held_out
        ]
        test_records = [
            record for record in records if getattr(record, group_field) == held_out
        ]
        if not train_records or not test_records:
            continue

        split = prepare_feature_split(
            train_records,
            test_records,
            config,
            int(best_window["window_packets"]),
            int(best_window["step_packets"]),
            seed=1000 + group_index,
        )
        result, selected, _, _ = evaluate_split_with_model(
            split,
            best_model["name"],
            best_model["parameters"],
            selected_budget,
            seed=1000 + group_index,
        )
        rows.append(
            {
                "held_out_group": held_out,
                "group_field": group_field,
                "train_files": len(train_records),
                "test_files": len(test_records),
                "train_windows": len(split.y_train),
                "test_windows": len(split.y_test),
                "selected_feature_count": len(selected),
                "selected_subcarrier_count": len(
                    split.preprocessing.selected_subcarriers
                ),
                **result.metrics,
            }
        )

    return pd.DataFrame(rows)


def run_generalization_holdouts(
    records: Sequence[FileRecord],
    config: dict[str, Any],
    best_window: dict[str, Any],
    best_model: dict[str, Any],
    selected_budget: int | str,
    paths: dict[str, Path],
    experiment_index: list[dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Stage 7/8: session and quadrant holdouts.")
    session_results = pd.DataFrame()
    quadrant_results = pd.DataFrame()

    if config["validation"].get("run_session_holdout", True):
        session_results = grouped_holdout_evaluation(
            records,
            "session",
            config,
            best_window,
            best_model,
            selected_budget,
        )
        save_dataframe(paths["tables"] / "session_holdout_results.csv", session_results)
        if not session_results.empty:
            plot_grouped_metric(
                session_results,
                "held_out_group",
                "macro_f1",
                None,
                "Generalization to an unseen acquisition session",
                "Held-out session",
                "Macro F1",
                paths["figures"] / "session_holdout_macro_f1.png",
            )
            experiment_index.append(
                {
                    "stage": "session_holdout",
                    "objective": "Evaluate generalization to an unseen session/day.",
                    "main_table": "tables/session_holdout_results.csv",
                    "main_figure": "figures/session_holdout_macro_f1.png",
                }
            )

    if config["validation"].get("run_quadrant_holdout", True):
        quadrant_results = grouped_holdout_evaluation(
            records,
            "quadrant",
            config,
            best_window,
            best_model,
            selected_budget,
        )
        save_dataframe(
            paths["tables"] / "quadrant_holdout_results.csv",
            quadrant_results,
        )
        if not quadrant_results.empty:
            plot_grouped_metric(
                quadrant_results,
                "held_out_group",
                "macro_f1",
                None,
                "Generalization to an unseen quadrant",
                "Held-out quadrant",
                "Macro F1",
                paths["figures"] / "quadrant_holdout_macro_f1.png",
            )
            experiment_index.append(
                {
                    "stage": "quadrant_holdout",
                    "objective": "Evaluate generalization to an unseen room position.",
                    "main_table": "tables/quadrant_holdout_results.csv",
                    "main_figure": "figures/quadrant_holdout_macro_f1.png",
                }
            )

    return session_results, quadrant_results


# ---------------------------------------------------------------------------
# Final fit and deployment export
# ---------------------------------------------------------------------------

def fit_complete_dataset(
    records: Sequence[FileRecord],
    config: dict[str, Any],
    best_window: dict[str, Any],
    best_model: dict[str, Any],
    selected_budget: int | str,
) -> tuple[
    BaseEstimator,
    PreprocessingState,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    state = fit_preprocessing_state(
        records,
        config,
        seed=int(config["execution"].get("random_state_for_correlation_sampling", 2026)),
    )
    x, y, metadata = create_feature_matrix(
        records,
        state,
        int(best_window["window_packets"]),
        int(best_window["step_packets"]),
    )
    selected_indices, scores = select_feature_indices(x, y, selected_budget)
    model = create_model(
        best_model["name"],
        best_model["parameters"],
        seed=2026,
    )
    if model is None:
        raise RuntimeError(f"Final model unavailable: {best_model['name']}")
    model.fit(x[:, selected_indices], y)
    return model, state, selected_indices, scores, x, metadata


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_sklearn_tree(tree: Any) -> dict[str, Any]:
    return {
        "node_count": int(tree.node_count),
        "children_left": tree.children_left.astype(int).tolist(),
        "children_right": tree.children_right.astype(int).tolist(),
        "feature": tree.feature.astype(int).tolist(),
        "threshold": tree.threshold.astype(float).tolist(),
        "value": tree.value.astype(float).tolist(),
        "impurity": tree.impurity.astype(float).tolist(),
        "n_node_samples": tree.n_node_samples.astype(int).tolist(),
    }


def export_model_structure(model: BaseEstimator) -> dict[str, Any]:
    if isinstance(model, DecisionTreeClassifier):
        return {
            "model_type": "decision_tree",
            "classes": model.classes_.astype(int).tolist(),
            "tree": export_sklearn_tree(model.tree_),
        }

    if isinstance(model, GradientBoostingClassifier):
        trees: list[list[dict[str, Any]]] = []
        for stage in model.estimators_:
            trees.append([export_sklearn_tree(estimator.tree_) for estimator in stage])

        initial_prior = getattr(model.init_, "class_prior_", None)
        return {
            "model_type": "gradient_boosting_classifier",
            "classes": model.classes_.astype(int).tolist(),
            "learning_rate": float(model.learning_rate),
            "n_estimators": int(model.n_estimators),
            "n_classes": int(model.n_classes_),
            "initial_class_prior": (
                np.asarray(initial_prior, dtype=float).tolist()
                if initial_prior is not None
                else None
            ),
            "trees_by_stage_and_class": trees,
        }

    if isinstance(model, (RandomForestClassifier, ExtraTreesClassifier)):
        return {
            "model_type": model.__class__.__name__,
            "classes": model.classes_.astype(int).tolist(),
            "n_estimators": len(model.estimators_),
            "trees": [export_sklearn_tree(estimator.tree_) for estimator in model.estimators_],
        }

    if isinstance(model, Pipeline):
        scaler = model.named_steps.get("scaler")
        classifier = model.named_steps.get("classifier")
        if isinstance(classifier, LogisticRegression):
            return {
                "model_type": "logistic_regression_pipeline",
                "classes": classifier.classes_.astype(int).tolist(),
                "scaler_mean": np.asarray(scaler.mean_, dtype=float).tolist(),
                "scaler_scale": np.asarray(scaler.scale_, dtype=float).tolist(),
                "coefficients": classifier.coef_.astype(float).tolist(),
                "intercepts": classifier.intercept_.astype(float).tolist(),
            }

    if model.__class__.__module__.startswith("xgboost"):
        booster = model.get_booster()
        return {
            "model_type": "xgboost_classifier",
            "classes": np.asarray(model.classes_, dtype=int).tolist(),
            "booster_configuration": json.loads(booster.save_config()),
            "trees": [json.loads(item) for item in booster.get_dump(dump_format="json")],
        }

    return {
        "model_type": model.__class__.__name__,
        "portable_structure_available": False,
    }


def export_final_artifacts(
    records: Sequence[FileRecord],
    config: dict[str, Any],
    sampling_rate_hz: float,
    best_window: dict[str, Any],
    best_model: dict[str, Any],
    selected_budget: int | str,
    final_summary: pd.DataFrame,
    paths: dict[str, Path],
) -> tuple[dict[str, Any], pd.DataFrame]:
    LOGGER.info("Stage 8/8: final fit and realtime export.")
    model, state, selected_indices, scores, x_all, metadata = fit_complete_dataset(
        records,
        config,
        best_window,
        best_model,
        selected_budget,
    )

    output_config = config["outputs"]
    processed_root = paths["processed"]
    model_path = processed_root / output_config["realtime_model_file"]
    structure_path = processed_root / output_config[
        "realtime_model_structure_file"
    ]
    realtime_config_path = processed_root / output_config[
        "realtime_config_file"
    ]
    bundle_path = processed_root / output_config["training_bundle_file"]

    joblib.dump(model, model_path)
    model_structure = export_model_structure(model)
    save_json(structure_path, model_structure)

    feature_names = config["features"].get(
        "names_per_subcarrier", FEATURE_NAMES_DEFAULT
    )
    selected_feature_rows: list[dict[str, Any]] = []
    for rank, feature_index in enumerate(selected_indices, start=1):
        descriptor = feature_descriptor(
            int(feature_index),
            state.selected_subcarriers,
            feature_names,
        )
        descriptor.update(
            {
                "rank": rank,
                "fisher_score": float(scores[int(feature_index)]),
            }
        )
        selected_feature_rows.append(descriptor)
    selected_features = pd.DataFrame(selected_feature_rows)
    save_dataframe(paths["tables"] / "selected_features_final.csv", selected_features)

    validation_summary = (
        final_summary.iloc[0].to_dict() if not final_summary.empty else {}
    )
    validation_summary = json_safe_mapping(validation_summary)

    realtime_config: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": now_utc_iso(),
        "purpose": "PC realtime inference configuration for Dataset v2",
        "dataset": {
            "root_used_for_training": str(
                resolve_project_path(config["dataset"]["root"])
            ),
            "file_count": len(records),
            "window_count_for_final_fit": int(x_all.shape[0]),
            "sessions": sorted({record.session for record in records}),
            "quadrants": sorted({record.quadrant for record in records}),
            "class_file_counts": dict(Counter(record.label for record in records)),
        },
        "input": {
            "sampling_rate_hz": float(sampling_rate_hz),
            "expected_subcarriers": int(records[0].subcarrier_count),
            "class_order": CLASS_ORDER,
        },
        "window": {
            "size_seconds": float(best_window["window_seconds"]),
            "step_seconds": float(best_window["step_seconds"]),
            "size_packets": int(best_window["window_packets"]),
            "step_packets": int(best_window["step_packets"]),
        },
        "preprocessing": {
            "hampel_radius": int(config["preprocessing"]["hampel_radius"]),
            "hampel_n_sigmas": float(
                config["preprocessing"]["hampel_n_sigmas"]
            ),
            "moving_average_window": int(
                config["preprocessing"]["moving_average_window"]
            ),
            "minimum_informative_std": float(
                config["preprocessing"]["minimum_informative_std"]
            ),
            "correlation_threshold": float(
                config["preprocessing"]["correlation_threshold"]
            ),
            "zscore_means": state.means.astype(float).tolist(),
            "zscore_stds": state.stds.astype(float).tolist(),
            "informative_subcarriers": state.informative_indices.astype(int).tolist(),
            "selected_subcarriers": state.selected_subcarriers.astype(int).tolist(),
        },
        "features": {
            "names_per_subcarrier": feature_names,
            "full_feature_count": int(x_all.shape[1]),
            "selection_method": "Fisher Score fitted on complete Dataset v2",
            "top_k": (
                "all" if selected_budget == "all" else int(selected_budget)
            ),
            "selected_indices": selected_indices.astype(int).tolist(),
            "selected_descriptors": selected_feature_rows,
        },
        "classifier": {
            "name": best_model["name"],
            "parameters": best_model["parameters"],
            "serialized_model_file": model_path.name,
            "serialized_model_sha256": file_sha256(model_path),
            "portable_structure_file": structure_path.name,
            "prediction_output": "integer class index mapped through input.class_order",
        },
        "validation_summary": validation_summary,
        "notes": [
            "The serialized model file is required for immediate PC realtime use.",
            "The JSON structure export is intended for inspection and future embedded translation.",
            "Do not refit normalization, correlation filtering or Fisher selection during inference.",
        ],
    }
    save_json(realtime_config_path, realtime_config)

    # Keep a human-readable deployment manifest in a versionable path.
    # The serialized joblib model remains under datasets/processed because it
    # is a generated binary artifact, while these JSON files can be reviewed
    # and committed together with the realtime implementation.
    quick_mode = bool(config["execution"].get("quick_mode", False))
    versioned_config_value = (
        None
        if quick_mode
        else output_config.get("versioned_realtime_config")
    )
    versioned_structure_value = (
        None
        if quick_mode
        else output_config.get("versioned_realtime_model_structure")
    )
    if versioned_config_value:
        versioned_config_path = resolve_project_path(versioned_config_value)
        save_json(versioned_config_path, realtime_config)
    else:
        versioned_config_path = None

    if versioned_structure_value:
        versioned_structure_path = resolve_project_path(
            versioned_structure_value
        )
        save_json(versioned_structure_path, model_structure)
    else:
        versioned_structure_path = None

    training_bundle = {
        "model": model,
        "realtime_config": realtime_config,
        "preprocessing_state": state,
        "selected_feature_indices": selected_indices,
        "selected_feature_scores": scores[selected_indices],
        "training_metadata": metadata,
    }
    joblib.dump(training_bundle, bundle_path)

    save_json(
        paths["reports"] / "final_artifact_manifest.json",
        {
            "realtime_config": str(realtime_config_path),
            "serialized_model": str(model_path),
            "model_structure": str(structure_path),
            "training_bundle": str(bundle_path),
            "versioned_realtime_config": (
                str(versioned_config_path) if versioned_config_path else None
            ),
            "versioned_model_structure": (
                str(versioned_structure_path)
                if versioned_structure_path
                else None
            ),
        },
    )
    return realtime_config, selected_features


def json_safe_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, (np.integer,)):
            output[key] = int(value)
        elif isinstance(value, (np.floating,)):
            output[key] = float(value)
        elif pd.isna(value):
            output[key] = None
        else:
            output[key] = value
    return output


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_grouped_metric(
    dataframe: pd.DataFrame,
    label_column: str,
    metric_column: str,
    error_column: str | None,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if dataframe.empty or metric_column not in dataframe.columns:
        return

    labels = dataframe[label_column].astype(str).tolist()
    values = dataframe[metric_column].astype(float).tolist()
    errors = (
        dataframe[error_column].fillna(0).astype(float).tolist()
        if error_column and error_column in dataframe.columns
        else None
    )

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(labels, values, yerr=errors, capsize=4 if errors else 0)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, 1)
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_window_step(dataframe: pd.DataFrame, output_path: Path) -> None:
    if dataframe.empty:
        return

    plot_frame = dataframe.copy()
    plot_frame["configuration"] = plot_frame.apply(
        lambda row: f"{row['window_seconds']:.2f}s / {row['step_seconds']:.2f}s",
        axis=1,
    )
    plot_grouped_metric(
        plot_frame,
        "configuration",
        "macro_f1_mean",
        "macro_f1_std",
        "Window and step comparison",
        "Window / step",
        "Macro F1",
        output_path,
    )


def plot_seed_stability(dataframe: pd.DataFrame, output_path: Path) -> None:
    if dataframe.empty:
        return

    ordered = dataframe.sort_values("seed")
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(ordered["seed"], ordered["macro_f1"], marker="o", label="Macro F1")
    axis.plot(
        ordered["seed"],
        ordered["movement_f1"],
        marker="o",
        label="Movement F1",
    )
    axis.set_title("Final-candidate stability across file splits")
    axis.set_xlabel("Seed")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: Sequence[str],
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix)
    figure.colorbar(image, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=30, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels=labels)

    threshold = float(matrix.max()) / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(int(matrix[row, column])),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_class_f1(class_metrics: pd.DataFrame, output_path: Path) -> None:
    plot_grouped_metric(
        class_metrics,
        "class",
        "f1",
        None,
        "Per-class F1 for aggregated repeated holdout",
        "Class",
        "F1",
        output_path,
    )


def plot_dataset_file_counts(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    counts = (
        summary.groupby(["session", "quadrant", "label"])
        .size()
        .reset_index(name="file_count")
    )
    counts["group"] = counts.apply(
        lambda row: f"{row['session']}|{row['quadrant']}|{row['label']}",
        axis=1,
    )
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.bar(counts["group"], counts["file_count"])
    axis.set_title("Dataset v2 file distribution")
    axis.set_xlabel("Session | quadrant | class")
    axis.set_ylabel("Files")
    axis.tick_params(axis="x", rotation=75)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_sampling_rate(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(summary["sampling_rate_hz"].dropna(), bins=20)
    axis.set_title("Sampling-rate distribution across Dataset v2 files")
    axis.set_xlabel("Sampling rate (Hz)")
    axis.set_ylabel("Files")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


# ---------------------------------------------------------------------------
# Report and experiment index
# ---------------------------------------------------------------------------

def markdown_table(dataframe: pd.DataFrame, max_rows: int = 20) -> str:
    if dataframe.empty:
        return "_No results generated._"

    frame = dataframe.head(max_rows).copy()
    for column in frame.columns:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )

    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    if len(dataframe) > max_rows:
        lines.append(f"\n_Only the first {max_rows} rows are shown._")
    return "\n".join(lines)


def generate_report(
    records: Sequence[FileRecord],
    sampling_rate_hz: float,
    best_window: dict[str, Any],
    best_tree_parameters: dict[str, Any],
    best_model: dict[str, Any],
    selected_budget: int | str,
    model_summary: pd.DataFrame,
    budget_summary: pd.DataFrame,
    final_summary: pd.DataFrame,
    binary_summary: pd.DataFrame,
    hierarchical_summary: pd.DataFrame,
    session_results: pd.DataFrame,
    quadrant_results: pd.DataFrame,
    realtime_config: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    report_path = paths["reports"] / "dataset_v2_retraining_report.md"
    counts = Counter(record.label for record in records)
    sessions = sorted({record.session for record in records})
    quadrants = sorted({record.quadrant for record in records})

    text = f"""# Dataset v2 retraining report

Generated at: `{now_utc_iso()}`

## Dataset

- Accepted acquisition files: **{len(records)}**
- Sessions: **{', '.join(sessions)}**
- Quadrants: **{', '.join(quadrants)}**
- Empty files: **{counts.get('empty', 0)}**
- Static-presence files: **{counts.get('static_presence', 0)}**
- Movement files: **{counts.get('movement', 0)}**
- Median effective sampling rate used for temporal conversion: **{sampling_rate_hz:.3f} Hz**

## Methodological safeguards

- Splitting is performed by acquisition file, never by isolated windows.
- Repeated file holdout is stratified by class, session and quadrant.
- Z-score calibration is fitted only on training files in each split.
- Correlation filtering is fitted only on training files in each split.
- Fisher Score is calculated only on training windows in each split.
- Session and quadrant holdouts are reported separately from ordinary file holdout.

## Selected temporal configuration

- Window: **{float(best_window['window_seconds']):.3f} s** / **{int(best_window['window_packets'])} packets**
- Step: **{float(best_window['step_seconds']):.3f} s** / **{int(best_window['step_packets'])} packets**

## Decision-tree baseline tuning

Selected parameters: `{json.dumps(best_tree_parameters, ensure_ascii=False)}`

## Classifier comparison

{markdown_table(model_summary)}

Selected classifier: **{best_model['name']}**

Parameters:

```json
{json.dumps(best_model['parameters'], indent=2, ensure_ascii=False)}
```

## Feature budget

{markdown_table(budget_summary)}

Selected Fisher budget: **{selected_budget}**

## Repeated file-holdout result

{markdown_table(final_summary)}

The mean repeated-holdout metrics above are the main estimates for new acquisition files under sessions and quadrants represented in training.

## Binary diagnostics

{markdown_table(binary_summary)}

## Direct versus hierarchical classification

{markdown_table(hierarchical_summary)}

## Session holdout

{markdown_table(session_results)}

## Quadrant holdout

{markdown_table(quadrant_results)}

## Realtime artifacts

- Configuration: `{realtime_config['classifier']['serialized_model_file']}` is referenced by `realtime_pipeline_config.json`.
- Portable model structure: `{realtime_config['classifier']['portable_structure_file']}`.
- Output class order: `{', '.join(realtime_config['input']['class_order'])}`.

The final model is fitted with all labeled Dataset v2 files only after the validation stages are complete. Performance must be reported from the validation tables, not from the final all-data fit.
"""

    report_path.write_text(text, encoding="utf-8")


def save_experiment_index(
    experiment_index: Sequence[dict[str, str]],
    path: Path,
) -> None:
    dataframe = pd.DataFrame(experiment_index)
    dataframe.insert(0, "order", np.arange(1, len(dataframe) + 1))
    save_dataframe(path, dataframe)


# ---------------------------------------------------------------------------
# Extended Dataset v2 experiment suite
# ---------------------------------------------------------------------------


def quick_subset_by_stratum(
    records: Sequence[FileRecord],
    config: dict[str, Any],
) -> list[FileRecord]:
    """Keep a small but valid class/session/quadrant subset for smoke tests."""
    if not config["execution"].get("quick_mode", False):
        return list(records)

    limit = int(config["execution"].get("quick_files_per_stratum", 2))
    seed = int(config["execution"].get("quick_subset_seed", 2026))
    rng = np.random.default_rng(seed)
    fields = config["validation"]["stratify_by"]
    groups: dict[str, list[FileRecord]] = defaultdict(list)
    for record in records:
        groups[make_stratum(record, fields)].append(record)

    selected: list[FileRecord] = []
    for stratum in sorted(groups):
        items = groups[stratum]
        if len(items) < 2:
            raise ValueError(
                "Quick mode still requires at least two files per stratum. "
                f"Problematic stratum: {stratum}"
            )
        indices = rng.permutation(len(items))[: min(limit, len(items))]
        selected.extend(items[int(index)] for index in indices)

    LOGGER.info(
        "Quick subset selected %d/%d files (%d per stratum maximum).",
        len(selected),
        len(records),
        limit,
    )
    return selected


def normalized_top_k(value: Any) -> int | str:
    if isinstance(value, str) and value.lower() == "all":
        return "all"
    return int(value)


def top_k_label(value: int | str) -> str:
    return "all" if value == "all" else str(int(value))


def build_model_from_spec(
    spec: dict[str, Any],
    seed: int,
    num_classes: int = 3,
    skip_xgboost: bool = False,
) -> BaseEstimator | None:
    family = str(spec["family"])
    parameters = dict(spec.get("parameters", {}))

    if family == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, **parameters)
    if family == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **parameters)
    if family == "extra_trees":
        return ExtraTreesClassifier(random_state=seed, n_jobs=1, **parameters)
    if family == "gradient_boosting":
        return GradientBoostingClassifier(random_state=seed, **parameters)
    if family == "knn":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(**parameters)),
            ]
        )
    if family == "linear_svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LinearSVC(random_state=seed, **parameters),
                ),
            ]
        )
    if family == "rbf_svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(random_state=seed, **parameters)),
            ]
        )
    if family == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(random_state=seed, **parameters),
                ),
            ]
        )
    if family == "xgboost":
        if skip_xgboost:
            return None
        model = optional_xgboost_classifier(parameters)
        if model is not None:
            model.set_params(random_state=seed, num_class=num_classes)
        return model
    raise ValueError(f"Unsupported model family in specification: {family}")


def unwrap_estimator(model: BaseEstimator) -> BaseEstimator:
    if isinstance(model, Pipeline):
        return model.named_steps.get("classifier", model)
    return model


def _tree_estimators(model: BaseEstimator) -> list[Any]:
    estimator = unwrap_estimator(model)
    if isinstance(estimator, DecisionTreeClassifier):
        return [estimator]
    if isinstance(estimator, GradientBoostingClassifier):
        return list(estimator.estimators_.ravel())
    if isinstance(estimator, (RandomForestClassifier, ExtraTreesClassifier)):
        return list(estimator.estimators_)
    return []


def _xgboost_dump_stats(model: BaseEstimator) -> tuple[int, int, int, float]:
    estimator = unwrap_estimator(model)
    if not estimator.__class__.__module__.startswith("xgboost"):
        return 0, 0, 0, 0.0

    booster = estimator.get_booster()
    dumps = booster.get_dump(dump_format="json")
    node_count = 0
    used_features: set[int] = set()
    leaf_depths: list[int] = []

    def walk(node: dict[str, Any], depth: int) -> None:
        nonlocal node_count
        node_count += 1
        if "leaf" in node:
            leaf_depths.append(depth)
            return
        split = str(node.get("split", ""))
        if split.startswith("f") and split[1:].isdigit():
            used_features.add(int(split[1:]))
        for child in node.get("children", []):
            walk(child, depth + 1)

    for raw_tree in dumps:
        walk(json.loads(raw_tree), 0)

    mean_leaf_depth = float(np.mean(leaf_depths)) if leaf_depths else 0.0
    comparisons = mean_leaf_depth * len(dumps)
    return len(dumps), node_count, len(used_features), comparisons


def model_complexity_metrics(
    model: BaseEstimator,
    x_reference: np.ndarray,
) -> dict[str, Any]:
    estimator = unwrap_estimator(model)
    trees = _tree_estimators(model)
    if trees:
        node_count = sum(int(tree.tree_.node_count) for tree in trees)
        used_features: set[int] = set()
        max_depth = 0
        for tree in trees:
            feature_values = tree.tree_.feature
            used_features.update(int(v) for v in feature_values if int(v) >= 0)
            max_depth = max(max_depth, int(tree.tree_.max_depth))

        sample = x_reference[: min(500, len(x_reference))]
        comparisons = np.zeros(len(sample), dtype=float)
        if len(sample) > 0:
            for tree in trees:
                path = tree.decision_path(sample)
                comparisons += np.asarray(path.sum(axis=1)).ravel() - 1.0
        mean_comparisons = float(np.mean(comparisons)) if len(sample) else 0.0
        return {
            "actual_tree_count": len(trees),
            "actual_node_count": node_count,
            "actual_max_depth": max_depth,
            "used_feature_count": len(used_features),
            "mean_comparisons_per_prediction": mean_comparisons,
        }

    if estimator.__class__.__module__.startswith("xgboost"):
        tree_count, node_count, used_count, comparisons = _xgboost_dump_stats(model)
        return {
            "actual_tree_count": tree_count,
            "actual_node_count": node_count,
            "actual_max_depth": int(getattr(estimator, "max_depth", 0) or 0),
            "used_feature_count": used_count,
            "mean_comparisons_per_prediction": comparisons,
        }

    if isinstance(estimator, LogisticRegression):
        coefficients = np.asarray(estimator.coef_)
        used = np.flatnonzero(np.any(np.abs(coefficients) > 1e-12, axis=0))
        return {
            "actual_tree_count": 0,
            "actual_node_count": 0,
            "actual_max_depth": 0,
            "used_feature_count": int(len(used)),
            "mean_comparisons_per_prediction": float(coefficients.size),
        }

    return {
        "actual_tree_count": 0,
        "actual_node_count": 0,
        "actual_max_depth": 0,
        "used_feature_count": int(x_reference.shape[1]),
        "mean_comparisons_per_prediction": float(x_reference.shape[1]),
    }


def detailed_model_evaluation(
    model: BaseEstimator,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    fit_started = perf_counter()
    model.fit(x_train, y_train)
    fit_time_ms = (perf_counter() - fit_started) * 1000.0

    prediction_started = perf_counter()
    y_pred = np.asarray(model.predict(x_test), dtype=int)
    probabilities = normalize_probabilities(model, x_test, len(CLASS_ORDER))
    prediction_time_ms = (perf_counter() - prediction_started) * 1000.0

    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=np.arange(len(CLASS_ORDER)),
        zero_division=0,
    )
    y_binary = label_binarize(y_test, classes=np.arange(len(CLASS_ORDER)))

    per_class_auc: list[float] = []
    per_class_ap: list[float] = []
    for class_index in range(len(CLASS_ORDER)):
        binary_target = y_binary[:, class_index]
        if len(np.unique(binary_target)) < 2:
            per_class_auc.append(float("nan"))
            per_class_ap.append(float("nan"))
            continue
        per_class_auc.append(
            float(roc_auc_score(binary_target, probabilities[:, class_index]))
        )
        per_class_ap.append(
            float(
                average_precision_score(
                    binary_target,
                    probabilities[:, class_index],
                )
            )
        )

    try:
        auc_macro = float(
            roc_auc_score(
                y_test,
                probabilities,
                labels=np.arange(len(CLASS_ORDER)),
                multi_class="ovr",
                average="macro",
            )
        )
        auc_weighted = float(
            roc_auc_score(
                y_test,
                probabilities,
                labels=np.arange(len(CLASS_ORDER)),
                multi_class="ovr",
                average="weighted",
            )
        )
    except ValueError:
        auc_macro = float("nan")
        auc_weighted = float("nan")

    try:
        ap_macro = float(
            average_precision_score(y_binary, probabilities, average="macro")
        )
        ap_weighted = float(
            average_precision_score(y_binary, probabilities, average="weighted")
        )
    except ValueError:
        ap_macro = float("nan")
        ap_weighted = float("nan")

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(
            f1_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "roc_auc_ovr_macro": auc_macro,
        "roc_auc_ovr_weighted": auc_weighted,
        "average_precision_macro": ap_macro,
        "average_precision_weighted": ap_weighted,
        "empty_f1": float(class_f1[0]),
        "static_presence_f1": float(class_f1[1]),
        "movement_f1": float(class_f1[2]),
        "fit_time_ms": fit_time_ms,
        "prediction_time_ms": prediction_time_ms,
        "prediction_time_us_per_window": (
            prediction_time_ms * 1000.0 / max(1, len(y_test))
        ),
        "test_samples": int(len(y_test)),
    }
    metrics.update(model_complexity_metrics(model, x_test))

    class_metrics = pd.DataFrame(
        {
            "class_index": np.arange(len(CLASS_ORDER)),
            "class": CLASS_ORDER,
            "precision": precision,
            "recall": recall,
            "f1": class_f1,
            "support": support,
            "roc_auc": per_class_auc,
            "average_precision": per_class_ap,
        }
    )
    confusion = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(CLASS_ORDER)),
    )
    return metrics, class_metrics, confusion, y_pred, probabilities


def expand_specs_with_budgets(
    base_specs: Sequence[dict[str, Any]],
    budgets: Sequence[int | str],
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for base in base_specs:
        for raw_budget in budgets:
            budget = normalized_top_k(raw_budget)
            item = copy.deepcopy(base)
            item["top_k"] = budget
            item["name"] = f"{base['name']}_top{top_k_label(budget)}"
            expanded.append(item)
    return expanded


def run_model_specs(
    records: Sequence[FileRecord],
    config: dict[str, Any],
    best_window: dict[str, Any],
    specs: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    split_cache: dict[tuple[int, int, int], FeatureSplit],
    include_predictions: bool = False,
    skip_xgboost: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    window_packets = int(best_window["window_packets"])
    step_packets = int(best_window["step_packets"])

    for seed in seeds:
        LOGGER.info("Evaluating model grid for seed %s.", seed)
        split = get_cached_file_feature_split(
            records,
            config,
            window_packets,
            step_packets,
            int(seed),
            split_cache,
        )
        scores = fisher_scores(split.x_train, split.y_train)
        ranking = np.argsort(scores)[::-1]

        for spec in specs:
            LOGGER.info("Model start: %s", spec["name"])
            budget = normalized_top_k(spec.get("top_k", "all"))
            if budget == "all":
                selected = ranking
            else:
                selected = ranking[: min(int(budget), split.x_train.shape[1])]

            model = build_model_from_spec(
                spec,
                seed=int(seed),
                skip_xgboost=skip_xgboost,
            )
            if model is None:
                LOGGER.warning("Skipping unavailable model: %s", spec["name"])
                continue

            metrics, class_metrics, confusion, y_pred, probabilities = (
                detailed_model_evaluation(
                    model,
                    split.x_train[:, selected],
                    split.y_train,
                    split.x_test[:, selected],
                    split.y_test,
                )
            )
            common = base_experiment_row(int(seed), split, selected)
            common.update(
                {
                    "model_name": str(spec["name"]),
                    "family": str(spec["family"]),
                    "top_k": top_k_label(budget),
                    "top_k_sort": (
                        int(split.x_train.shape[1])
                        if budget == "all"
                        else int(budget)
                    ),
                    "parameters_json": json.dumps(
                        spec.get("parameters", {}),
                        sort_keys=True,
                    ),
                }
            )
            metric_rows.append({**common, **metrics})
            LOGGER.info("Model done: %s | macro_f1=%.4f", spec["name"], metrics["macro_f1"])

            for row in class_metrics.to_dict(orient="records"):
                class_rows.append({**common, **row})

            for true_index, true_name in enumerate(CLASS_ORDER):
                for predicted_index, predicted_name in enumerate(CLASS_ORDER):
                    confusion_rows.append(
                        {
                            **common,
                            "true_class": true_name,
                            "predicted_class": predicted_name,
                            "count": int(confusion[true_index, predicted_index]),
                        }
                    )

            for rank, feature_index in enumerate(selected, start=1):
                selection_rows.append(
                    {
                        "seed": int(seed),
                        "model_name": str(spec["name"]),
                        "family": str(spec["family"]),
                        "top_k": top_k_label(budget),
                        "rank": rank,
                        "feature_index": int(feature_index),
                        "fisher_score": float(scores[int(feature_index)]),
                    }
                )

            if include_predictions:
                predictions = split.test_metadata.copy()
                predictions.insert(0, "seed", int(seed))
                predictions.insert(1, "model_name", str(spec["name"]))
                predictions["true_class"] = [
                    CLASS_ORDER[index] for index in split.y_test
                ]
                predictions["predicted_class"] = [
                    CLASS_ORDER[index] for index in y_pred
                ]
                predictions["correct"] = predictions["true_class"] == predictions[
                    "predicted_class"
                ]
                predictions["confidence"] = np.max(probabilities, axis=1)
                for class_index, class_name in enumerate(CLASS_ORDER):
                    predictions[f"probability_{class_name}"] = probabilities[
                        :, class_index
                    ]
                prediction_frames.append(predictions)

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(class_rows),
        pd.DataFrame(confusion_rows),
        pd.DataFrame(selection_rows),
        (
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else pd.DataFrame()
        ),
    )


def summarize_model_results(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    group_columns = [
        "model_name",
        "family",
        "top_k",
        "top_k_sort",
        "parameters_json",
    ]
    metric_columns = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "roc_auc_ovr_macro",
        "roc_auc_ovr_weighted",
        "average_precision_macro",
        "average_precision_weighted",
        "empty_f1",
        "static_presence_f1",
        "movement_f1",
        "fit_time_ms",
        "prediction_time_ms",
        "prediction_time_us_per_window",
        "actual_tree_count",
        "actual_node_count",
        "actual_max_depth",
        "used_feature_count",
        "mean_comparisons_per_prediction",
        "selected_feature_count",
        "selected_subcarrier_count",
        "train_windows",
        "test_windows",
    ]
    available = [column for column in metric_columns if column in detail.columns]
    grouped = detail.groupby(group_columns, dropna=False)[available]
    means = grouped.mean().add_suffix("_mean")
    stds = grouped.std(ddof=0).add_suffix("_std")
    return means.join(stds).reset_index()


def order_summary_by_policy(
    summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    if summary.empty:
        return summary
    policy = config["selection_policy"]
    best_macro = float(summary["macro_f1_mean"].max())
    tolerance = float(policy.get("macro_f1_tolerance", 0.005))
    eligible = summary[
        summary["macro_f1_mean"] >= best_macro - tolerance
    ].copy()

    sort_columns: list[str] = []
    ascending: list[bool] = []
    if policy.get("prefer_movement_f1", True):
        sort_columns.append("movement_f1_mean")
        ascending.append(False)
    if "roc_auc_ovr_macro_mean" in eligible.columns:
        sort_columns.append("roc_auc_ovr_macro_mean")
        ascending.append(False)
    if policy.get("prefer_lower_std", True):
        sort_columns.append("macro_f1_std")
        ascending.append(True)
    if policy.get("prefer_fewer_features", True):
        sort_columns.append("selected_feature_count_mean")
        ascending.append(True)
    if policy.get("prefer_fewer_comparisons", True):
        sort_columns.append("mean_comparisons_per_prediction_mean")
        ascending.append(True)

    if sort_columns:
        eligible = eligible.sort_values(sort_columns, ascending=ascending)
    selected_index = eligible.index[0]
    remaining = summary.drop(index=selected_index).sort_values(
        ["macro_f1_mean", "movement_f1_mean", "macro_f1_std"],
        ascending=[False, False, True],
    )
    ordered = pd.concat(
        [summary.loc[[selected_index]], remaining],
        ignore_index=True,
    )
    ordered.insert(
        0,
        "selected_by_policy",
        [True] + [False] * (len(ordered) - 1),
    )
    return ordered


def save_grid_outputs(
    prefix: str,
    detail: pd.DataFrame,
    per_class: pd.DataFrame,
    confusion: pd.DataFrame,
    selections: pd.DataFrame,
    summary: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    save_dataframe(paths["tables"] / f"{prefix}_results_by_seed.csv", detail)
    save_dataframe(paths["tables"] / f"{prefix}_summary.csv", summary)
    save_dataframe(paths["tables"] / f"{prefix}_per_class_by_seed.csv", per_class)
    save_dataframe(paths["tables"] / f"{prefix}_confusion_by_seed.csv", confusion)
    save_dataframe(
        paths["tables"] / f"{prefix}_feature_selection_by_seed.csv",
        selections,
    )


def plot_model_summary(
    summary: pd.DataFrame,
    title: str,
    output_path: Path,
    max_models: int = 30,
) -> None:
    if summary.empty:
        return
    frame = summary.head(max_models).copy()
    frame["display"] = frame.apply(
        lambda row: f"{row['model_name']} (K={row['top_k']})",
        axis=1,
    )
    plot_grouped_metric(
        frame,
        "display",
        "macro_f1_mean",
        "macro_f1_std",
        title,
        "Model",
        "Macro F1",
        output_path,
    )


def correlation_threshold_search_v2(
    records: Sequence[FileRecord],
    config: dict[str, Any],
    sampling_rate_hz: float,
    paths: dict[str, Path],
    experiment_index: list[dict[str, str]],
    skip_xgboost: bool,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Extended stage: correlation-threshold search.")
    thresholds = list(
        config["preprocessing"].get(
            "correlation_threshold_candidates",
            [config["preprocessing"]["correlation_threshold"]],
        )
    )
    seeds = list(config["validation"]["search_seeds"])
    if config["execution"].get("quick_mode", False):
        thresholds = thresholds[:2]
        seeds = seeds[:1]

    reference = config["validation"]["correlation_reference_window_seconds"]
    best_window = {
        "window_seconds": float(reference["window"]),
        "step_seconds": float(reference["step"]),
        "window_packets": seconds_to_packets(
            float(reference["window"]), sampling_rate_hz
        ),
        "step_packets": max(
            1,
            int(round(float(reference["step"]) * sampling_rate_hz)),
        ),
    }
    reference_spec = copy.deepcopy(config["models"]["reference_model"])
    rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        local_config = copy.deepcopy(config)
        local_config["preprocessing"]["correlation_threshold"] = float(threshold)
        local_cache: dict[tuple[int, int, int], FeatureSplit] = {}
        for seed in seeds:
            split = get_cached_file_feature_split(
                records,
                local_config,
                int(best_window["window_packets"]),
                int(best_window["step_packets"]),
                int(seed),
                local_cache,
            )
            spec = copy.deepcopy(reference_spec)
            model = build_model_from_spec(
                spec,
                int(seed),
                skip_xgboost=skip_xgboost,
            )
            if model is None:
                raise RuntimeError("Reference model is unavailable.")
            budget = normalized_top_k(spec.get("top_k", 126))
            selected, _ = select_feature_indices(
                split.x_train,
                split.y_train,
                budget,
            )
            metrics, _, _, _, _ = detailed_model_evaluation(
                model,
                split.x_train[:, selected],
                split.y_train,
                split.x_test[:, selected],
                split.y_test,
            )
            row = base_experiment_row(int(seed), split, selected)
            row.update(metrics)
            row.update(
                {
                    "correlation_threshold": float(threshold),
                    "model_name": spec["name"],
                    "top_k": top_k_label(budget),
                }
            )
            rows.append(row)

    detail = pd.DataFrame(rows)
    summary = summarize_metric_rows(
        detail,
        ["correlation_threshold"],
    )
    best_macro = float(summary["macro_f1_mean"].max())
    tolerance = float(config["selection_policy"].get("macro_f1_tolerance", 0.005))
    eligible = summary[
        summary["macro_f1_mean"] >= best_macro - tolerance
    ].copy()
    eligible = eligible.sort_values(
        ["movement_f1_mean", "selected_subcarrier_count_mean", "macro_f1_std"],
        ascending=[False, True, True],
    )
    selected_threshold = float(eligible.iloc[0]["correlation_threshold"])
    summary["selected"] = (
        summary["correlation_threshold"] == selected_threshold
    )
    summary = summary.sort_values(
        ["selected", "macro_f1_mean"],
        ascending=[False, False],
    ).reset_index(drop=True)

    save_dataframe(paths["tables"] / "correlation_threshold_by_seed.csv", detail)
    save_dataframe(paths["tables"] / "correlation_threshold_summary.csv", summary)
    plot_grouped_metric(
        summary.assign(
            threshold_label=summary["correlation_threshold"].map(
                lambda value: f"{float(value):.3f}"
            )
        ),
        "threshold_label",
        "macro_f1_mean",
        "macro_f1_std",
        "Correlation-threshold comparison",
        "Absolute correlation threshold",
        "Macro F1",
        paths["figures"] / "correlation_threshold_macro_f1.png",
    )
    experiment_index.append(
        {
            "stage": "correlation_threshold_search",
            "objective": "Retune redundant-subcarrier removal for Dataset v2.",
            "main_table": "tables/correlation_threshold_summary.csv",
            "main_figure": "figures/correlation_threshold_macro_f1.png",
        }
    )
    return selected_threshold, detail, summary


def run_named_grid_stage(
    stage_name: str,
    objective: str,
    records: Sequence[FileRecord],
    config: dict[str, Any],
    best_window: dict[str, Any],
    specs: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    paths: dict[str, Path],
    experiment_index: list[dict[str, str]],
    split_cache: dict[tuple[int, int, int], FeatureSplit],
    skip_xgboost: bool,
    include_predictions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    LOGGER.info("Extended stage: %s.", stage_name)
    detail, per_class, confusion, selections, predictions = run_model_specs(
        records,
        config,
        best_window,
        specs,
        seeds,
        split_cache,
        include_predictions=include_predictions,
        skip_xgboost=skip_xgboost,
    )
    LOGGER.info("Summarizing stage: %s", stage_name)
    summary = order_summary_by_policy(summarize_model_results(detail), config)
    LOGGER.info("Saving stage tables: %s", stage_name)
    save_grid_outputs(
        stage_name,
        detail,
        per_class,
        confusion,
        selections,
        summary,
        paths,
    )
    LOGGER.info("Stage tables saved: %s", stage_name)
    if include_predictions:
        save_dataframe(
            paths["tables"] / f"{stage_name}_predictions.csv",
            predictions,
        )
    LOGGER.info("Plotting stage: %s", stage_name)
    plot_model_summary(
        summary,
        objective,
        paths["figures"] / f"{stage_name}_macro_f1.png",
    )
    LOGGER.info("Stage plot completed: %s", stage_name)
    experiment_index.append(
        {
            "stage": stage_name,
            "objective": objective,
            "main_table": f"tables/{stage_name}_summary.csv",
            "main_figure": f"figures/{stage_name}_macro_f1.png",
        }
    )
    return detail, summary, per_class, confusion, selections, predictions


def configured_specs(config: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [copy.deepcopy(item) for item in config["models"][key]]


def capacity_diagnostic_v2(
    records: Sequence[FileRecord],
    config: dict[str, Any],
    best_window: dict[str, Any],
    paths: dict[str, Path],
    experiment_index: list[dict[str, str]],
    split_cache: dict[tuple[int, int, int], FeatureSplit],
    skip_xgboost: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_specs = configured_specs(config, "capacity_diagnostic")
    budgets = list(config["features"]["capacity_top_k_candidates"])
    seeds = list(config["validation"]["capacity_seeds"])
    if config["execution"].get("quick_mode", False):
        base_specs = base_specs[:1]
        budgets = budgets[:1]
        seeds = seeds[:1]
    specs = expand_specs_with_budgets(base_specs, budgets)
    detail, summary, *_ = run_named_grid_stage(
        "classifier_capacity_diagnostic",
        "Classifier-capacity diagnostic",
        records,
        config,
        best_window,
        specs,
        seeds,
        paths,
        experiment_index,
        split_cache,
        skip_xgboost,
    )
    return detail, summary


def compact_ensemble_search_v2(
    records: Sequence[FileRecord],
    config: dict[str, Any],
    best_window: dict[str, Any],
    paths: dict[str, Path],
    experiment_index: list[dict[str, str]],
    split_cache: dict[tuple[int, int, int], FeatureSplit],
    skip_xgboost: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_specs = configured_specs(config, "compact_ensemble")
    budgets = list(config["features"]["compact_top_k_candidates"])
    seeds = list(config["validation"]["compact_search_seeds"])
    if config["execution"].get("quick_mode", False):
        base_specs = base_specs[:2]
        budgets = budgets[:1]
        seeds = seeds[:1]
    specs = expand_specs_with_budgets(base_specs, budgets)
    detail, summary, *_ = run_named_grid_stage(
        "compact_ensemble_search",
        "Compact tree and ensemble comparison",
        records,
        config,
        best_window,
        specs,
        seeds,
        paths,
        experiment_index,
        split_cache,
        skip_xgboost,
    )
    save_dataframe(
        paths["tables"] / "compact_ensemble_recommendations.csv",
        summary.head(10),
    )
    return detail, summary


def stability_validation_v2(
    records: Sequence[FileRecord],
    config: dict[str, Any],
    best_window: dict[str, Any],
    paths: dict[str, Path],
    experiment_index: list[dict[str, str]],
    split_cache: dict[tuple[int, int, int], FeatureSplit],
    skip_xgboost: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = configured_specs(config, "stability_candidates")
    seeds = list(config["validation"]["evaluation_seeds"])
    if config["execution"].get("quick_mode", False):
        specs = specs[:1]
        seeds = seeds[:1]
    detail, summary, *_ = run_named_grid_stage(
        "classifier_stability_validation",
        "Classifier stability across repeated file splits",
        records,
        config,
        best_window,
        specs,
        seeds,
        paths,
        experiment_index,
        split_cache,
        skip_xgboost,
    )
    return detail, summary


def professor_suggestions_v2(
    records: Sequence[FileRecord],
    config: dict[str, Any],
    best_window: dict[str, Any],
    paths: dict[str, Path],
    experiment_index: list[dict[str, str]],
    split_cache: dict[tuple[int, int, int], FeatureSplit],
    skip_xgboost: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = configured_specs(config, "professor_suggestions")
    seeds = list(config["validation"]["evaluation_seeds"])
    if config["execution"].get("quick_mode", False):
        quick_families = (
            "gradient_boosting",
            "logistic_regression",
            "xgboost",
        )
        quick_specs: list[dict[str, Any]] = []

        for family in quick_families:
            representative = next(
                (
                    spec
                    for spec in specs
                    if str(spec.get("family")) == family
                ),
                None,
            )
            if representative is not None:
                quick_specs.append(representative)

        specs = quick_specs
        seeds = seeds[:1]
        LOGGER.info(
            "Quick professor candidates: %s",
            [spec["name"] for spec in specs],
        )
    detail, summary, *_ = run_named_grid_stage(
        "professor_suggestions_comparison",
        "Logistic Regression, Gradient Boosting and XGBoost comparison",
        records,
        config,
        best_window,
        specs,
        seeds,
        paths,
        experiment_index,
        split_cache,
        skip_xgboost,
    )
    return detail, summary


def feature_budget_comparison_v2(
    records: Sequence[FileRecord],
    config: dict[str, Any],
    best_window: dict[str, Any],
    paths: dict[str, Path],
    experiment_index: list[dict[str, str]],
    split_cache: dict[tuple[int, int, int], FeatureSplit],
    skip_xgboost: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_specs = configured_specs(config, "feature_budget_families")
    budgets = list(config["features"]["top_k_candidates"])
    seeds = list(config["validation"]["evaluation_seeds"])
    if config["execution"].get("quick_mode", False):
        quick_families = {"gradient_boosting", "xgboost"}
        base_specs = [
            spec
            for spec in base_specs
            if str(spec.get("family")) in quick_families
        ]

        preferred_budgets: tuple[int | str, ...] = (30, "all")
        budgets = [
            budget
            for budget in preferred_budgets
            if budget in budgets
        ]
        seeds = seeds[:1]
        LOGGER.info(
            "Quick feature-budget candidates: families=%s | budgets=%s",
            [spec["family"] for spec in base_specs],
            budgets,
        )
    specs = expand_specs_with_budgets(base_specs, budgets)
    detail, summary, per_class, confusion, selections, _ = run_named_grid_stage(
        "feature_budget_comparison",
        "Feature-budget comparison for Gradient Boosting and XGBoost",
        records,
        config,
        best_window,
        specs,
        seeds,
        paths,
        experiment_index,
        split_cache,
        skip_xgboost,
    )

    if not selections.empty:
        frequency = (
            selections.groupby(
                ["family", "top_k", "feature_index"],
                dropna=False,
            )
            .agg(
                selected_runs=("seed", "nunique"),
                mean_rank=("rank", "mean"),
                mean_fisher_score=("fisher_score", "mean"),
            )
            .reset_index()
        )
        run_counts = (
            selections.groupby(["family", "top_k"])["seed"]
            .nunique()
            .to_dict()
        )
        frequency["selection_rate"] = frequency.apply(
            lambda row: row["selected_runs"]
            / max(1, run_counts[(row["family"], row["top_k"])]),
            axis=1,
        )
        save_dataframe(
            paths["tables"] / "feature_selection_frequency.csv",
            frequency,
        )

    recommendations = summary.groupby("family", group_keys=False).head(1)
    save_dataframe(
        paths["tables"] / "feature_budget_compact_recommendations.csv",
        recommendations,
    )
    return detail, summary


def candidate_from_summary_row(row: pd.Series) -> dict[str, Any]:
    return {
        "name": str(row["model_name"]),
        "family": str(row["family"]),
        "top_k": normalized_top_k(row["top_k"]),
        "parameters": json.loads(str(row["parameters_json"])),
    }


def select_final_candidate_v2(
    summaries: Sequence[tuple[str, pd.DataFrame]],
    config: dict[str, Any],
    paths: dict[str, Path],
    experiment_index: list[dict[str, str]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for source, summary in summaries:
        if summary.empty:
            continue
        frame = summary.copy()
        frame.insert(0, "source_experiment", source)
        frames.append(frame)
    if not frames:
        raise RuntimeError("No candidate summary is available for final selection.")

    combined = pd.concat(frames, ignore_index=True)
    identifying = ["family", "top_k", "parameters_json"]
    metric_columns = [
        column
        for column in combined.columns
        if column.endswith("_mean") or column.endswith("_std")
    ]
    aggregation: dict[str, str] = {column: "mean" for column in metric_columns}
    aggregation["model_name"] = "first"
    aggregation["source_experiment"] = lambda values: ",".join(
        sorted(set(str(value) for value in values))
    )
    deduplicated = combined.groupby(identifying, dropna=False).agg(aggregation)
    deduplicated = deduplicated.reset_index()
    deduplicated["top_k_sort"] = deduplicated["top_k"].map(
        lambda value: 10**9 if str(value) == "all" else int(value)
    )
    ordered = order_summary_by_policy(deduplicated, config)
    selected = candidate_from_summary_row(ordered.iloc[0])

    save_dataframe(paths["tables"] / "final_candidate_selection.csv", ordered)
    plot_model_summary(
        ordered,
        "Final candidate selection",
        paths["figures"] / "final_candidate_selection.png",
        max_models=20,
    )
    experiment_index.append(
        {
            "stage": "final_candidate_selection",
            "objective": "Select the final model from stable compact candidates.",
            "main_table": "tables/final_candidate_selection.csv",
            "main_figure": "figures/final_candidate_selection.png",
        }
    )
    return selected, ordered


def final_candidate_evaluation_v2(
    records: Sequence[FileRecord],
    config: dict[str, Any],
    best_window: dict[str, Any],
    candidate: dict[str, Any],
    paths: dict[str, Path],
    experiment_index: list[dict[str, str]],
    split_cache: dict[tuple[int, int, int], FeatureSplit],
    skip_xgboost: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specs = [candidate]
    seeds = list(config["validation"]["evaluation_seeds"])
    if config["execution"].get("quick_mode", False):
        seeds = seeds[:1]
    detail, summary, per_class, confusion, _, predictions = run_named_grid_stage(
        "final_candidate_evaluation",
        "Final candidate repeated file-holdout evaluation",
        records,
        config,
        best_window,
        specs,
        seeds,
        paths,
        experiment_index,
        split_cache,
        skip_xgboost,
        include_predictions=True,
    )

    if not confusion.empty:
        aggregate = (
            confusion.groupby(["true_class", "predicted_class"])["count"]
            .sum()
            .unstack(fill_value=0)
            .reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0)
        )
        save_dataframe(
            paths["tables"] / "final_candidate_confusion_matrix.csv",
            aggregate.rename_axis("true_class").reset_index(),
        )
        plot_confusion_matrix(
            aggregate.to_numpy(),
            CLASS_ORDER,
            "Aggregated final-candidate confusion matrix",
            paths["figures"] / "final_candidate_confusion_matrix.png",
        )

    if not per_class.empty:
        per_class_summary = (
            per_class.groupby("class")
            .agg(
                precision_mean=("precision", "mean"),
                precision_std=("precision", "std"),
                recall_mean=("recall", "mean"),
                recall_std=("recall", "std"),
                f1_mean=("f1", "mean"),
                f1_std=("f1", "std"),
                roc_auc_mean=("roc_auc", "mean"),
                average_precision_mean=("average_precision", "mean"),
                support_total=("support", "sum"),
            )
            .reset_index()
            .fillna(0.0)
        )
        save_dataframe(
            paths["tables"] / "final_candidate_class_summary.csv",
            per_class_summary,
        )
        plot_grouped_metric(
            per_class_summary,
            "class",
            "f1_mean",
            "f1_std",
            "Final candidate per-class F1",
            "Class",
            "F1",
            paths["figures"] / "final_candidate_class_f1.png",
        )

    if not detail.empty:
        plot_seed_stability(
            detail,
            paths["figures"] / "final_candidate_stability.png",
        )
    return detail, summary, predictions, per_class


def split_quality_diagnostics_v2(
    predictions: pd.DataFrame,
    paths: dict[str, Path],
    experiment_index: list[dict[str, str]],
) -> dict[str, pd.DataFrame]:
    if predictions.empty:
        return {}

    by_file = (
        predictions.groupby(
            ["seed", "source_file", "file_name", "session", "quadrant", "label"],
            dropna=False,
        )
        .agg(
            window_count=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
    )
    file_summary = (
        by_file.groupby(
            ["source_file", "file_name", "session", "quadrant", "label"],
            dropna=False,
        )
        .agg(
            evaluation_count=("seed", "nunique"),
            mean_window_count=("window_count", "mean"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            mean_confidence=("mean_confidence", "mean"),
        )
        .reset_index()
        .fillna(0.0)
        .sort_values(["accuracy_mean", "evaluation_count"])
    )
    quadrant_summary = (
        predictions.groupby(["seed", "quadrant"])["correct"]
        .mean()
        .reset_index(name="accuracy")
        .groupby("quadrant")
        .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"))
        .reset_index()
        .fillna(0.0)
    )
    session_summary = (
        predictions.groupby(["seed", "session"])["correct"]
        .mean()
        .reset_index(name="accuracy")
        .groupby("session")
        .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"))
        .reset_index()
        .fillna(0.0)
    )
    class_quadrant = (
        predictions.groupby(["seed", "label", "quadrant"])["correct"]
        .mean()
        .reset_index(name="accuracy")
        .groupby(["label", "quadrant"])
        .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"))
        .reset_index()
        .fillna(0.0)
    )
    split_summary = (
        predictions.groupby("seed")
        .agg(
            test_windows=("correct", "size"),
            test_files=("source_file", "nunique"),
            accuracy=("correct", "mean"),
            mean_confidence=("confidence", "mean"),
        )
        .reset_index()
    )

    outputs = {
        "file_performance": file_summary,
        "worst_files": file_summary.head(25),
        "quadrant_performance": quadrant_summary,
        "session_performance": session_summary,
        "class_quadrant_performance": class_quadrant,
        "split_quality": split_summary,
    }
    for name, frame in outputs.items():
        save_dataframe(paths["tables"] / f"{name}.csv", frame)

    plot_grouped_metric(
        quadrant_summary,
        "quadrant",
        "accuracy_mean",
        "accuracy_std",
        "Final candidate accuracy by quadrant",
        "Quadrant",
        "Accuracy",
        paths["figures"] / "quadrant_performance.png",
    )
    plot_grouped_metric(
        session_summary,
        "session",
        "accuracy_mean",
        "accuracy_std",
        "Final candidate accuracy by session",
        "Session",
        "Accuracy",
        paths["figures"] / "session_performance.png",
    )
    experiment_index.append(
        {
            "stage": "dataset_split_quality_diagnostic",
            "objective": "Document split stability and the weakest files, sessions and quadrants.",
            "main_table": "tables/file_performance.csv",
            "main_figure": "figures/quadrant_performance.png",
        }
    )
    return outputs


def final_model_dict(candidate: dict[str, Any]) -> dict[str, Any]:
    family = str(candidate["family"])
    return {
        "name": family,
        "parameters": dict(candidate["parameters"]),
    }


def generate_extended_report(
    records: Sequence[FileRecord],
    sampling_rate_hz: float,
    selected_threshold: float,
    best_window: dict[str, Any],
    tree_summary: pd.DataFrame,
    capacity_summary: pd.DataFrame,
    compact_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
    professor_summary: pd.DataFrame,
    budget_summary: pd.DataFrame,
    candidate_selection: pd.DataFrame,
    final_summary: pd.DataFrame,
    binary_summary: pd.DataFrame,
    hierarchical_summary: pd.DataFrame,
    session_results: pd.DataFrame,
    quadrant_results: pd.DataFrame,
    realtime_config: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    counts = Counter(record.label for record in records)
    report = f"""# Dataset v2 complete experiment report

Generated at: `{now_utc_iso()}`

## Dataset

- Accepted acquisition files: **{len(records)}**
- Sessions: **{', '.join(sorted({record.session for record in records}))}**
- Quadrants: **{', '.join(sorted({record.quadrant for record in records}))}**
- Empty: **{counts.get('empty', 0)} files**
- Static presence: **{counts.get('static_presence', 0)} files**
- Movement: **{counts.get('movement', 0)} files**
- Median acquisition rate: **{sampling_rate_hz:.3f} Hz**

## Methodological safeguards

- The split unit is the complete acquisition file, never an isolated window.
- File holdout is stratified by class, session and quadrant.
- Z-score, correlation filtering and Fisher ranking are fitted only on training files.
- The ten historical seeds from the advanced branch are reused for direct comparison.
- Session and quadrant holdouts are reported separately from ordinary file holdout.

## Selected preprocessing and temporal parameters

- Correlation threshold: **{selected_threshold:.3f}**
- Window: **{float(best_window['window_seconds']):.3f} s** ({int(best_window['window_packets'])} packets)
- Step: **{float(best_window['step_seconds']):.3f} s** ({int(best_window['step_packets'])} packets)

## Decision-tree tuning

{markdown_table(tree_summary, 12)}

## Classifier-capacity diagnostic

{markdown_table(capacity_summary, 15)}

## Compact ensembles

{markdown_table(compact_summary, 15)}

## Stability across repeated splits

{markdown_table(stability_summary, 15)}

## Professor-requested models and metrics

{markdown_table(professor_summary, 15)}

## Feature budgets

{markdown_table(budget_summary, 15)}

## Final candidate selection

{markdown_table(candidate_selection, 15)}

## Final repeated file holdout

{markdown_table(final_summary, 10)}

## Binary diagnostics

{markdown_table(binary_summary, 10)}

## Direct versus hierarchical architecture

{markdown_table(hierarchical_summary, 10)}

## Unseen-session generalization

{markdown_table(session_results, 10)}

## Unseen-quadrant generalization

{markdown_table(quadrant_results, 10)}

## Realtime export

- Serialized model: `{realtime_config['classifier']['serialized_model_file']}`
- Human-readable configuration: `realtime_pipeline_config.json`
- Portable model structure: `{realtime_config['classifier']['portable_structure_file']}`
- Class order: `{', '.join(realtime_config['input']['class_order'])}`

The all-data fit is used only to produce the realtime artifact. TCC performance values must be taken from the validation tables above.
"""
    (paths["reports"] / "dataset_v2_complete_report.md").write_text(
        report,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_extended_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Dataset v2 experiment suite and export the "
            "selected realtime candidate."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Dataset v2 JSON experiment configuration.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run a reduced smoke test. Results are written to separate quick "
            "directories and must not be reported in the TCC."
        ),
    )
    parser.add_argument(
        "--skip-xgboost",
        action="store_true",
        help="Skip optional XGBoost candidates without stopping other tests.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Run validations without fitting and exporting the realtime model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_extended_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    config["execution"]["quick_mode"] = bool(args.quick)

    if args.quick:
        config["outputs"]["results_root"] = config["outputs"].get(
            "quick_results_root",
            "Tools/datasets/results/dataset_v2_retraining_quick",
        )
        config["outputs"]["processed_root"] = config["outputs"].get(
            "quick_processed_root",
            "Tools/datasets/processed/quick_smoke_test",
        )
        # Smoke mode verifies dataset loading, preprocessing, model grids,
        # selection and result generation. The expensive generalization and
        # diagnostic stages remain enabled in the official execution only.
        config["validation"]["run_binary_diagnostics"] = False
        config["validation"]["run_hierarchical_comparison"] = False
        config["validation"]["run_session_holdout"] = False
        config["validation"]["run_quadrant_holdout"] = False

    paths = prepare_output_directories(config)
    configure_logging(paths["logs"] / "dataset_v2_complete_suite.log")
    LOGGER.info("Configuration: %s", config_path)
    LOGGER.info("Quick mode: %s", args.quick)
    LOGGER.info("Skip XGBoost: %s", args.skip_xgboost)
    save_json(paths["reports"] / "training_config_used.json", config)

    all_records, diagnostics = discover_and_load_dataset(config)
    dataset_summary = dataset_summary_table(all_records)
    sampling_rate_hz = determine_sampling_rate(all_records, config)

    save_dataframe(paths["tables"] / "dataset_file_diagnostics.csv", diagnostics)
    save_dataframe(paths["tables"] / "dataset_summary.csv", dataset_summary)
    save_dataframe(
        paths["tables"] / "dataset_counts_by_group.csv",
        dataset_summary.groupby(["session", "quadrant", "label"])
        .size()
        .reset_index(name="file_count"),
    )
    plot_dataset_file_counts(
        dataset_summary,
        paths["figures"] / "dataset_file_distribution.png",
    )
    plot_sampling_rate(
        dataset_summary,
        paths["figures"] / "dataset_sampling_rate_distribution.png",
    )

    records = quick_subset_by_stratum(all_records, config)
    experiment_index: list[dict[str, str]] = [
        {
            "stage": "dataset_diagnostics",
            "objective": (
                "Validate packet counts, rates, labels, sessions and quadrants."
            ),
            "main_table": "tables/dataset_summary.csv",
            "main_figure": "figures/dataset_file_distribution.png",
        }
    ]

    selected_threshold, _, threshold_summary = correlation_threshold_search_v2(
        records,
        config,
        sampling_rate_hz,
        paths,
        experiment_index,
        args.skip_xgboost,
    )
    config["preprocessing"]["correlation_threshold"] = selected_threshold
    save_json(paths["reports"] / "training_config_resolved.json", config)

    split_cache: dict[tuple[int, int, int], FeatureSplit] = {}
    best_window, _, window_summary = window_step_search(
        records,
        config,
        sampling_rate_hz,
        paths,
        experiment_index,
    )
    best_tree_parameters, _, tree_summary = tree_parameter_search(
        records,
        config,
        best_window,
        paths,
        experiment_index,
        split_cache,
    )

    capacity_detail, capacity_summary = capacity_diagnostic_v2(
        records,
        config,
        best_window,
        paths,
        experiment_index,
        split_cache,
        args.skip_xgboost,
    )
    compact_detail, compact_summary = compact_ensemble_search_v2(
        records,
        config,
        best_window,
        paths,
        experiment_index,
        split_cache,
        args.skip_xgboost,
    )
    stability_detail, stability_summary = stability_validation_v2(
        records,
        config,
        best_window,
        paths,
        experiment_index,
        split_cache,
        args.skip_xgboost,
    )
    professor_detail, professor_summary = professor_suggestions_v2(
        records,
        config,
        best_window,
        paths,
        experiment_index,
        split_cache,
        args.skip_xgboost,
    )
    budget_detail, budget_summary = feature_budget_comparison_v2(
        records,
        config,
        best_window,
        paths,
        experiment_index,
        split_cache,
        args.skip_xgboost,
    )

    candidate, candidate_selection = select_final_candidate_v2(
        [
            ("classifier_stability_validation", stability_summary),
            ("professor_suggestions_comparison", professor_summary),
            ("feature_budget_comparison", budget_summary),
        ],
        config,
        paths,
        experiment_index,
    )
    final_detail, final_summary, predictions, final_per_class = (
        final_candidate_evaluation_v2(
            records,
            config,
            best_window,
            candidate,
            paths,
            experiment_index,
            split_cache,
            args.skip_xgboost,
        )
    )
    split_quality_diagnostics_v2(predictions, paths, experiment_index)

    selected_budget = normalized_top_k(candidate["top_k"])
    selected_model = final_model_dict(candidate)
    binary_summary = binary_task_evaluation(
        records,
        config,
        best_window,
        selected_model,
        selected_budget,
        paths,
        experiment_index,
        split_cache,
    )
    hierarchical_summary = hierarchical_comparison(
        records,
        config,
        best_window,
        selected_model,
        selected_budget,
        paths,
        experiment_index,
        split_cache,
    )
    session_results, quadrant_results = run_generalization_holdouts(
        records,
        config,
        best_window,
        selected_model,
        selected_budget,
        paths,
        experiment_index,
    )

    if args.no_export:
        realtime_config = {
            "classifier": {
                "serialized_model_file": "not_exported",
                "portable_structure_file": "not_exported",
            },
            "input": {"class_order": CLASS_ORDER},
        }
    else:
        realtime_config, _ = export_final_artifacts(
            all_records if not args.quick else records,
            config,
            sampling_rate_hz,
            best_window,
            selected_model,
            selected_budget,
            final_summary,
            paths,
        )

    generate_extended_report(
        records,
        sampling_rate_hz,
        selected_threshold,
        best_window,
        tree_summary,
        capacity_summary,
        compact_summary,
        stability_summary,
        professor_summary,
        budget_summary,
        candidate_selection,
        final_summary,
        binary_summary,
        hierarchical_summary,
        session_results,
        quadrant_results,
        realtime_config,
        paths,
    )
    save_experiment_index(
        experiment_index,
        paths["results"] / "experiment_index.csv",
    )

    LOGGER.info("Dataset v2 experiment suite completed.")
    LOGGER.info("Results: %s", paths["results"])
    LOGGER.info("Selected model: %s", candidate["name"])
    LOGGER.info("Selected top_k: %s", selected_budget)
    LOGGER.info("Selected threshold: %.3f", selected_threshold)
    LOGGER.info(
        "Selected window/step: %.3fs / %.3fs",
        float(best_window["window_seconds"]),
        float(best_window["step_seconds"]),
    )


if __name__ == "__main__":
    main()