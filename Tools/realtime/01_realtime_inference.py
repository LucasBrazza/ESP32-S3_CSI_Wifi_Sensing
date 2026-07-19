from __future__ import annotations

"""
Offline simulation of the Dataset v2 realtime inference pipeline.

This script reads one continuous CSI binary file, or a directory containing
CSI binary files, reproduces the preprocessing and feature extraction used
during training, obtains class probabilities from the exported model and
optionally applies ``TemporalStateMachine``.

Run from the repository root:

    python -m Tools.realtime.01_realtime_inference \
        --input-file path/to/scenario.bin \
        --config Tools/datasets/processed/realtime_pipeline_config_extra_trees.json \
        --model Tools/datasets/processed/realtime_model_extra_trees.joblib \
        --state-machine-config Tools/realtime/state_machine_config.json \
        --output-csv Tools/datasets/results/realtime_offline_predictions.csv

For a directory of chunks that belong to the same continuous recording:

    python -m Tools.realtime.01_realtime_inference \
        --input-dir path/to/chunks \
        --concatenate-files \
        --output-csv Tools/datasets/results/realtime_offline_predictions.csv

The live serial implementation will reuse the same ``RealtimeInferenceEngine``
after the offline pipeline has been validated.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import joblib
    import numpy as np
    from scipy.ndimage import median_filter, uniform_filter1d
except ImportError as exc:
    raise SystemExit(
        "Missing realtime dependency. Ensure numpy, scipy and joblib are "
        f"installed in the active environment. Original error: {exc}"
    ) from exc


THIS_FILE = Path(__file__).resolve()
TOOLS_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = TOOLS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Tools.csi.csi_binary_io import read_packets  # noqa: E402
from Tools.realtime.temporal_state_machine import (  # noqa: E402
    TemporalStateMachine,
    load_configuration as load_state_machine_configuration,
)


DEFAULT_CONFIG_CANDIDATES = (
    PROJECT_ROOT
    / "Tools/datasets/processed/realtime_pipeline_config_extra_trees.json",
    PROJECT_ROOT
    / "Tools/datasets/processed/realtime_pipeline_config.json",
    PROJECT_ROOT
    / "Tools/realtime/config/dataset_v2_realtime_pipeline_config.json",
)

DEFAULT_MODEL_CANDIDATES = (
    PROJECT_ROOT / "Tools/datasets/processed/realtime_model_extra_trees.joblib",
    PROJECT_ROOT / "Tools/datasets/processed/realtime_model.joblib",
)

DEFAULT_STATE_MACHINE_CONFIG = (
    PROJECT_ROOT / "Tools/realtime/state_machine_config.json"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "Tools/datasets/results/realtime_offline_predictions.csv"
)


@dataclass(frozen=True)
class PacketSequence:
    source_name: str
    packets: list[dict[str, Any]]


@dataclass(frozen=True)
class PreparedSequence:
    source_name: str
    features: np.ndarray
    metadata: list[dict[str, Any]]


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def first_existing(candidates: Iterable[Path], description: str) -> Path:
    checked = []
    for candidate in candidates:
        resolved = resolve_path(candidate)
        checked.append(str(resolved))
        if resolved.exists():
            return resolved

    checked_text = "\n  - ".join(checked)
    raise FileNotFoundError(
        f"{description} not found. Checked:\n  - {checked_text}"
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in: {path}")
    return data


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_config_path(explicit: Path | None) -> Path:
    if explicit is not None:
        path = resolve_path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Realtime config not found: {path}")
        return path
    return first_existing(
        DEFAULT_CONFIG_CANDIDATES,
        "Realtime pipeline configuration",
    )


def resolve_model_path(
    explicit: Path | None,
    config_path: Path,
    config: Mapping[str, Any],
) -> Path:
    if explicit is not None:
        path = resolve_path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Serialized model not found: {path}")
        return path

    referenced = str(
        config.get("classifier", {}).get("serialized_model_file", "")
    ).strip()

    candidates: list[Path] = []
    if referenced:
        candidates.extend(
            [
                config_path.parent / referenced,
                PROJECT_ROOT / "Tools/datasets/processed" / referenced,
            ]
        )

        referenced_path = Path(referenced)
        candidates.extend(
            [
                config_path.parent
                / f"{referenced_path.stem}_extra_trees{referenced_path.suffix}",
                PROJECT_ROOT
                / "Tools/datasets/processed"
                / f"{referenced_path.stem}_extra_trees{referenced_path.suffix}",
            ]
        )

    candidates.extend(DEFAULT_MODEL_CANDIDATES)
    return first_existing(candidates, "Serialized realtime model")


def validate_realtime_config(config: Mapping[str, Any]) -> None:
    required_sections = (
        "input",
        "window",
        "preprocessing",
        "features",
        "classifier",
    )
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(
            f"Realtime config is missing sections: {', '.join(missing)}"
        )

    class_order = list(config["input"].get("class_order", []))
    if class_order != ["empty", "static_presence", "movement"]:
        raise ValueError(
            "Unexpected class order. Expected "
            "['empty', 'static_presence', 'movement'], got "
            f"{class_order}."
        )

    window_packets = int(config["window"]["size_packets"])
    step_packets = int(config["window"]["step_packets"])
    if window_packets < 2 or step_packets < 1:
        raise ValueError("Invalid window or step packet count.")


def discover_input_files(
    input_file: Path | None,
    input_dir: Path | None,
    recursive: bool,
) -> list[Path]:
    if input_file is not None:
        path = resolve_path(input_file)
        if not path.exists():
            raise FileNotFoundError(f"CSI input file not found: {path}")
        if path.suffix.lower() != ".bin":
            raise ValueError(f"Expected a .bin file: {path}")
        return [path]

    assert input_dir is not None
    directory = resolve_path(input_dir)
    if not directory.exists():
        raise FileNotFoundError(f"CSI input directory not found: {directory}")

    iterator = directory.rglob("*.bin") if recursive else directory.glob("*.bin")
    files = sorted(iterator)
    if not files:
        raise FileNotFoundError(f"No .bin files found under: {directory}")
    return files


def read_sequences(
    files: Sequence[Path],
    concatenate_files: bool,
) -> list[PacketSequence]:
    if concatenate_files:
        packets: list[dict[str, Any]] = []
        for path in files:
            for packet in read_packets(path):
                copied = dict(packet)
                copied["_source_file"] = str(path)
                packets.append(copied)
        return [
            PacketSequence(
                source_name="concatenated:" + ",".join(path.name for path in files),
                packets=packets,
            )
        ]

    sequences: list[PacketSequence] = []
    for path in files:
        packets = []
        for packet in read_packets(path):
            copied = dict(packet)
            copied["_source_file"] = str(path)
            packets.append(copied)
        sequences.append(PacketSequence(str(path), packets))
    return sequences


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
    threshold = float(n_sigmas) * scaled_mad
    replace_mask = (scaled_mad > 0) & (absolute_deviation > threshold)

    filtered = matrix.astype(np.float32, copy=True)
    filtered[replace_mask] = local_median[replace_mask]
    return filtered


def smooth_matrix(matrix: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return matrix.astype(np.float32, copy=True)

    return uniform_filter1d(
        matrix,
        size=int(window_size),
        axis=0,
        mode="nearest",
    ).astype(np.float32, copy=False)


def packets_to_amplitude_matrix(
    packets: Sequence[Mapping[str, Any]],
    expected_subcarriers: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows: list[np.ndarray] = []
    accepted: list[dict[str, Any]] = []

    for packet in packets:
        imag = np.asarray(packet.get("imag", []), dtype=np.float32)
        real = np.asarray(packet.get("real", []), dtype=np.float32)
        length = min(len(imag), len(real))

        if length != expected_subcarriers:
            continue

        amplitude = np.sqrt(
            imag[:expected_subcarriers] ** 2
            + real[:expected_subcarriers] ** 2
        )
        if not np.all(np.isfinite(amplitude)):
            continue

        rows.append(amplitude)
        accepted.append(dict(packet))

    if not rows:
        raise ValueError(
            "No valid packets matched the expected subcarrier count "
            f"({expected_subcarriers})."
        )

    return np.vstack(rows).astype(np.float32, copy=False), accepted


def extract_window_features(window: np.ndarray) -> np.ndarray:
    if window.ndim != 2 or window.shape[0] < 2:
        raise ValueError("A feature window must be 2-D with at least two rows.")

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


def label_summary(
    packets: Sequence[Mapping[str, Any]],
) -> tuple[str, float, str]:
    labels = [
        str(packet.get("label", "")).strip()
        for packet in packets
        if str(packet.get("label", "")).strip()
    ]
    if not labels:
        return "", 0.0, ""

    counts = Counter(labels)
    dominant_label, dominant_count = counts.most_common(1)[0]
    purity = dominant_count / len(labels)
    labels_present = "|".join(sorted(counts))
    return dominant_label, float(purity), labels_present


def packet_timestamp(packet: Mapping[str, Any]) -> float:
    return float(
        packet.get(
            "capture_timestamp",
            packet.get("pc_timestamp", 0.0),
        )
        or 0.0
    )


def prepare_sequence(
    sequence: PacketSequence,
    config: Mapping[str, Any],
) -> PreparedSequence:
    expected_subcarriers = int(config["input"]["expected_subcarriers"])
    matrix, accepted_packets = packets_to_amplitude_matrix(
        sequence.packets,
        expected_subcarriers,
    )

    preprocessing = config["preprocessing"]
    matrix = hampel_filter_matrix(
        matrix,
        radius=int(preprocessing["hampel_radius"]),
        n_sigmas=float(preprocessing["hampel_n_sigmas"]),
    )
    matrix = smooth_matrix(
        matrix,
        window_size=int(preprocessing["moving_average_window"]),
    )

    means = np.asarray(preprocessing["zscore_means"], dtype=np.float64)
    stds = np.asarray(preprocessing["zscore_stds"], dtype=np.float64)
    selected_subcarriers = np.asarray(
        preprocessing["selected_subcarriers"],
        dtype=int,
    )

    if means.shape[0] != expected_subcarriers:
        raise ValueError(
            "Z-score mean length does not match expected subcarriers: "
            f"{means.shape[0]} != {expected_subcarriers}"
        )
    if stds.shape[0] != expected_subcarriers:
        raise ValueError(
            "Z-score std length does not match expected subcarriers: "
            f"{stds.shape[0]} != {expected_subcarriers}"
        )
    if np.any(stds[selected_subcarriers] <= 0):
        raise ValueError("Selected subcarriers contain non-positive std values.")

    reduced = (
        matrix[:, selected_subcarriers] - means[selected_subcarriers]
    ) / stds[selected_subcarriers]
    reduced = reduced.astype(np.float32, copy=False)

    window_packets = int(config["window"]["size_packets"])
    step_packets = int(config["window"]["step_packets"])
    selected_feature_indices = np.asarray(
        config["features"]["selected_indices"],
        dtype=int,
    )

    feature_rows: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []

    if reduced.shape[0] < window_packets:
        raise ValueError(
            f"{sequence.source_name}: only {reduced.shape[0]} valid packets; "
            f"the model requires {window_packets}."
        )

    first_timestamp = packet_timestamp(accepted_packets[0])

    for window_index, start in enumerate(
        range(0, reduced.shape[0] - window_packets + 1, step_packets)
    ):
        end = start + window_packets
        full_features = extract_window_features(reduced[start:end])

        if selected_feature_indices.size == 0:
            raise ValueError("No selected feature indices in realtime config.")
        if int(np.max(selected_feature_indices)) >= full_features.shape[0]:
            raise ValueError(
                "Selected feature index exceeds the extracted vector: "
                f"max={int(np.max(selected_feature_indices))}, "
                f"full={full_features.shape[0]}."
            )

        selected_features = full_features[selected_feature_indices]
        feature_rows.append(selected_features)

        window_packets_metadata = accepted_packets[start:end]
        dominant_label, purity, labels_present = label_summary(
            window_packets_metadata
        )
        start_timestamp = packet_timestamp(window_packets_metadata[0])
        end_timestamp = packet_timestamp(window_packets_metadata[-1])

        metadata_rows.append(
            {
                "source_sequence": sequence.source_name,
                "source_file_start": window_packets_metadata[0].get(
                    "_source_file", sequence.source_name
                ),
                "source_file_end": window_packets_metadata[-1].get(
                    "_source_file", sequence.source_name
                ),
                "window_index": window_index,
                "start_packet": start,
                "end_packet_exclusive": end,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
                "offset_seconds": (
                    end_timestamp - first_timestamp
                    if end_timestamp > 0 and first_timestamp > 0
                    else float(window_index * config["window"]["step_seconds"])
                ),
                "expected_label": dominant_label,
                "expected_label_purity": purity,
                "labels_in_window": labels_present,
                "mean_rssi": float(
                    np.mean(
                        [
                            float(packet.get("rssi", 0) or 0)
                            for packet in window_packets_metadata
                        ]
                    )
                ),
            }
        )

    return PreparedSequence(
        source_name=sequence.source_name,
        features=np.vstack(feature_rows).astype(np.float32, copy=False),
        metadata=metadata_rows,
    )


def normalized_probabilities(
    model: Any,
    features: np.ndarray,
    class_order: Sequence[str],
) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"Model {model.__class__.__name__} does not provide predict_proba."
        )

    raw = np.asarray(model.predict_proba(features), dtype=float)
    if raw.ndim != 2:
        raise ValueError(f"Unexpected predict_proba shape: {raw.shape}")

    model_classes = np.asarray(
        getattr(model, "classes_", np.arange(raw.shape[1]))
    )
    probabilities = np.zeros((raw.shape[0], len(class_order)), dtype=float)

    for column_index, class_value in enumerate(model_classes):
        class_index = int(class_value)
        if class_index < 0 or class_index >= len(class_order):
            raise ValueError(f"Unexpected model class index: {class_index}")
        probabilities[:, class_index] = raw[:, column_index]

    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Model returned a non-positive probability sum.")
    return probabilities / row_sums


class RealtimeInferenceEngine:
    def __init__(
        self,
        config: Mapping[str, Any],
        model: Any,
        state_machine: TemporalStateMachine | None,
    ) -> None:
        self.config = config
        self.model = model
        self.state_machine = state_machine
        self.class_order = list(config["input"]["class_order"])

        selected_count = len(config["features"]["selected_indices"])
        model_feature_count = getattr(model, "n_features_in_", None)
        if (
            model_feature_count is not None
            and int(model_feature_count) != selected_count
        ):
            raise ValueError(
                "Model/config feature mismatch: "
                f"model expects {int(model_feature_count)}, "
                f"config selects {selected_count}."
            )

    def process(
        self,
        sequences: Sequence[PacketSequence],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        for sequence_index, sequence in enumerate(sequences):
            if self.state_machine is not None and sequence_index > 0:
                self.state_machine.reset()

            prepared = prepare_sequence(sequence, self.config)
            probabilities = normalized_probabilities(
                self.model,
                prepared.features,
                self.class_order,
            )
            predictions = np.argmax(probabilities, axis=1)

            for index, metadata in enumerate(prepared.metadata):
                row = dict(metadata)
                row["raw_class_index"] = int(predictions[index])
                row["raw_state"] = self.class_order[int(predictions[index])]

                for class_index, class_name in enumerate(self.class_order):
                    row[f"probability_{class_name}"] = float(
                        probabilities[index, class_index]
                    )

                row["raw_confidence"] = float(
                    np.max(probabilities[index])
                )

                if self.state_machine is not None:
                    decision = self.state_machine.update(
                        {
                            class_name: float(
                                probabilities[index, class_index]
                            )
                            for class_index, class_name
                            in enumerate(self.class_order)
                        }
                    )
                    state_data = asdict(decision)

                    # Avoid replacing raw classifier fields and probabilities.
                    row["state_machine_raw_state"] = state_data.pop(
                        "raw_state"
                    )
                    for key, value in state_data.items():
                        if key.startswith("probability_"):
                            row[f"state_machine_{key}"] = value
                        else:
                            row[key] = value
                else:
                    row["stable_state"] = row["raw_state"]
                    row["transition_reason"] = "state_machine_disabled"

                rows.append(row)

        return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("No prediction rows were generated.")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    rows: Sequence[Mapping[str, Any]],
    config_path: Path,
    model_path: Path,
    output_path: Path,
) -> None:
    raw_counts = Counter(str(row["raw_state"]) for row in rows)
    stable_counts = Counter(str(row["stable_state"]) for row in rows)
    accepted = sum(bool(row.get("transition_accepted", False)) for row in rows)
    inferred = sum(bool(row.get("inferred_transition", False)) for row in rows)

    expected_rows = [
        row
        for row in rows
        if str(row.get("expected_label", ""))
        in {"empty", "static_presence", "movement"}
        and float(row.get("expected_label_purity", 0.0)) >= 0.99
    ]

    raw_accuracy: float | None = None
    stable_accuracy: float | None = None
    if expected_rows:
        raw_accuracy = sum(
            row["raw_state"] == row["expected_label"]
            for row in expected_rows
        ) / len(expected_rows)
        stable_accuracy = sum(
            row["stable_state"] == row["expected_label"]
            for row in expected_rows
        ) / len(expected_rows)

    print()
    print("Dataset v2 Offline Realtime Simulation")
    print("=" * 72)
    print(f"Config:                 {config_path}")
    print(f"Model:                  {model_path}")
    print(f"Generated windows:      {len(rows)}")
    print(f"Raw-state counts:       {dict(raw_counts)}")
    print(f"Stable-state counts:    {dict(stable_counts)}")
    print(f"Accepted transitions:   {accepted}")
    print(f"Inferred transitions:   {inferred}")

    if raw_accuracy is not None and stable_accuracy is not None:
        print(f"Raw accuracy (pure):    {raw_accuracy:.4f}")
        print(f"Stable accuracy (pure): {stable_accuracy:.4f}")

    print(f"Output CSV:             {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the exported Dataset v2 model and temporal state machine "
            "to recorded CSI binary data."
        )
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-file",
        type=Path,
        help="One continuous CSI .bin recording.",
    )
    source.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing CSI .bin recordings or chunks.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directories recursively.",
    )
    parser.add_argument(
        "--concatenate-files",
        action="store_true",
        help=(
            "Treat all discovered files as consecutive chunks of one "
            "continuous recording. Without this option, the state machine "
            "is reset for each file."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Exported realtime_pipeline_config JSON.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Serialized sklearn/joblib realtime model.",
    )
    parser.add_argument(
        "--state-machine-config",
        type=Path,
        default=DEFAULT_STATE_MACHINE_CONFIG,
        help="Temporal state-machine JSON.",
    )
    parser.add_argument(
        "--without-state-machine",
        action="store_true",
        help="Generate only raw classifier predictions.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Detailed prediction output.",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Do not compare the model SHA-256 with the exported config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = resolve_config_path(args.config)
    config = load_json(config_path)
    validate_realtime_config(config)

    model_path = resolve_model_path(args.model, config_path, config)
    model = joblib.load(model_path)

    expected_hash = str(
        config["classifier"].get("serialized_model_sha256", "")
    ).strip()
    if expected_hash and not args.skip_hash_check:
        actual_hash = file_sha256(model_path)
        if actual_hash.lower() != expected_hash.lower():
            raise ValueError(
                "Serialized model SHA-256 differs from the realtime config. "
                "Use matching artifacts or pass --skip-hash-check only after "
                "confirming that the model was merely renamed."
            )

    files = discover_input_files(
        args.input_file,
        args.input_dir,
        recursive=bool(args.recursive),
    )
    sequences = read_sequences(
        files,
        concatenate_files=bool(args.concatenate_files),
    )

    state_machine = None
    if not args.without_state_machine:
        state_machine_path = resolve_path(args.state_machine_config)
        state_config = load_state_machine_configuration(state_machine_path)
        state_machine = TemporalStateMachine(state_config)

    engine = RealtimeInferenceEngine(config, model, state_machine)
    rows = engine.process(sequences)

    output_path = resolve_path(args.output_csv)
    write_csv(output_path, rows)
    print_summary(rows, config_path, model_path, output_path)


if __name__ == "__main__":
    main()
