from __future__ import annotations

"""
Incremental realtime inference engine for Dataset v2.

The engine receives decoded CSI sample events from ``CSIFrameParser`` or
packet dictionaries with ``imag`` and ``real`` arrays. It maintains the
sliding buffer, reproduces the exported preprocessing pipeline, obtains model
probabilities and updates the temporal state machine.

The GUI should call ``push_sample_event`` for each serial sample event. The
method returns ``None`` until a complete inference window is available and
then returns one ``RealtimeInferenceResult`` every configured step.

The field ``stable_state_changed`` is the trigger intended for TTS. The GUI
must announce only the stable state after an accepted state-machine
transition, not every raw model prediction.
"""

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import joblib
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing realtime dependency. Ensure numpy and joblib are installed "
        f"in the active environment. Original error: {exc}"
    ) from exc


THIS_FILE = Path(__file__).resolve()
TOOLS_DIR = THIS_FILE.parents[1]
PROJECT_ROOT = TOOLS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Import through importlib because the existing offline module starts with a
# numeric prefix and cannot be referenced in a regular ``from ... import``.
_OFFLINE_MODULE = importlib.import_module(
    "Tools.realtime.01_realtime_inference"
)

extract_window_features = _OFFLINE_MODULE.extract_window_features
hampel_filter_matrix = _OFFLINE_MODULE.hampel_filter_matrix
normalized_probabilities = _OFFLINE_MODULE.normalized_probabilities
smooth_matrix = _OFFLINE_MODULE.smooth_matrix
validate_realtime_config = _OFFLINE_MODULE.validate_realtime_config

from Tools.realtime.temporal_state_machine import (  # noqa: E402
    TemporalStateMachine,
    load_configuration as load_state_machine_configuration,
)


DEFAULT_PIPELINE_CONFIG_CANDIDATES = (
    PROJECT_ROOT
    / "Tools/datasets/processed/realtime_pipeline_config_extra_trees.json",
    PROJECT_ROOT
    / "Tools/datasets/processed/realtime_pipeline_config.json",
    PROJECT_ROOT
    / "Tools/realtime/config/dataset_v2_realtime_pipeline_config.json",
)

DEFAULT_MODEL_CANDIDATES = (
    PROJECT_ROOT
    / "Tools/datasets/processed/realtime_model_extra_trees.joblib",
    PROJECT_ROOT
    / "Tools/datasets/processed/realtime_model.joblib",
)

DEFAULT_STATE_MACHINE_CONFIG_CANDIDATES = (
    PROJECT_ROOT
    / "Tools/realtime/state_machine_config_candidate_v4.json",
    PROJECT_ROOT
    / "Tools/datasets/results/state_machine_tuning_v4"
    / "state_machine_config_best.json",
    PROJECT_ROOT
    / "Tools/realtime/state_machine_config.json",
)


@dataclass(frozen=True)
class RealtimeInferenceResult:
    inference_index: int
    packet_count: int
    window_start_sequence: int
    window_end_sequence: int
    window_start_timestamp: float
    window_end_timestamp: float
    latest_packet_timestamp: float
    pipeline_delay_seconds: float
    inference_time_ms: float
    packet_rate_hz: float
    raw_state: str
    raw_confidence: float
    stable_state_before: str
    stable_state: str
    stable_state_changed: bool
    should_announce: bool
    movement_origin: str | None
    transition_accepted: bool
    transition_reason: str
    inferred_transition: bool
    probability_empty: float
    probability_static_presence: float
    probability_movement: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RealtimeEngineDiagnostics:
    accepted_packets: int
    rejected_packets: int
    inference_count: int
    packets_since_inference: int
    buffer_packets: int
    required_buffer_packets: int
    window_packets: int
    step_packets: int
    filter_context_packets: int
    packet_rate_hz: float
    last_error: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def first_existing(
    candidates: Iterable[Path],
    description: str,
) -> Path:
    checked: list[str] = []

    for candidate in candidates:
        resolved = resolve_path(candidate)
        checked.append(str(resolved))
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        f"{description} not found. Checked:\n  - "
        + "\n  - ".join(checked)
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in: {path}")

    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_pipeline_config_path(
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        resolved = resolve_path(explicit)
        if not resolved.exists():
            raise FileNotFoundError(
                f"Realtime pipeline config not found: {resolved}"
            )
        return resolved

    return first_existing(
        DEFAULT_PIPELINE_CONFIG_CANDIDATES,
        "Realtime pipeline configuration",
    )


def resolve_model_path(
    explicit: Path | None,
    pipeline_config_path: Path,
    pipeline_config: Mapping[str, Any],
) -> Path:
    if explicit is not None:
        resolved = resolve_path(explicit)
        if not resolved.exists():
            raise FileNotFoundError(
                f"Serialized realtime model not found: {resolved}"
            )
        return resolved

    candidates: list[Path] = []
    referenced = str(
        pipeline_config.get("classifier", {}).get(
            "serialized_model_file",
            "",
        )
    ).strip()

    if referenced:
        referenced_path = Path(referenced)
        candidates.extend(
            [
                pipeline_config_path.parent / referenced_path,
                PROJECT_ROOT
                / "Tools/datasets/processed"
                / referenced_path.name,
                pipeline_config_path.parent
                / (
                    f"{referenced_path.stem}_extra_trees"
                    f"{referenced_path.suffix}"
                ),
                PROJECT_ROOT
                / "Tools/datasets/processed"
                / (
                    f"{referenced_path.stem}_extra_trees"
                    f"{referenced_path.suffix}"
                ),
            ]
        )

    candidates.extend(DEFAULT_MODEL_CANDIDATES)

    return first_existing(
        candidates,
        "Serialized realtime model",
    )


def resolve_state_machine_config_path(
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        resolved = resolve_path(explicit)
        if not resolved.exists():
            raise FileNotFoundError(
                f"State-machine config not found: {resolved}"
            )
        return resolved

    return first_existing(
        DEFAULT_STATE_MACHINE_CONFIG_CANDIDATES,
        "State-machine configuration",
    )


def sample_event_to_packet(
    event: Mapping[str, Any],
) -> dict[str, Any]:
    """Converts a decoded parser event into the packet representation."""

    if str(event.get("type", "")) != "sample":
        raise ValueError(
            "Expected a CSI sample event with event['type'] == 'sample'."
        )

    metadata = event.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("CSI sample event metadata must be a mapping.")

    pc_timestamp = float(event.get("pc_timestamp", time.time()) or time.time())
    capture_timestamp = float(
        event.get("capture_timestamp", pc_timestamp) or pc_timestamp
    )

    return {
        "pc_timestamp": pc_timestamp,
        "capture_timestamp": capture_timestamp,
        "esp_timestamp_us": int(
            metadata.get("timestamp_us", 0) or 0
        ),
        "sequence": int(metadata.get("sequence", 0) or 0),
        "packet_index": int(metadata.get("sequence", 0) or 0),
        "rssi": int(metadata.get("rssi", 0) or 0),
        "rate": int(metadata.get("rate", 0) or 0),
        "channel": int(metadata.get("channel", 0) or 0),
        "csi_len": int(metadata.get("csi_len", 0) or 0),
        "flags": int(metadata.get("flags", 0) or 0),
        "imag": event.get("imag", []),
        "real": event.get("real", []),
    }


def packet_timestamp(packet: Mapping[str, Any]) -> float:
    return float(
        packet.get(
            "capture_timestamp",
            packet.get("pc_timestamp", 0.0),
        )
        or 0.0
    )


class IncrementalRealtimeInferenceEngine:
    """Sliding-window CSI inference engine for live serial data."""

    def __init__(
        self,
        pipeline_config: Mapping[str, Any],
        model: Any,
        state_machine: TemporalStateMachine,
        *,
        pipeline_config_path: Path | None = None,
        model_path: Path | None = None,
        state_machine_config_path: Path | None = None,
    ) -> None:
        validate_realtime_config(pipeline_config)

        self.pipeline_config = dict(pipeline_config)
        self.model = model
        self.state_machine = state_machine

        self.pipeline_config_path = pipeline_config_path
        self.model_path = model_path
        self.state_machine_config_path = state_machine_config_path

        input_config = self.pipeline_config["input"]
        window_config = self.pipeline_config["window"]
        preprocessing = self.pipeline_config["preprocessing"]
        feature_config = self.pipeline_config["features"]

        self.class_order = list(input_config["class_order"])
        self.expected_subcarriers = int(
            input_config["expected_subcarriers"]
        )
        self.window_packets = int(window_config["size_packets"])
        self.step_packets = int(window_config["step_packets"])

        self.hampel_radius = int(
            preprocessing["hampel_radius"]
        )
        self.hampel_n_sigmas = float(
            preprocessing["hampel_n_sigmas"]
        )
        self.moving_average_window = int(
            preprocessing["moving_average_window"]
        )

        self.zscore_means = np.asarray(
            preprocessing["zscore_means"],
            dtype=np.float64,
        )
        self.zscore_stds = np.asarray(
            preprocessing["zscore_stds"],
            dtype=np.float64,
        )
        self.selected_subcarriers = np.asarray(
            preprocessing["selected_subcarriers"],
            dtype=int,
        )
        self.selected_feature_indices = np.asarray(
            feature_config["selected_indices"],
            dtype=int,
        )

        self.filter_context_packets = self._calculate_filter_context()
        self.required_buffer_packets = (
            self.window_packets + 2 * self.filter_context_packets
        )

        self._validate_artifact_shapes()

        self.amplitude_buffer: deque[np.ndarray] = deque(
            maxlen=self.required_buffer_packets
        )
        self.packet_buffer: deque[dict[str, Any]] = deque(
            maxlen=self.required_buffer_packets
        )
        self.rate_timestamps: deque[float] = deque()

        self.accepted_packets = 0
        self.rejected_packets = 0
        self.inference_count = 0
        self.packets_since_inference = 0
        self.last_error = ""

    @classmethod
    def from_artifacts(
        cls,
        *,
        pipeline_config_path: Path | None = None,
        model_path: Path | None = None,
        state_machine_config_path: Path | None = None,
        verify_model_hash: bool = True,
    ) -> "IncrementalRealtimeInferenceEngine":
        resolved_pipeline = resolve_pipeline_config_path(
            pipeline_config_path
        )
        pipeline_config = load_json(resolved_pipeline)

        resolved_model = resolve_model_path(
            model_path,
            resolved_pipeline,
            pipeline_config,
        )

        if verify_model_hash:
            expected_hash = str(
                pipeline_config["classifier"].get(
                    "serialized_model_sha256",
                    "",
                )
            ).strip()

            if expected_hash:
                actual_hash = sha256_file(resolved_model)
                if actual_hash.lower() != expected_hash.lower():
                    raise ValueError(
                        "Serialized model SHA-256 differs from the realtime "
                        "pipeline configuration. Use matching artifacts."
                    )

        resolved_state_config = resolve_state_machine_config_path(
            state_machine_config_path
        )
        state_config = load_state_machine_configuration(
            resolved_state_config
        )

        return cls(
            pipeline_config=pipeline_config,
            model=joblib.load(resolved_model),
            state_machine=TemporalStateMachine(state_config),
            pipeline_config_path=resolved_pipeline,
            model_path=resolved_model,
            state_machine_config_path=resolved_state_config,
        )

    def _calculate_filter_context(self) -> int:
        moving_average_context = max(
            0,
            int(math.ceil((self.moving_average_window - 1) / 2)),
        )
        return max(
            0,
            self.hampel_radius + moving_average_context,
        )

    def _validate_artifact_shapes(self) -> None:
        if self.zscore_means.shape[0] != self.expected_subcarriers:
            raise ValueError(
                "Z-score mean length does not match expected subcarriers: "
                f"{self.zscore_means.shape[0]} != "
                f"{self.expected_subcarriers}."
            )

        if self.zscore_stds.shape[0] != self.expected_subcarriers:
            raise ValueError(
                "Z-score std length does not match expected subcarriers: "
                f"{self.zscore_stds.shape[0]} != "
                f"{self.expected_subcarriers}."
            )

        if self.selected_subcarriers.size == 0:
            raise ValueError("No selected subcarriers in realtime config.")

        if (
            int(np.min(self.selected_subcarriers)) < 0
            or int(np.max(self.selected_subcarriers))
            >= self.expected_subcarriers
        ):
            raise ValueError(
                "Selected subcarrier index is outside the expected CSI "
                "vector."
            )

        if np.any(self.zscore_stds[self.selected_subcarriers] <= 0):
            raise ValueError(
                "Selected subcarriers contain non-positive Z-score std."
            )

        if self.selected_feature_indices.size == 0:
            raise ValueError(
                "No selected feature indices in realtime config."
            )

        expected_model_features = getattr(
            self.model,
            "n_features_in_",
            None,
        )
        if (
            expected_model_features is not None
            and int(expected_model_features)
            != int(self.selected_feature_indices.size)
        ):
            raise ValueError(
                "Model/config feature mismatch: model expects "
                f"{int(expected_model_features)}, config selects "
                f"{int(self.selected_feature_indices.size)}."
            )

    def reset(self) -> None:
        self.amplitude_buffer.clear()
        self.packet_buffer.clear()
        self.rate_timestamps.clear()

        self.accepted_packets = 0
        self.rejected_packets = 0
        self.inference_count = 0
        self.packets_since_inference = 0
        self.last_error = ""

        self.state_machine.reset()

    def push_sample_event(
        self,
        event: Mapping[str, Any],
    ) -> RealtimeInferenceResult | None:
        return self.push_packet(sample_event_to_packet(event))

    def push_packet(
        self,
        packet: Mapping[str, Any],
    ) -> RealtimeInferenceResult | None:
        try:
            amplitude = self._packet_amplitude(packet)
        except (TypeError, ValueError) as exc:
            self.rejected_packets += 1
            self.last_error = str(exc)
            return None

        packet_copy = dict(packet)
        timestamp = packet_timestamp(packet_copy)

        self.amplitude_buffer.append(amplitude)
        self.packet_buffer.append(packet_copy)
        self.accepted_packets += 1
        self.packets_since_inference += 1
        self._update_packet_rate(timestamp)

        if len(self.amplitude_buffer) < self.required_buffer_packets:
            return None

        if (
            self.inference_count > 0
            and self.packets_since_inference < self.step_packets
        ):
            return None

        result = self._run_inference()
        self.inference_count += 1
        self.packets_since_inference = 0
        return result

    def _packet_amplitude(
        self,
        packet: Mapping[str, Any],
    ) -> np.ndarray:
        imag = np.asarray(
            packet.get("imag", []),
            dtype=np.float32,
        )
        real = np.asarray(
            packet.get("real", []),
            dtype=np.float32,
        )

        length = min(imag.size, real.size)

        if length != self.expected_subcarriers:
            raise ValueError(
                "Unexpected CSI subcarrier count: "
                f"{length}; expected {self.expected_subcarriers}."
            )

        amplitude = np.sqrt(
            imag[: self.expected_subcarriers] ** 2
            + real[: self.expected_subcarriers] ** 2
        )

        if not np.all(np.isfinite(amplitude)):
            raise ValueError(
                "CSI amplitude contains non-finite values."
            )

        return amplitude.astype(np.float32, copy=False)

    def _update_packet_rate(self, timestamp: float) -> None:
        if timestamp <= 0:
            timestamp = time.time()

        self.rate_timestamps.append(timestamp)
        cutoff = timestamp - 1.0

        while (
            self.rate_timestamps
            and self.rate_timestamps[0] < cutoff
        ):
            self.rate_timestamps.popleft()

    def packet_rate_hz(self) -> float:
        if len(self.rate_timestamps) < 2:
            return float(len(self.rate_timestamps))

        duration = (
            self.rate_timestamps[-1] - self.rate_timestamps[0]
        )
        if duration <= 0:
            return float(len(self.rate_timestamps))

        return float(
            (len(self.rate_timestamps) - 1) / duration
        )

    def _run_inference(self) -> RealtimeInferenceResult:
        started = time.perf_counter()

        amplitude_matrix = np.vstack(
            list(self.amplitude_buffer)
        ).astype(np.float32, copy=False)
        packets = list(self.packet_buffer)

        filtered = hampel_filter_matrix(
            amplitude_matrix,
            radius=self.hampel_radius,
            n_sigmas=self.hampel_n_sigmas,
        )
        filtered = smooth_matrix(
            filtered,
            window_size=self.moving_average_window,
        )

        start = self.filter_context_packets
        end = start + self.window_packets

        filtered_window = filtered[start:end]
        window_packets = packets[start:end]

        if filtered_window.shape[0] != self.window_packets:
            raise RuntimeError(
                "Internal realtime window has an unexpected size."
            )

        reduced = (
            filtered_window[:, self.selected_subcarriers]
            - self.zscore_means[self.selected_subcarriers]
        ) / self.zscore_stds[self.selected_subcarriers]
        reduced = reduced.astype(np.float32, copy=False)

        full_features = extract_window_features(reduced)

        if int(np.max(self.selected_feature_indices)) >= full_features.size:
            raise ValueError(
                "Selected feature index exceeds extracted realtime vector: "
                f"max={int(np.max(self.selected_feature_indices))}, "
                f"features={full_features.size}."
            )

        selected_features = full_features[
            self.selected_feature_indices
        ].reshape(1, -1)

        probability_matrix = normalized_probabilities(
            self.model,
            selected_features,
            self.class_order,
        )
        probabilities = probability_matrix[0]
        raw_index = int(np.argmax(probabilities))
        raw_state = self.class_order[raw_index]

        decision = self.state_machine.update(
            {
                class_name: float(probabilities[class_index])
                for class_index, class_name
                in enumerate(self.class_order)
            }
        )

        inference_time_ms = (
            time.perf_counter() - started
        ) * 1000.0

        first_packet = window_packets[0]
        last_packet = window_packets[-1]
        latest_packet = packets[-1]

        window_start_timestamp = packet_timestamp(first_packet)
        window_end_timestamp = packet_timestamp(last_packet)
        latest_packet_timestamp = packet_timestamp(latest_packet)

        pipeline_delay = max(
            0.0,
            latest_packet_timestamp - window_end_timestamp,
        )

        stable_state_changed = bool(
            decision.transition_accepted
            and decision.stable_state
            != decision.stable_state_before
        )

        probability_by_name = {
            class_name: float(probabilities[class_index])
            for class_index, class_name
            in enumerate(self.class_order)
        }

        return RealtimeInferenceResult(
            inference_index=self.inference_count,
            packet_count=self.accepted_packets,
            window_start_sequence=int(
                first_packet.get("sequence", 0) or 0
            ),
            window_end_sequence=int(
                last_packet.get("sequence", 0) or 0
            ),
            window_start_timestamp=window_start_timestamp,
            window_end_timestamp=window_end_timestamp,
            latest_packet_timestamp=latest_packet_timestamp,
            pipeline_delay_seconds=pipeline_delay,
            inference_time_ms=float(inference_time_ms),
            packet_rate_hz=self.packet_rate_hz(),
            raw_state=raw_state,
            raw_confidence=float(np.max(probabilities)),
            stable_state_before=decision.stable_state_before,
            stable_state=decision.stable_state,
            stable_state_changed=stable_state_changed,
            should_announce=stable_state_changed,
            movement_origin=decision.movement_origin,
            transition_accepted=bool(
                decision.transition_accepted
            ),
            transition_reason=decision.transition_reason,
            inferred_transition=bool(
                decision.inferred_transition
            ),
            probability_empty=probability_by_name["empty"],
            probability_static_presence=probability_by_name[
                "static_presence"
            ],
            probability_movement=probability_by_name["movement"],
        )

    def diagnostics(self) -> RealtimeEngineDiagnostics:
        return RealtimeEngineDiagnostics(
            accepted_packets=self.accepted_packets,
            rejected_packets=self.rejected_packets,
            inference_count=self.inference_count,
            packets_since_inference=self.packets_since_inference,
            buffer_packets=len(self.packet_buffer),
            required_buffer_packets=self.required_buffer_packets,
            window_packets=self.window_packets,
            step_packets=self.step_packets,
            filter_context_packets=self.filter_context_packets,
            packet_rate_hz=self.packet_rate_hz(),
            last_error=self.last_error,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the incremental realtime inference artifacts."
    )
    parser.add_argument(
        "--pipeline-config",
        type=Path,
    )
    parser.add_argument(
        "--model",
        type=Path,
    )
    parser.add_argument(
        "--state-machine-config",
        type=Path,
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Load the artifacts and print the engine configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.print_config:
        raise SystemExit(
            "Use --print-config to validate and inspect the realtime engine."
        )

    engine = IncrementalRealtimeInferenceEngine.from_artifacts(
        pipeline_config_path=args.pipeline_config,
        model_path=args.model,
        state_machine_config_path=args.state_machine_config,
        verify_model_hash=not args.skip_hash_check,
    )
    diagnostics = engine.diagnostics()

    print()
    print("Incremental Realtime Inference Engine")
    print("=" * 72)
    print(f"Pipeline config:        {engine.pipeline_config_path}")
    print(f"Model:                  {engine.model_path}")
    print(f"State-machine config:   {engine.state_machine_config_path}")
    print(f"Expected subcarriers:   {engine.expected_subcarriers}")
    print(f"Window packets:         {engine.window_packets}")
    print(f"Step packets:           {engine.step_packets}")
    print(f"Filter context:         {engine.filter_context_packets}")
    print(f"Required buffer:        {engine.required_buffer_packets}")
    print(f"Selected subcarriers:   {engine.selected_subcarriers.size}")
    print(f"Selected features:      {engine.selected_feature_indices.size}")
    print(f"Initial stable state:   {engine.state_machine.current_state}")
    print(f"Diagnostics:            {diagnostics.to_dict()}")


if __name__ == "__main__":
    main()
