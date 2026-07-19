from __future__ import annotations

"""
Minimal temporal state machine for Dataset v2 realtime inference.

Goals
-----
* Start in ``empty`` because the operational protocol requires an empty room.
* Preserve the physically meaningful movement state.
* Avoid conservative biases that reduce per-window accuracy.
* Keep direct empty/static transitions configurable rather than impossible.
* Remain compatible with ``01_realtime_inference.py``.

With smoothing [1.0], startup_discard_windows=0 and one confirmation for every
transition, the stable output reproduces the raw classifier output. This is
the baseline used by ``02_tune_state_machine.py``.
"""

import argparse
import csv
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


CLASS_ORDER = ("empty", "static_presence", "movement")


DEFAULT_CONFIG: dict[str, Any] = {
    "initial_state": "empty",
    "startup_discard_windows": 0,
    "probability_smoothing": {
        "weights_oldest_to_newest": [1.0]
    },
    "normal_transitions": {
        "empty_to_movement": {
            "confirmations": 1,
            "min_target_probability": 0.50,
        },
        "movement_to_static_presence": {
            "confirmations": 1,
            "min_target_probability": 0.50,
        },
        "movement_to_empty": {
            "confirmations": 1,
            "min_target_probability": 0.50,
        },
        "static_presence_to_movement": {
            "confirmations": 1,
            "min_target_probability": 0.50,
        },
    },
    "direct_transitions": {
        "empty_to_static_presence": {
            "enabled": True,
            "confirmations": 2,
            "min_target_probability": 0.55,
        },
        "static_presence_to_empty": {
            "enabled": True,
            "confirmations": 2,
            "min_target_probability": 0.55,
        },
    },
}


@dataclass(frozen=True)
class TransitionRule:
    confirmations: int
    min_target_probability: float
    enabled: bool = True


@dataclass(frozen=True)
class StateMachineDecision:
    window_index: int
    warmup: bool
    raw_state: str
    smoothed_state: str
    stable_state_before: str
    stable_state: str
    movement_origin: str | None
    movement_age_windows: int
    candidate_state: str | None
    candidate_count: int
    transition_accepted: bool
    transition_reason: str
    inferred_transition: bool
    probability_empty: float
    probability_static_presence: float
    probability_movement: float
    probability_presence: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_configuration(path: Path | None) -> dict[str, Any]:
    if path is None:
        config = deep_merge({}, DEFAULT_CONFIG)
    else:
        if not path.exists():
            raise FileNotFoundError(f"State-machine config not found: {path}")

        with path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)

        if "state_machine" in loaded:
            loaded = loaded["state_machine"]

        config = deep_merge(DEFAULT_CONFIG, loaded)

    validate_configuration(config)
    return config


def _validate_rule(
    raw_rule: Mapping[str, Any],
    rule_name: str,
) -> None:
    confirmations = int(raw_rule["confirmations"])
    probability = float(raw_rule["min_target_probability"])

    if confirmations < 1:
        raise ValueError(f"{rule_name}: confirmations must be >= 1.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"{rule_name}: min_target_probability must be in [0, 1]."
        )


def validate_configuration(config: Mapping[str, Any]) -> None:
    if config.get("initial_state") != "empty":
        raise ValueError(
            "The operational protocol requires initial_state='empty'."
        )

    startup = int(config.get("startup_discard_windows", 0))
    if startup < 0:
        raise ValueError("startup_discard_windows cannot be negative.")

    weights = [
        float(value)
        for value in config["probability_smoothing"][
            "weights_oldest_to_newest"
        ]
    ]
    if not weights:
        raise ValueError("At least one smoothing weight is required.")
    if any(value < 0 for value in weights):
        raise ValueError("Smoothing weights cannot be negative.")
    if sum(weights) <= 0:
        raise ValueError("Smoothing weights must have a positive sum.")

    for name, raw_rule in config["normal_transitions"].items():
        _validate_rule(raw_rule, f"normal_transitions.{name}")

    for name, raw_rule in config["direct_transitions"].items():
        _validate_rule(raw_rule, f"direct_transitions.{name}")


def normalize_probabilities(
    values: Mapping[str, float] | Sequence[float],
    class_order: Sequence[str] = CLASS_ORDER,
) -> dict[str, float]:
    if isinstance(values, Mapping):
        probabilities = {
            label: float(values.get(label, 0.0))
            for label in class_order
        }
    else:
        if len(values) != len(class_order):
            raise ValueError(
                f"Expected {len(class_order)} probabilities, got {len(values)}."
            )
        probabilities = {
            label: float(value)
            for label, value in zip(class_order, values)
        }

    for label, value in probabilities.items():
        if not math.isfinite(value):
            raise ValueError(f"Non-finite probability for {label}: {value}")
        if value < 0:
            raise ValueError(f"Negative probability for {label}: {value}")

    total = sum(probabilities.values())
    if total <= 0:
        raise ValueError("Probability sum must be positive.")

    return {
        label: probabilities[label] / total
        for label in class_order
    }


def winner(probabilities: Mapping[str, float]) -> str:
    return max(
        CLASS_ORDER,
        key=lambda label: (
            float(probabilities[label]),
            -CLASS_ORDER.index(label),
        ),
    )


class TemporalStateMachine:
    """Minimal state machine that stays close to raw model decisions."""

    NORMAL_RULE_NAMES: dict[tuple[str, str], str] = {
        ("empty", "movement"): "empty_to_movement",
        ("movement", "static_presence"): "movement_to_static_presence",
        ("movement", "empty"): "movement_to_empty",
        ("static_presence", "movement"): "static_presence_to_movement",
    }

    DIRECT_RULE_NAMES: dict[tuple[str, str], str] = {
        ("empty", "static_presence"): "empty_to_static_presence",
        ("static_presence", "empty"): "static_presence_to_empty",
    }

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        merged = deep_merge(DEFAULT_CONFIG, config or {})
        validate_configuration(merged)
        self.config = merged

        weights = [
            float(value)
            for value in merged["probability_smoothing"][
                "weights_oldest_to_newest"
            ]
        ]
        total = sum(weights)
        self.smoothing_weights = [value / total for value in weights]
        self.probability_history: deque[dict[str, float]] = deque(
            maxlen=len(self.smoothing_weights)
        )

        self.current_state = "empty"
        self.window_index = -1
        self.movement_origin: str | None = None
        self.movement_age_windows = 0
        self.candidate_state: str | None = None
        self.candidate_count = 0

    def reset(self) -> None:
        self.current_state = "empty"
        self.window_index = -1
        self.movement_origin = None
        self.movement_age_windows = 0
        self.candidate_state = None
        self.candidate_count = 0
        self.probability_history.clear()

    def _smoothed_probabilities(
        self,
        probabilities: dict[str, float],
    ) -> dict[str, float]:
        self.probability_history.append(probabilities)
        history = list(self.probability_history)

        weights = self.smoothing_weights[-len(history):]
        total = sum(weights)
        normalized_weights = [value / total for value in weights]

        return {
            label: sum(
                row[label] * weight
                for row, weight in zip(history, normalized_weights)
            )
            for label in CLASS_ORDER
        }

    @staticmethod
    def _rule(
        raw_rule: Mapping[str, Any],
    ) -> TransitionRule:
        return TransitionRule(
            confirmations=int(raw_rule["confirmations"]),
            min_target_probability=float(
                raw_rule["min_target_probability"]
            ),
            enabled=bool(raw_rule.get("enabled", True)),
        )

    def _reset_candidate(self) -> None:
        self.candidate_state = None
        self.candidate_count = 0

    def _advance_candidate(
        self,
        target: str,
        target_probability: float,
        rule: TransitionRule,
    ) -> bool:
        if not rule.enabled:
            self._reset_candidate()
            return False

        if target_probability < rule.min_target_probability:
            self._reset_candidate()
            return False

        if self.candidate_state == target:
            self.candidate_count += 1
        else:
            self.candidate_state = target
            self.candidate_count = 1

        return self.candidate_count >= rule.confirmations

    def _accept_transition(
        self,
        target: str,
    ) -> tuple[str, str | None]:
        source = self.current_state
        previous_origin = self.movement_origin

        if target == "movement":
            self.movement_origin = source
            self.movement_age_windows = 0
        elif source == "movement":
            self.movement_origin = None
            self.movement_age_windows = 0

        self.current_state = target
        self._reset_candidate()
        return source, previous_origin

    def update(
        self,
        probabilities: Mapping[str, float] | Sequence[float],
    ) -> StateMachineDecision:
        normalized = normalize_probabilities(probabilities)
        raw_state = winner(normalized)
        smoothed = self._smoothed_probabilities(normalized)
        smoothed_state = winner(smoothed)

        self.window_index += 1
        stable_before = self.current_state
        origin_before = self.movement_origin
        accepted = False
        inferred = False
        reason = "stable_state_confirmed"

        warmup = (
            self.window_index
            < int(self.config["startup_discard_windows"])
        )

        if self.current_state == "movement":
            self.movement_age_windows += 1

        if warmup:
            self._reset_candidate()
            reason = "startup_warmup_state_locked_empty"

        elif smoothed_state == self.current_state:
            self._reset_candidate()
            reason = "stable_state_confirmed"

        else:
            pair = (self.current_state, smoothed_state)

            if pair in self.NORMAL_RULE_NAMES:
                rule_name = self.NORMAL_RULE_NAMES[pair]
                rule = self._rule(
                    self.config["normal_transitions"][rule_name]
                )
                accepted = self._advance_candidate(
                    smoothed_state,
                    float(smoothed[smoothed_state]),
                    rule,
                )

                if accepted:
                    source, previous_origin = self._accept_transition(
                        smoothed_state
                    )
                    reason = (
                        f"normal_transition:{source}_to_{smoothed_state}"
                    )
                elif not rule.enabled:
                    reason = f"normal_transition_disabled:{rule_name}"
                elif (
                    float(smoothed[smoothed_state])
                    < rule.min_target_probability
                ):
                    reason = f"normal_transition_below_threshold:{rule_name}"
                else:
                    reason = (
                        f"normal_transition_waiting_confirmation:{rule_name}"
                    )

            elif pair in self.DIRECT_RULE_NAMES:
                rule_name = self.DIRECT_RULE_NAMES[pair]
                rule = self._rule(
                    self.config["direct_transitions"][rule_name]
                )
                accepted = self._advance_candidate(
                    smoothed_state,
                    float(smoothed[smoothed_state]),
                    rule,
                )

                if accepted:
                    source, previous_origin = self._accept_transition(
                        smoothed_state
                    )
                    inferred = True
                    if pair == ("empty", "static_presence"):
                        reason = "inferred_entry_movement_not_observed"
                    else:
                        reason = "inferred_exit_movement_not_observed"
                elif not rule.enabled:
                    reason = f"direct_transition_blocked:{rule_name}"
                elif (
                    float(smoothed[smoothed_state])
                    < rule.min_target_probability
                ):
                    reason = f"direct_transition_below_threshold:{rule_name}"
                else:
                    reason = (
                        f"direct_transition_waiting_confirmation:{rule_name}"
                    )

            else:
                self._reset_candidate()
                reason = "transition_not_supported"

        reported_origin: str | None
        if self.current_state == "movement":
            reported_origin = self.movement_origin
        elif stable_before == "movement":
            reported_origin = origin_before
        else:
            reported_origin = None

        confidence = float(smoothed[smoothed_state])

        return StateMachineDecision(
            window_index=self.window_index,
            warmup=warmup,
            raw_state=raw_state,
            smoothed_state=smoothed_state,
            stable_state_before=stable_before,
            stable_state=self.current_state,
            movement_origin=reported_origin,
            movement_age_windows=self.movement_age_windows,
            candidate_state=self.candidate_state,
            candidate_count=self.candidate_count,
            transition_accepted=accepted,
            transition_reason=reason,
            inferred_transition=inferred,
            probability_empty=float(smoothed["empty"]),
            probability_static_presence=float(
                smoothed["static_presence"]
            ),
            probability_movement=float(smoothed["movement"]),
            probability_presence=float(
                smoothed["static_presence"] + smoothed["movement"]
            ),
            confidence=confidence,
        )


def demo_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        scenario: str,
        true_state: str,
        probabilities: tuple[float, float, float],
        repetitions: int,
    ) -> None:
        for _ in range(repetitions):
            rows.append(
                {
                    "scenario": scenario,
                    "true_state": true_state,
                    "probability_empty": probabilities[0],
                    "probability_static_presence": probabilities[1],
                    "probability_movement": probabilities[2],
                }
            )

    add("startup_empty", "empty", (0.85, 0.10, 0.05), 3)

    # One false static prediction must not change state with the default config.
    add("isolated_static_error", "empty", (0.30, 0.60, 0.10), 1)
    add("isolated_static_error", "empty", (0.82, 0.12, 0.06), 2)

    add("normal_entry", "movement", (0.05, 0.10, 0.85), 2)
    add("normal_entry", "static_presence", (0.08, 0.84, 0.08), 2)

    add("normal_exit", "movement", (0.08, 0.10, 0.82), 2)
    add("normal_exit", "empty", (0.84, 0.10, 0.06), 2)

    # Missed motion: direct transitions need two strong confirmations.
    add("missed_entry", "static_presence", (0.10, 0.82, 0.08), 2)
    add("missed_exit", "empty", (0.82, 0.12, 0.06), 2)

    return rows


def process_rows(
    input_rows: Sequence[Mapping[str, Any]],
    machine: TemporalStateMachine,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []

    for row_index, row in enumerate(input_rows):
        decision = machine.update(
            {
                "empty": float(row["probability_empty"]),
                "static_presence": float(
                    row["probability_static_presence"]
                ),
                "movement": float(row["probability_movement"]),
            }
        )
        output = dict(row)
        output.setdefault("input_row", row_index)
        output.update(decision.to_dict())
        output_rows.append(output)

    return output_rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("No output rows to save.")

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


def print_transition_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    print()
    print("Minimal Temporal State Machine")
    print("=" * 82)

    previous_scenario: str | None = None
    for row in rows:
        scenario = str(row.get("scenario", "input_csv"))
        if scenario != previous_scenario:
            print()
            print(f"[{scenario}]")
            previous_scenario = scenario

        if bool(row["transition_accepted"]):
            print(
                f"window={int(row['window_index']):03d} | "
                f"raw={row['raw_state']:<16} | "
                f"stable={row['stable_state']:<16} | "
                f"origin={str(row.get('movement_origin') or '-'):<16} | "
                f"reason={row['transition_reason']}"
            )

    print()
    print("-" * 82)
    print(f"Processed windows:       {len(rows)}")
    print(
        "Accepted transitions:    "
        f"{sum(bool(row['transition_accepted']) for row in rows)}"
    )
    print(
        "Inferred transitions:    "
        f"{sum(bool(row['inferred_transition']) for row in rows)}"
    )
    print(f"Final stable state:      {rows[-1]['stable_state']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the minimal temporal state machine."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--input-csv", type=Path)

    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_configuration(args.config)
    rows = demo_rows() if args.demo else read_csv(args.input_csv)
    output = process_rows(rows, TemporalStateMachine(config))
    print_transition_summary(output)

    if args.output_csv:
        write_csv(args.output_csv, output)
        print(f"Detailed output:         {args.output_csv}")


if __name__ == "__main__":
    main()
