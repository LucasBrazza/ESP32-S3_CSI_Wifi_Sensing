"""
Confidence-aware Temporal Decision Simulation

Experiment 022

This experiment extends the scenario-based realtime simulation by adding
confidence-aware temporal rules.

Purpose:

    Reduce false state changes without making the system too slow to react.

It compares:

    - raw prediction
    - majority vote baselines
    - confidence-gated state transitions
    - confidence + confirmation state machines
    - movement-priority transition rules

Important:

    This is still an architecture simulation. Scenario streams are assembled
    from the current short files, so results should guide realtime logic but
    should not be treated as final model validation.
"""

import csv
import importlib


base = importlib.import_module(
    "Tools.realtime.02_scenario_stream_state_machine"
)


EXPERIMENT_ID = "022"
EXPERIMENT_FOLDER_NAME = "022_confidence_aware_temporal_decision"

CONFIDENCE_STRATEGIES = [
    {
        "name": "raw_prediction",
        "type": "raw",
    },
    {
        "name": "majority_vote_3",
        "type": "majority_vote",
        "vote_window": 3,
    },
    {
        "name": "majority_vote_5",
        "type": "majority_vote",
        "vote_window": 5,
    },
    {
        "name": "confidence_hold_055",
        "type": "confidence_hold",
        "min_confidence": 0.55,
    },
    {
        "name": "confidence_hold_065",
        "type": "confidence_hold",
        "min_confidence": 0.65,
    },
    {
        "name": "confirm2_conf050_override075",
        "type": "confidence_state_machine",
        "confirmations": 2,
        "min_confidence": 0.50,
        "high_confidence_override": 0.75,
    },
    {
        "name": "confirm2_conf060_override080",
        "type": "confidence_state_machine",
        "confirmations": 2,
        "min_confidence": 0.60,
        "high_confidence_override": 0.80,
    },
    {
        "name": "confirm3_conf050_override080",
        "type": "confidence_state_machine",
        "confirmations": 3,
        "min_confidence": 0.50,
        "high_confidence_override": 0.80,
    },
    {
        "name": "vote3_confirm2_conf050",
        "type": "vote_then_confidence_state_machine",
        "vote_window": 3,
        "confirmations": 2,
        "min_confidence": 0.50,
        "high_confidence_override": 0.75,
    },
    {
        "name": "vote3_confirm2_conf060",
        "type": "vote_then_confidence_state_machine",
        "vote_window": 3,
        "confirmations": 2,
        "min_confidence": 0.60,
        "high_confidence_override": 0.80,
    },
    {
        "name": "movement_priority_confirm",
        "type": "label_specific_confidence_state_machine",
        "default_confirmations": 3,
        "movement_confirmations": 2,
        "min_confidence": 0.55,
        "movement_min_confidence": 0.50,
        "high_confidence_override": 0.80,
    },
]


# ================= DECISION HELPERS =================

def mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def std(values):
    if not values:
        return 0.0

    average = mean(values)
    total = 0.0

    for value in values:
        difference = value - average
        total += difference * difference

    return (total / len(values)) ** 0.5


def apply_confidence_hold(raw_labels, confidences, min_confidence):
    if not raw_labels:
        return []

    current_state = raw_labels[0]
    output_labels = []

    for raw_label, confidence in zip(raw_labels, confidences):
        if confidence >= min_confidence:
            current_state = raw_label

        output_labels.append(current_state)

    return output_labels


def apply_confidence_state_machine(
    raw_labels,
    confidences,
    confirmations,
    min_confidence,
    high_confidence_override,
):
    if not raw_labels:
        return []

    current_state = raw_labels[0]
    candidate_state = None
    candidate_count = 0
    candidate_confidences = []

    output_labels = []

    for raw_label, confidence in zip(raw_labels, confidences):
        if raw_label == current_state:
            candidate_state = None
            candidate_count = 0
            candidate_confidences = []
            output_labels.append(current_state)
            continue

        if confidence >= high_confidence_override:
            current_state = raw_label
            candidate_state = None
            candidate_count = 0
            candidate_confidences = []
            output_labels.append(current_state)
            continue

        if raw_label == candidate_state:
            candidate_count += 1
            candidate_confidences.append(confidence)
        else:
            candidate_state = raw_label
            candidate_count = 1
            candidate_confidences = [confidence]

        if (
            candidate_count >= confirmations
            and mean(candidate_confidences) >= min_confidence
        ):
            current_state = candidate_state
            candidate_state = None
            candidate_count = 0
            candidate_confidences = []

        output_labels.append(current_state)

    return output_labels


def apply_label_specific_confidence_state_machine(
    raw_labels,
    confidences,
    default_confirmations,
    movement_confirmations,
    min_confidence,
    movement_min_confidence,
    high_confidence_override,
):
    if not raw_labels:
        return []

    current_state = raw_labels[0]
    candidate_state = None
    candidate_count = 0
    candidate_confidences = []

    output_labels = []

    for raw_label, confidence in zip(raw_labels, confidences):
        if raw_label == current_state:
            candidate_state = None
            candidate_count = 0
            candidate_confidences = []
            output_labels.append(current_state)
            continue

        if confidence >= high_confidence_override:
            current_state = raw_label
            candidate_state = None
            candidate_count = 0
            candidate_confidences = []
            output_labels.append(current_state)
            continue

        if raw_label == candidate_state:
            candidate_count += 1
            candidate_confidences.append(confidence)
        else:
            candidate_state = raw_label
            candidate_count = 1
            candidate_confidences = [confidence]

        required_confirmations = default_confirmations
        required_confidence = min_confidence

        if candidate_state == "movement":
            required_confirmations = movement_confirmations
            required_confidence = movement_min_confidence

        if (
            candidate_count >= required_confirmations
            and mean(candidate_confidences) >= required_confidence
        ):
            current_state = candidate_state
            candidate_state = None
            candidate_count = 0
            candidate_confidences = []

        output_labels.append(current_state)

    return output_labels


def apply_confidence_strategy(raw_labels, confidences, strategy):
    if strategy["type"] == "raw":
        return raw_labels[:]

    if strategy["type"] == "majority_vote":
        return base.apply_majority_vote(
            raw_labels,
            strategy["vote_window"],
        )

    if strategy["type"] == "confidence_hold":
        return apply_confidence_hold(
            raw_labels,
            confidences,
            strategy["min_confidence"],
        )

    if strategy["type"] == "confidence_state_machine":
        return apply_confidence_state_machine(
            raw_labels,
            confidences,
            strategy["confirmations"],
            strategy["min_confidence"],
            strategy["high_confidence_override"],
        )

    if strategy["type"] == "vote_then_confidence_state_machine":
        voted_labels = base.apply_majority_vote(
            raw_labels,
            strategy["vote_window"],
        )

        return apply_confidence_state_machine(
            voted_labels,
            confidences,
            strategy["confirmations"],
            strategy["min_confidence"],
            strategy["high_confidence_override"],
        )

    if strategy["type"] == "label_specific_confidence_state_machine":
        return apply_label_specific_confidence_state_machine(
            raw_labels,
            confidences,
            strategy["default_confirmations"],
            strategy["movement_confirmations"],
            strategy["min_confidence"],
            strategy["movement_min_confidence"],
            strategy["high_confidence_override"],
        )

    raise ValueError(f"Unknown strategy type: {strategy['type']}")


# ================= EVALUATION =================

def evaluate_scenario_strategy(raw_records, strategy):
    raw_labels = []
    true_labels = []
    confidences = []

    for record in raw_records:
        raw_labels.append(record["raw_predicted_label"])
        true_labels.append(record["true_label"])
        confidences.append(float(record["raw_confidence"]))

    final_labels = apply_confidence_strategy(
        raw_labels,
        confidences,
        strategy,
    )

    metrics = base.compute_metrics(
        true_labels,
        final_labels,
    )

    state_changes = base.count_state_changes(final_labels)
    true_state_changes = base.count_state_changes(true_labels)

    extra_state_changes = max(
        0,
        state_changes - true_state_changes,
    )

    missing_state_changes = max(
        0,
        true_state_changes - state_changes,
    )

    transition_info = base.compute_transition_delay(
        true_labels,
        final_labels,
    )

    stability_penalty = (
        0.005 * extra_state_changes
        + 0.030 * missing_state_changes
        + 0.010 * transition_info["transition_delay_mean"]
    )

    realtime_score = metrics["macro_f1"] - stability_penalty

    result = {
        "seed": raw_records[0]["seed"],
        "scenario_id": raw_records[0]["scenario_id"],
        "quadrant": raw_records[0]["quadrant"],
        "strategy": strategy["name"],
        "windows": len(raw_records),

        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "empty_f1": metrics["empty_f1"],
        "static_presence_f1": metrics["static_presence_f1"],
        "movement_f1": metrics["movement_f1"],

        "true_state_changes": true_state_changes,
        "final_state_changes": state_changes,
        "extra_state_changes": extra_state_changes,
        "missing_state_changes": missing_state_changes,

        "transition_count": transition_info["transition_count"],
        "transition_success_count": transition_info["transition_success_count"],
        "transition_success_rate": transition_info["transition_success_rate"],
        "transition_delay_mean": transition_info["transition_delay_mean"],

        "confidence_mean": mean(confidences),
        "confidence_std": std(confidences),
        "realtime_score": realtime_score,
    }

    window_rows = []

    for index, record in enumerate(raw_records):
        window_rows.append(
            {
                "seed": record["seed"],
                "scenario_id": record["scenario_id"],
                "quadrant": record["quadrant"],
                "strategy": strategy["name"],
                "segment_index": record["segment_index"],
                "segment_label": record["segment_label"],
                "window_in_segment": record["window_in_segment"],
                "stream_window_index": record["stream_window_index"],
                "source_group_id": record["source_group_id"],
                "source_file": record["source_file"],
                "true_label": record["true_label"],
                "raw_predicted_label": record["raw_predicted_label"],
                "final_predicted_label": final_labels[index],
                "raw_confidence": record["raw_confidence"],
                "correct": final_labels[index] == record["true_label"],
            }
        )

    return result, window_rows


def summarize_strategy_results(scenario_results):
    rows_by_strategy = {}

    for row in scenario_results:
        strategy = row["strategy"]

        if strategy not in rows_by_strategy:
            rows_by_strategy[strategy] = []

        rows_by_strategy[strategy].append(row)

    summary_rows = []

    for strategy in rows_by_strategy:
        rows = rows_by_strategy[strategy]

        accuracy_values = []
        macro_f1_values = []
        weighted_f1_values = []
        movement_f1_values = []
        empty_f1_values = []
        static_f1_values = []
        final_state_changes_values = []
        extra_state_changes_values = []
        missing_state_changes_values = []
        transition_success_values = []
        transition_delay_values = []
        realtime_score_values = []

        for row in rows:
            accuracy_values.append(float(row["accuracy"]))
            macro_f1_values.append(float(row["macro_f1"]))
            weighted_f1_values.append(float(row["weighted_f1"]))
            movement_f1_values.append(float(row["movement_f1"]))
            empty_f1_values.append(float(row["empty_f1"]))
            static_f1_values.append(float(row["static_presence_f1"]))
            final_state_changes_values.append(float(row["final_state_changes"]))
            extra_state_changes_values.append(float(row["extra_state_changes"]))
            missing_state_changes_values.append(float(row["missing_state_changes"]))
            transition_success_values.append(float(row["transition_success_rate"]))
            transition_delay_values.append(float(row["transition_delay_mean"]))
            realtime_score_values.append(float(row["realtime_score"]))

        summary_rows.append(
            {
                "strategy": strategy,
                "scenarios": len(rows),

                "accuracy_mean": mean(accuracy_values),
                "accuracy_std": std(accuracy_values),

                "macro_f1_mean": mean(macro_f1_values),
                "macro_f1_std": std(macro_f1_values),

                "weighted_f1_mean": mean(weighted_f1_values),

                "empty_f1_mean": mean(empty_f1_values),
                "static_presence_f1_mean": mean(static_f1_values),
                "movement_f1_mean": mean(movement_f1_values),

                "final_state_changes_mean": mean(final_state_changes_values),
                "extra_state_changes_mean": mean(extra_state_changes_values),
                "missing_state_changes_mean": mean(missing_state_changes_values),

                "transition_success_rate_mean": mean(transition_success_values),
                "transition_delay_mean": mean(transition_delay_values),

                "realtime_score_mean": mean(realtime_score_values),
                "realtime_score_std": std(realtime_score_values),
            }
        )

    summary_rows = sorted(
        summary_rows,
        key=lambda item: (
            item["realtime_score_mean"],
            item["macro_f1_mean"],
            item["movement_f1_mean"],
            -item["extra_state_changes_mean"],
            -item["transition_delay_mean"],
        ),
        reverse=True,
    )

    return summary_rows


# ================= OUTPUT =================

def get_experiment_paths():
    run_dir = base.RESULTS_DIR / "runs" / EXPERIMENT_FOLDER_NAME
    tables_dir = run_dir / "tables"

    tables_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "tables_dir": tables_dir,

        "scenario_results": tables_dir / "confidence_scenario_results.csv",
        "strategy_summary": tables_dir / "confidence_strategy_summary.csv",
        "window_predictions": tables_dir / "confidence_window_predictions.csv",
        "experiment_notes": run_dir / "experiment_notes.txt",
    }


def save_csv(rows, output_path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def update_experiment_index(best_summary):
    index_path = base.RESULTS_DIR / "experiment_index.csv"

    fieldnames = [
        "experiment_id",
        "folder",
        "description",
        "model",
        "validation_method",
        "top_k",
        "max_depth",
        "min_samples_split",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "main_observation",
    ]

    existing_rows = []

    if index_path.exists():
        with open(index_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row.get("experiment_id") != EXPERIMENT_ID:
                    existing_rows.append(row)

    new_row = {
        "experiment_id": EXPERIMENT_ID,
        "folder": EXPERIMENT_FOLDER_NAME,
        "description": "Confidence-aware realtime temporal decision simulation",
        "model": "gradient_boosting_depth3_20 + confidence-aware temporal layer",
        "validation_method": "Artificial scenario streams assembled from file-based holdout test groups",
        "top_k": base.MODEL_TOP_K,
        "max_depth": base.MODEL_MAX_DEPTH,
        "min_samples_split": "",
        "accuracy": f"{best_summary['accuracy_mean']:.6f}",
        "macro_f1": f"{best_summary['macro_f1_mean']:.6f}",
        "weighted_f1": f"{best_summary['weighted_f1_mean']:.6f}",
        "main_observation": (
            f"Best realtime-score strategy: {best_summary['strategy']}. "
            f"Realtime score={best_summary['realtime_score_mean']:.6f}. "
            "This favors accuracy while penalizing false transitions and delay."
        ),
    }

    existing_rows.append(new_row)

    with open(index_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in existing_rows:
            writer.writerow(row)


def save_experiment_notes(
    output_path,
    feature_dataset,
    scenario_results,
    strategy_summary,
):
    best = strategy_summary[0]

    lines = []

    lines.append("Experiment 022 - Confidence-aware temporal decision simulation")
    lines.append("")
    lines.append("Objective:")
    lines.append(
        "Evaluate confidence-aware temporal rules to reduce false state "
        "changes while preserving class transition responsiveness."
    )
    lines.append("")
    lines.append("Scenario sequence:")
    lines.append(" -> ".join(base.SEGMENT_SEQUENCE))
    lines.append("")
    lines.append("Model:")
    lines.append(f"- Gradient Boosting, n_estimators={base.MODEL_N_ESTIMATORS}")
    lines.append(f"- max_depth={base.MODEL_MAX_DEPTH}")
    lines.append(f"- top_k={base.MODEL_TOP_K}")
    lines.append("")
    lines.append("Dataset:")
    lines.append(f"- Total samples/windows: {len(feature_dataset)}")
    lines.append(f"- Features per sample: {len(feature_dataset[0]['features'])}")
    lines.append("")
    lines.append("Simulation parameters:")
    lines.append(f"- Target windows per segment: {base.TARGET_WINDOWS_PER_SEGMENT}")
    lines.append(f"- Scenarios per quadrant per seed: {base.SCENARIOS_PER_QUADRANT_PER_SEED}")
    lines.append(f"- Generated scenario evaluations: {len(scenario_results)}")
    lines.append("")
    lines.append("Best strategy by realtime score:")
    lines.append(f"- strategy: {best['strategy']}")
    lines.append(f"- realtime score mean: {best['realtime_score_mean']:.6f}")
    lines.append(f"- accuracy mean: {best['accuracy_mean']:.6f}")
    lines.append(f"- macro F1 mean: {best['macro_f1_mean']:.6f}")
    lines.append(f"- movement F1 mean: {best['movement_f1_mean']:.6f}")
    lines.append(f"- final state changes mean: {best['final_state_changes_mean']:.6f}")
    lines.append(f"- extra state changes mean: {best['extra_state_changes_mean']:.6f}")
    lines.append(f"- missing state changes mean: {best['missing_state_changes_mean']:.6f}")
    lines.append(f"- transition success rate mean: {best['transition_success_rate_mean']:.6f}")
    lines.append(f"- transition delay mean: {best['transition_delay_mean']:.6f}")
    lines.append("")
    lines.append("Realtime score:")
    lines.append(
        "The realtime score is macro F1 penalized by extra state changes, "
        "missing state changes, and transition delay. It is used only to "
        "compare temporal decision strategies."
    )
    lines.append("")
    lines.append("Important note:")
    lines.append(
        "Scenario streams are artificially assembled from short files of the "
        "same class/quadrant. This allows realtime logic development before "
        "Dataset v2, but these metrics should not be treated as final model "
        "performance."
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


# ================= MAIN =================

def run_experiment():
    feature_dataset = base.load_pickle(base.FEATURE_DATASET_FILE)

    if not feature_dataset:
        raise ValueError("Feature dataset is empty.")

    features_per_sample = len(feature_dataset[0]["features"])

    if features_per_sample != base.EXPECTED_FEATURES_PER_SAMPLE:
        raise ValueError(
            "This experiment must be run with the official 014 feature set. "
            f"Expected {base.EXPECTED_FEATURES_PER_SAMPLE} features per sample, "
            f"but found {features_per_sample}."
        )

    print()
    print("Confidence-aware Temporal Decision Simulation")
    print("=" * 70)
    print("Experiment:", EXPERIMENT_ID)
    print("Feature dataset:", base.FEATURE_DATASET_FILE)
    print("Samples/windows:", len(feature_dataset))
    print("Seeds:", base.SEEDS)
    print()

    all_scenario_results = []
    all_window_prediction_rows = []

    for seed in base.SEEDS:
        print()
        print("=" * 70)
        print(f"Seed {seed}")
        print("=" * 70)

        train_dataset, test_dataset, train_groups, test_groups = base.file_stratified_holdout_split(
            feature_dataset,
            base.TEST_SIZE,
            seed,
        )

        selected_train_dataset, selected_indices = base.select_top_k_from_training(
            train_dataset,
            base.MODEL_TOP_K,
        )

        model = base.train_gradient_boosting(
            selected_train_dataset,
            seed,
        )

        scenarios = base.build_scenarios_from_test_groups(
            test_groups,
            seed,
        )

        print(f"Generated scenarios: {len(scenarios)}")

        for scenario in scenarios:
            raw_records = base.predict_stream_windows(
                model,
                scenario["windows"],
                selected_indices,
            )

            for strategy in CONFIDENCE_STRATEGIES:
                scenario_result, window_rows = evaluate_scenario_strategy(
                    raw_records,
                    strategy,
                )

                all_scenario_results.append(scenario_result)
                all_window_prediction_rows.extend(window_rows)

    if not all_scenario_results:
        raise ValueError(
            "No scenarios were generated. The test split did not contain "
            "all required labels within any quadrant."
        )

    strategy_summary = summarize_strategy_results(
        all_scenario_results,
    )

    paths = get_experiment_paths()

    save_csv(
        all_scenario_results,
        paths["scenario_results"],
    )

    save_csv(
        strategy_summary,
        paths["strategy_summary"],
    )

    save_csv(
        all_window_prediction_rows,
        paths["window_predictions"],
    )

    save_experiment_notes(
        paths["experiment_notes"],
        feature_dataset,
        all_scenario_results,
        strategy_summary,
    )

    update_experiment_index(
        strategy_summary[0],
    )

    print()
    print("=" * 70)
    print("Confidence-aware temporal decision simulation finished.")
    print("=" * 70)
    print("Best strategy by realtime score:")
    print(f"strategy: {strategy_summary[0]['strategy']}")
    print(f"realtime_score_mean: {strategy_summary[0]['realtime_score_mean']:.6f}")
    print(f"accuracy_mean: {strategy_summary[0]['accuracy_mean']:.6f}")
    print(f"macro_f1_mean: {strategy_summary[0]['macro_f1_mean']:.6f}")
    print(f"movement_f1_mean: {strategy_summary[0]['movement_f1_mean']:.6f}")
    print(f"extra_state_changes_mean: {strategy_summary[0]['extra_state_changes_mean']:.6f}")
    print(f"transition_delay_mean: {strategy_summary[0]['transition_delay_mean']:.6f}")
    print()
    print("Results saved to:")
    print(paths["run_dir"])


def main():
    run_experiment()


if __name__ == "__main__":
    main()