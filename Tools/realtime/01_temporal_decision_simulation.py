"""
Realtime Temporal Decision Simulation

Experiment 020

This script simulates the realtime decision layer using the current
window-based CSI feature dataset.

Purpose:

    Compare raw per-window classification against simple temporal
    stabilization strategies:

        - raw window prediction
        - rolling majority vote
        - state machine / hysteresis

This experiment does not claim final realtime performance because the
current dataset has very short file sequences. It is mainly used to build
and validate the realtime architecture before collecting Dataset v2.
"""

import csv
import random
from collections import Counter

from Tools.common.io_utils import load_pickle

from Tools.common.project_paths import (
    FEATURE_DATASET_FILE,
    RESULTS_DIR,
)

from Tools.preprocessing.feature_selection import (
    rank_features_by_fisher_score,
)


try:
    from sklearn.ensemble import GradientBoostingClassifier
except ImportError as error:
    raise ImportError(
        "scikit-learn is required for experiment 020. "
        "Install it with: pip install scikit-learn"
    ) from error


CLASS_ORDER = [
    "empty",
    "static_presence",
    "movement",
]

EXPERIMENT_ID = "020"
EXPERIMENT_FOLDER_NAME = "020_realtime_temporal_decision_simulation"

TEST_SIZE = 0.20
EXPECTED_FEATURES_PER_SAMPLE = 231

MODEL_TOP_K = 231
MODEL_N_ESTIMATORS = 20
MODEL_MAX_DEPTH = 3
MODEL_LEARNING_RATE = 0.10

SEEDS = [
    7,
    13,
    21,
    42,
    84,
    126,
    168,
    210,
    336,
    512,
]


TEMPORAL_STRATEGIES = [
    {
        "name": "raw_window_prediction",
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
        "name": "state_machine_confirm_2",
        "type": "state_machine",
        "confirmations": 2,
    },
    {
        "name": "state_machine_confirm_3",
        "type": "state_machine",
        "confirmations": 3,
    },
]


# ================= BASIC HELPERS =================

def safe_division(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def std(values):
    if not values:
        return 0.0

    avg = mean(values)
    total = 0.0

    for value in values:
        difference = value - avg
        total += difference * difference

    return (total / len(values)) ** 0.5


def get_sample_group_id(sample):
    source_file = sample.get("source_file", "")

    if source_file:
        return str(source_file)

    label = sample.get("label", "unknown_label")
    quadrant = sample.get("quadrant", "unknown_quadrant")
    file_name = sample.get("file_name", "unknown_file")

    return f"{label}|{quadrant}|{file_name}"


def get_quadrant(sample):
    return sample.get("quadrant", "unknown")


def get_file_name(sample):
    return sample.get("file_name", "")


def get_source_file(sample):
    return sample.get("source_file", "")


def get_window_index(sample):
    value = sample.get("window_index", 0)

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ================= DATASET GROUPING =================

def group_samples_by_file(dataset):
    groups = {}

    for sample in dataset:
        group_id = get_sample_group_id(sample)

        if group_id not in groups:
            groups[group_id] = []

        groups[group_id].append(sample)

    grouped_items = []

    for group_id, samples in groups.items():
        labels = []
        quadrants = []

        for sample in samples:
            label = sample["label"]
            quadrant = get_quadrant(sample)

            if label not in labels:
                labels.append(label)

            if quadrant not in quadrants:
                quadrants.append(quadrant)

        if len(labels) != 1:
            raise ValueError(
                "A file group contains more than one label. "
                f"group_id={group_id}, labels={labels}"
            )

        if len(quadrants) != 1:
            raise ValueError(
                "A file group contains more than one quadrant. "
                f"group_id={group_id}, quadrants={quadrants}"
            )

        samples = sorted(
            samples,
            key=lambda item: get_window_index(item),
        )

        representative = samples[0]

        grouped_items.append(
            {
                "group_id": group_id,
                "label": labels[0],
                "quadrant": quadrants[0],
                "file_name": get_file_name(representative),
                "source_file": get_source_file(representative),
                "samples": samples,
                "sample_count": len(samples),
            }
        )

    return grouped_items


def file_stratified_holdout_split(dataset, test_size, seed):
    random_generator = random.Random(seed)

    groups = group_samples_by_file(dataset)

    groups_by_label = {}

    for group in groups:
        label = group["label"]

        if label not in groups_by_label:
            groups_by_label[label] = []

        groups_by_label[label].append(group)

    train_groups = []
    test_groups = []

    for label in groups_by_label:
        label_groups = groups_by_label[label][:]
        random_generator.shuffle(label_groups)

        test_group_count = int(round(len(label_groups) * test_size))

        if test_group_count < 1 and len(label_groups) > 1:
            test_group_count = 1

        if test_group_count >= len(label_groups):
            test_group_count = len(label_groups) - 1

        selected_test_groups = label_groups[:test_group_count]
        selected_train_groups = label_groups[test_group_count:]

        test_groups.extend(selected_test_groups)
        train_groups.extend(selected_train_groups)

    random_generator.shuffle(train_groups)
    random_generator.shuffle(test_groups)

    train_dataset = []
    test_dataset = []

    for group in train_groups:
        train_dataset.extend(group["samples"])

    for group in test_groups:
        test_dataset.extend(group["samples"])

    random_generator.shuffle(train_dataset)

    test_dataset = sorted(
        test_dataset,
        key=lambda item: (
            get_sample_group_id(item),
            get_window_index(item),
        ),
    )

    return train_dataset, test_dataset, train_groups, test_groups


# ================= FEATURE SELECTION =================

def select_features_by_indices(feature_dataset, selected_indices):
    selected_dataset = []

    for sample in feature_dataset:
        selected_features = []

        for index in selected_indices:
            selected_features.append(sample["features"][index])

        selected_sample = {}

        for key in sample:
            if key != "features":
                selected_sample[key] = sample[key]

        selected_sample["features"] = selected_features

        selected_dataset.append(selected_sample)

    return selected_dataset


def select_top_k_from_training(train_dataset, test_dataset, top_k):
    ranking = rank_features_by_fisher_score(train_dataset)

    if top_k > len(ranking):
        raise ValueError(
            f"Invalid top_k={top_k}. Ranking has only {len(ranking)} features."
        )

    selected_indices = []

    for item in ranking[:top_k]:
        selected_indices.append(item["feature_index"])

    selected_train_dataset = select_features_by_indices(
        train_dataset,
        selected_indices,
    )

    selected_test_dataset = select_features_by_indices(
        test_dataset,
        selected_indices,
    )

    return selected_train_dataset, selected_test_dataset, selected_indices


def dataset_to_xy(dataset):
    x_values = []
    y_values = []

    for sample in dataset:
        x_values.append(sample["features"])
        y_values.append(sample["label"])

    return x_values, y_values


# ================= MODEL =================

def train_gradient_boosting(train_dataset, seed):
    x_train, y_train = dataset_to_xy(train_dataset)

    model = GradientBoostingClassifier(
        n_estimators=MODEL_N_ESTIMATORS,
        learning_rate=MODEL_LEARNING_RATE,
        max_depth=MODEL_MAX_DEPTH,
        random_state=seed,
    )

    model.fit(x_train, y_train)

    return model


def predict_windows(model, test_dataset):
    x_test, _ = dataset_to_xy(test_dataset)

    predicted_labels = model.predict(x_test)

    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)

    records = []

    for index, sample in enumerate(test_dataset):
        predicted_label = predicted_labels[index]

        confidence = 0.0

        if probabilities is not None:
            class_index = list(model.classes_).index(predicted_label)
            confidence = float(probabilities[index][class_index])

        records.append(
            {
                "group_id": get_sample_group_id(sample),
                "label": sample["label"],
                "quadrant": get_quadrant(sample),
                "file_name": get_file_name(sample),
                "source_file": get_source_file(sample),
                "window_index": get_window_index(sample),
                "true_label": sample["label"],
                "raw_predicted_label": predicted_label,
                "raw_confidence": confidence,
            }
        )

    records = sorted(
        records,
        key=lambda item: (
            item["group_id"],
            item["window_index"],
        ),
    )

    return records


# ================= TEMPORAL STRATEGIES =================

def choose_majority_label(labels):
    counter = Counter(labels)

    highest_count = max(counter.values())
    candidates = []

    for label in counter:
        if counter[label] == highest_count:
            candidates.append(label)

    for label in reversed(labels):
        if label in candidates:
            return label

    return labels[-1]


def apply_majority_vote(labels, vote_window):
    output_labels = []

    for index in range(len(labels)):
        start = index - vote_window + 1

        if start < 0:
            start = 0

        history = labels[start:index + 1]

        output_labels.append(
            choose_majority_label(history)
        )

    return output_labels


def apply_state_machine(labels, confirmations):
    if not labels:
        return []

    current_state = labels[0]
    candidate_state = None
    candidate_count = 0

    output_labels = []

    for raw_label in labels:
        if raw_label == current_state:
            candidate_state = None
            candidate_count = 0
            output_labels.append(current_state)
            continue

        if raw_label == candidate_state:
            candidate_count += 1
        else:
            candidate_state = raw_label
            candidate_count = 1

        if candidate_count >= confirmations:
            current_state = candidate_state
            candidate_state = None
            candidate_count = 0

        output_labels.append(current_state)

    return output_labels


def apply_temporal_strategy(raw_labels, strategy):
    if strategy["type"] == "raw":
        return raw_labels[:]

    if strategy["type"] == "majority_vote":
        return apply_majority_vote(
            raw_labels,
            strategy["vote_window"],
        )

    if strategy["type"] == "state_machine":
        return apply_state_machine(
            raw_labels,
            strategy["confirmations"],
        )

    raise ValueError(f"Unknown temporal strategy: {strategy['type']}")


def count_state_changes(labels):
    if not labels:
        return 0

    changes = 0
    previous_label = labels[0]

    for label in labels[1:]:
        if label != previous_label:
            changes += 1

        previous_label = label

    return changes


def group_prediction_records(records):
    groups = {}

    for record in records:
        group_id = record["group_id"]

        if group_id not in groups:
            groups[group_id] = []

        groups[group_id].append(record)

    for group_id in groups:
        groups[group_id] = sorted(
            groups[group_id],
            key=lambda item: item["window_index"],
        )

    return groups


def apply_temporal_strategies(raw_records, seed):
    grouped_records = group_prediction_records(raw_records)

    strategy_prediction_rows = []
    file_sequence_rows = []

    for group_id in grouped_records:
        group_records = grouped_records[group_id]

        raw_labels = []

        for record in group_records:
            raw_labels.append(record["raw_predicted_label"])

        true_label = group_records[0]["true_label"]
        quadrant = group_records[0]["quadrant"]
        file_name = group_records[0]["file_name"]
        source_file = group_records[0]["source_file"]

        raw_state_changes = count_state_changes(raw_labels)

        for strategy in TEMPORAL_STRATEGIES:
            strategy_name = strategy["name"]

            final_labels = apply_temporal_strategy(
                raw_labels,
                strategy,
            )

            final_state_changes = count_state_changes(final_labels)

            correct_windows = 0

            for index, final_label in enumerate(final_labels):
                record = group_records[index]

                is_correct = final_label == record["true_label"]

                if is_correct:
                    correct_windows += 1

                strategy_prediction_rows.append(
                    {
                        "seed": seed,
                        "strategy": strategy_name,
                        "group_id": group_id,
                        "label": record["label"],
                        "quadrant": record["quadrant"],
                        "file_name": record["file_name"],
                        "source_file": record["source_file"],
                        "window_index": record["window_index"],
                        "true_label": record["true_label"],
                        "raw_predicted_label": record["raw_predicted_label"],
                        "final_predicted_label": final_label,
                        "raw_confidence": record["raw_confidence"],
                        "correct": is_correct,
                    }
                )

            file_sequence_rows.append(
                {
                    "seed": seed,
                    "strategy": strategy_name,
                    "group_id": group_id,
                    "label": true_label,
                    "quadrant": quadrant,
                    "file_name": file_name,
                    "source_file": source_file,
                    "window_count": len(group_records),
                    "raw_state_changes": raw_state_changes,
                    "final_state_changes": final_state_changes,
                    "correct_windows": correct_windows,
                    "accuracy": safe_division(
                        correct_windows,
                        len(group_records),
                    ),
                }
            )

    return strategy_prediction_rows, file_sequence_rows


# ================= METRICS =================

def build_confusion_matrix(true_labels, predicted_labels, labels):
    label_to_index = {}

    for index, label in enumerate(labels):
        label_to_index[label] = index

    matrix = []

    for _ in labels:
        row = []

        for _ in labels:
            row.append(0)

        matrix.append(row)

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        true_index = label_to_index[true_label]
        predicted_index = label_to_index[predicted_label]

        matrix[true_index][predicted_index] += 1

    return matrix


def compute_metrics(true_labels, predicted_labels):
    matrix = build_confusion_matrix(
        true_labels,
        predicted_labels,
        CLASS_ORDER,
    )

    total_samples = 0
    total_correct = 0
    per_class = []

    for row_index in range(len(CLASS_ORDER)):
        total_correct += matrix[row_index][row_index]

        for value in matrix[row_index]:
            total_samples += value

    for index, label in enumerate(CLASS_ORDER):
        true_positive = matrix[index][index]

        false_positive = 0
        false_negative = 0

        for row_index in range(len(CLASS_ORDER)):
            if row_index != index:
                false_positive += matrix[row_index][index]

        for column_index in range(len(CLASS_ORDER)):
            if column_index != index:
                false_negative += matrix[index][column_index]

        support = sum(matrix[index])

        precision = safe_division(
            true_positive,
            true_positive + false_positive,
        )

        recall = safe_division(
            true_positive,
            true_positive + false_negative,
        )

        f1_score = safe_division(
            2 * precision * recall,
            precision + recall,
        )

        per_class.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "support": support,
            }
        )

    accuracy = safe_division(
        total_correct,
        total_samples,
    )

    macro_f1 = safe_division(
        sum(item["f1_score"] for item in per_class),
        len(per_class),
    )

    weighted_f1 = safe_division(
        sum(item["f1_score"] * item["support"] for item in per_class),
        total_samples,
    )

    result = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    for item in per_class:
        label = item["label"]
        result[f"{label}_f1"] = item["f1_score"]
        result[f"{label}_support"] = item["support"]

    return result


def build_strategy_results_by_seed(strategy_prediction_rows):
    grouped = {}

    for row in strategy_prediction_rows:
        key = (
            row["seed"],
            row["strategy"],
        )

        if key not in grouped:
            grouped[key] = {
                "true_labels": [],
                "predicted_labels": [],
                "state_changes": 0,
                "groups": set(),
            }

        grouped[key]["true_labels"].append(row["true_label"])
        grouped[key]["predicted_labels"].append(row["final_predicted_label"])
        grouped[key]["groups"].add(row["group_id"])

    for row in strategy_prediction_rows:
        key = (
            row["seed"],
            row["strategy"],
        )

    rows = []

    for key in grouped:
        seed, strategy = key

        metrics = compute_metrics(
            grouped[key]["true_labels"],
            grouped[key]["predicted_labels"],
        )

        rows.append(
            {
                "seed": seed,
                "strategy": strategy,
                "test_windows": len(grouped[key]["true_labels"]),
                "test_groups": len(grouped[key]["groups"]),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "empty_f1": metrics["empty_f1"],
                "static_presence_f1": metrics["static_presence_f1"],
                "movement_f1": metrics["movement_f1"],
            }
        )

    rows = sorted(
        rows,
        key=lambda item: (
            item["strategy"],
            item["seed"],
        ),
    )

    return rows


def summarize_strategy_results(strategy_results_by_seed, file_sequence_rows):
    rows_by_strategy = {}

    for row in strategy_results_by_seed:
        strategy = row["strategy"]

        if strategy not in rows_by_strategy:
            rows_by_strategy[strategy] = []

        rows_by_strategy[strategy].append(row)

    state_changes_by_strategy = {}

    for row in file_sequence_rows:
        strategy = row["strategy"]

        if strategy not in state_changes_by_strategy:
            state_changes_by_strategy[strategy] = []

        state_changes_by_strategy[strategy].append(
            int(row["final_state_changes"])
        )

    summary_rows = []

    for strategy in rows_by_strategy:
        rows = rows_by_strategy[strategy]

        accuracy_values = []
        macro_f1_values = []
        weighted_f1_values = []
        movement_f1_values = []
        empty_f1_values = []
        static_f1_values = []

        for row in rows:
            accuracy_values.append(float(row["accuracy"]))
            macro_f1_values.append(float(row["macro_f1"]))
            weighted_f1_values.append(float(row["weighted_f1"]))
            movement_f1_values.append(float(row["movement_f1"]))
            empty_f1_values.append(float(row["empty_f1"]))
            static_f1_values.append(float(row["static_presence_f1"]))

        state_changes = state_changes_by_strategy.get(strategy, [])

        summary_rows.append(
            {
                "strategy": strategy,
                "seeds": len(rows),
                "accuracy_mean": mean(accuracy_values),
                "accuracy_std": std(accuracy_values),
                "macro_f1_mean": mean(macro_f1_values),
                "macro_f1_std": std(macro_f1_values),
                "weighted_f1_mean": mean(weighted_f1_values),
                "movement_f1_mean": mean(movement_f1_values),
                "empty_f1_mean": mean(empty_f1_values),
                "static_presence_f1_mean": mean(static_f1_values),
                "state_changes_mean_per_file": mean(state_changes),
                "state_changes_std_per_file": std(state_changes),
            }
        )

    summary_rows = sorted(
        summary_rows,
        key=lambda item: (
            item["macro_f1_mean"],
            item["accuracy_mean"],
            item["movement_f1_mean"],
        ),
        reverse=True,
    )

    return summary_rows


# ================= OUTPUT =================

def get_experiment_paths():
    run_dir = RESULTS_DIR / "runs" / EXPERIMENT_FOLDER_NAME
    tables_dir = run_dir / "tables"
    reports_dir = run_dir / "reports"

    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "tables_dir": tables_dir,
        "reports_dir": reports_dir,
        "strategy_results_by_seed": tables_dir / "temporal_results_by_seed.csv",
        "strategy_summary": tables_dir / "temporal_strategy_summary.csv",
        "window_predictions": tables_dir / "temporal_window_predictions.csv",
        "file_sequence_summary": tables_dir / "temporal_file_sequence_summary.csv",
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
    index_path = RESULTS_DIR / "experiment_index.csv"

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
        "description": "Realtime temporal decision simulation with sliding-window predictions",
        "model": "gradient_boosting_depth3_20 + temporal decision layer",
        "validation_method": f"Repeated file-based holdout across {best_summary['seeds']} seeds",
        "top_k": MODEL_TOP_K,
        "max_depth": MODEL_MAX_DEPTH,
        "min_samples_split": "",
        "accuracy": f"{best_summary['accuracy_mean']:.6f}",
        "macro_f1": f"{best_summary['macro_f1_mean']:.6f}",
        "weighted_f1": f"{best_summary['weighted_f1_mean']:.6f}",
        "main_observation": (
            f"Best temporal strategy: {best_summary['strategy']}. "
            "Current dataset has very short file sequences, so this is mainly "
            "an architecture simulation before Dataset v2."
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
    groups,
    strategy_summary,
):
    window_counts = []

    for group in groups:
        window_counts.append(group["sample_count"])

    best = strategy_summary[0]

    lines = []

    lines.append("Experiment 020 - Realtime temporal decision simulation")
    lines.append("")
    lines.append("Objective:")
    lines.append(
        "Simulate the decision layer that will be used in realtime after "
        "per-window CSI classification."
    )
    lines.append("")
    lines.append("Model:")
    lines.append(f"- Gradient Boosting, n_estimators={MODEL_N_ESTIMATORS}")
    lines.append(f"- max_depth={MODEL_MAX_DEPTH}")
    lines.append(f"- top_k={MODEL_TOP_K}")
    lines.append("")
    lines.append("Temporal strategies:")
    for strategy in TEMPORAL_STRATEGIES:
        lines.append(f"- {strategy['name']}")
    lines.append("")
    lines.append("Dataset:")
    lines.append(f"- Total samples/windows: {len(feature_dataset)}")
    lines.append(f"- Features per sample: {len(feature_dataset[0]['features'])}")
    lines.append(f"- Total file groups: {len(groups)}")
    lines.append(f"- Minimum windows per file: {min(window_counts)}")
    lines.append(f"- Maximum windows per file: {max(window_counts)}")
    lines.append(f"- Mean windows per file: {mean(window_counts):.6f}")
    lines.append("")
    lines.append("Best temporal strategy:")
    lines.append(f"- strategy: {best['strategy']}")
    lines.append(f"- accuracy mean: {best['accuracy_mean']:.6f}")
    lines.append(f"- macro F1 mean: {best['macro_f1_mean']:.6f}")
    lines.append(f"- weighted F1 mean: {best['weighted_f1_mean']:.6f}")
    lines.append(f"- movement F1 mean: {best['movement_f1_mean']:.6f}")
    lines.append(f"- mean state changes per file: {best['state_changes_mean_per_file']:.6f}")
    lines.append("")
    lines.append("Important note:")
    lines.append(
        "The current dataset has very short file sequences. Therefore, "
        "majority voting and state-machine behavior cannot be fully evaluated "
        "yet. This experiment is mainly used to establish the realtime "
        "simulation structure before Dataset v2."
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


# ================= MAIN =================

def run_experiment():
    feature_dataset = load_pickle(FEATURE_DATASET_FILE)

    if not feature_dataset:
        raise ValueError("Feature dataset is empty.")

    features_per_sample = len(feature_dataset[0]["features"])

    if features_per_sample != EXPECTED_FEATURES_PER_SAMPLE:
        raise ValueError(
            "This experiment must be run with the official 014 feature set. "
            f"Expected {EXPECTED_FEATURES_PER_SAMPLE} features per sample, "
            f"but found {features_per_sample}."
        )

    groups = group_samples_by_file(feature_dataset)

    print()
    print("Realtime Temporal Decision Simulation")
    print("=" * 70)
    print("Experiment:", EXPERIMENT_ID)
    print("Feature dataset:", FEATURE_DATASET_FILE)
    print("Samples/windows:", len(feature_dataset))
    print("File groups:", len(groups))
    print("Seeds:", SEEDS)
    print()

    all_strategy_prediction_rows = []
    all_file_sequence_rows = []

    for seed in SEEDS:
        print()
        print("=" * 70)
        print(f"Seed {seed}")
        print("=" * 70)

        train_dataset, test_dataset, train_groups, test_groups = file_stratified_holdout_split(
            feature_dataset,
            TEST_SIZE,
            seed,
        )

        selected_train_dataset, selected_test_dataset, selected_indices = select_top_k_from_training(
            train_dataset,
            test_dataset,
            MODEL_TOP_K,
        )

        model = train_gradient_boosting(
            selected_train_dataset,
            seed,
        )

        raw_records = predict_windows(
            model,
            selected_test_dataset,
        )

        strategy_prediction_rows, file_sequence_rows = apply_temporal_strategies(
            raw_records,
            seed,
        )

        all_strategy_prediction_rows.extend(strategy_prediction_rows)
        all_file_sequence_rows.extend(file_sequence_rows)

        temporary_results = build_strategy_results_by_seed(
            strategy_prediction_rows,
        )

        for row in temporary_results:
            print(
                f"{row['strategy']:<30} "
                f"accuracy={row['accuracy']:.6f} "
                f"macro_f1={row['macro_f1']:.6f} "
                f"movement_f1={row['movement_f1']:.6f}"
            )

    strategy_results_by_seed = build_strategy_results_by_seed(
        all_strategy_prediction_rows,
    )

    strategy_summary = summarize_strategy_results(
        strategy_results_by_seed,
        all_file_sequence_rows,
    )

    paths = get_experiment_paths()

    save_csv(
        strategy_results_by_seed,
        paths["strategy_results_by_seed"],
    )

    save_csv(
        strategy_summary,
        paths["strategy_summary"],
    )

    save_csv(
        all_strategy_prediction_rows,
        paths["window_predictions"],
    )

    save_csv(
        all_file_sequence_rows,
        paths["file_sequence_summary"],
    )

    save_experiment_notes(
        paths["experiment_notes"],
        feature_dataset,
        groups,
        strategy_summary,
    )

    update_experiment_index(
        strategy_summary[0],
    )

    print()
    print("=" * 70)
    print("Realtime temporal decision simulation finished.")
    print("=" * 70)
    print("Best temporal strategy:")
    print(f"strategy: {strategy_summary[0]['strategy']}")
    print(f"accuracy_mean: {strategy_summary[0]['accuracy_mean']:.6f}")
    print(f"macro_f1_mean: {strategy_summary[0]['macro_f1_mean']:.6f}")
    print(f"movement_f1_mean: {strategy_summary[0]['movement_f1_mean']:.6f}")
    print(f"state_changes_mean_per_file: {strategy_summary[0]['state_changes_mean_per_file']:.6f}")
    print()
    print("Results saved to:")
    print(paths["run_dir"])


def main():
    run_experiment()


if __name__ == "__main__":
    main()