"""
Scenario-based Realtime Stream Simulation

Experiment 021

This script builds artificial continuous CSI streams from the current
window-based dataset.

Purpose:

    Simulate realtime scenarios such as:

        empty -> static_presence -> movement -> static_presence -> empty

    and evaluate temporal decision strategies:

        - raw prediction
        - rolling majority vote
        - state machine / hysteresis
        - majority vote followed by state machine

Important:

    This experiment is used for realtime architecture development only.
    Since the current dataset contains very short files, scenario streams
    are artificially assembled from files of the same class/quadrant.
    Results should not be treated as final scientific validation.
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
        "scikit-learn is required for experiment 021. "
        "Install it with: pip install scikit-learn"
    ) from error


CLASS_ORDER = [
    "empty",
    "static_presence",
    "movement",
]

EXPERIMENT_ID = "021"
EXPERIMENT_FOLDER_NAME = "021_scenario_stream_state_machine"

EXPECTED_FEATURES_PER_SAMPLE = 231

TEST_SIZE = 0.20

MODEL_TOP_K = 231
MODEL_N_ESTIMATORS = 20
MODEL_MAX_DEPTH = 3
MODEL_LEARNING_RATE = 0.10

SEGMENT_SEQUENCE = [
    "empty",
    "static_presence",
    "movement",
    "static_presence",
    "empty",
]

TARGET_WINDOWS_PER_SEGMENT = 12
SCENARIOS_PER_QUADRANT_PER_SEED = 3

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
        "name": "majority_vote_7",
        "type": "majority_vote",
        "vote_window": 7,
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
    {
        "name": "vote5_state_machine_confirm_2",
        "type": "vote_then_state_machine",
        "vote_window": 5,
        "confirmations": 2,
    },
    {
        "name": "vote5_state_machine_confirm_3",
        "type": "vote_then_state_machine",
        "vote_window": 5,
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

    average = mean(values)
    total = 0.0

    for value in values:
        difference = value - average
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


def flatten_groups(groups):
    dataset = []

    for group in groups:
        dataset.extend(group["samples"])

    return dataset


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

    train_dataset = flatten_groups(train_groups)
    test_dataset = flatten_groups(test_groups)

    random_generator.shuffle(train_dataset)

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


def select_top_k_from_training(train_dataset, top_k):
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

    return selected_train_dataset, selected_indices


def transform_sample_features(sample, selected_indices):
    selected_features = []

    for index in selected_indices:
        selected_features.append(sample["features"][index])

    return selected_features


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


def predict_stream_windows(model, scenario_windows, selected_indices):
    x_values = []

    for item in scenario_windows:
        sample = item["sample"]
        x_values.append(
            transform_sample_features(sample, selected_indices)
        )

    predicted_labels = model.predict(x_values)

    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_values)

    records = []

    for index, item in enumerate(scenario_windows):
        predicted_label = predicted_labels[index]
        confidence = 0.0

        if probabilities is not None:
            class_index = list(model.classes_).index(predicted_label)
            confidence = float(probabilities[index][class_index])

        records.append(
            {
                "seed": item["seed"],
                "scenario_id": item["scenario_id"],
                "quadrant": item["quadrant"],
                "segment_index": item["segment_index"],
                "segment_label": item["segment_label"],
                "window_in_segment": item["window_in_segment"],
                "stream_window_index": item["stream_window_index"],
                "source_group_id": item["source_group_id"],
                "source_file": item["source_file"],
                "true_label": item["segment_label"],
                "raw_predicted_label": predicted_label,
                "raw_confidence": confidence,
            }
        )

    return records


# ================= SCENARIO GENERATION =================

def build_group_pool_by_quadrant_label(groups):
    pool = {}

    for group in groups:
        key = (
            group["quadrant"],
            group["label"],
        )

        if key not in pool:
            pool[key] = []

        pool[key].append(group)

    return pool


def get_available_quadrants_for_scenarios(pool):
    quadrants = []

    for key in pool:
        quadrant, _ = key

        if quadrant not in quadrants:
            quadrants.append(quadrant)

    quadrants.sort()

    valid_quadrants = []

    for quadrant in quadrants:
        has_all_labels = True

        for label in CLASS_ORDER:
            if (quadrant, label) not in pool:
                has_all_labels = False

        if has_all_labels:
            valid_quadrants.append(quadrant)

    return valid_quadrants


def collect_segment_windows(
    groups,
    target_window_count,
    random_generator,
):
    selected_windows = []

    if not groups:
        return selected_windows

    shuffled_groups = groups[:]
    random_generator.shuffle(shuffled_groups)

    group_index = 0

    while len(selected_windows) < target_window_count:
        group = shuffled_groups[group_index]

        for sample in group["samples"]:
            selected_windows.append(
                {
                    "sample": sample,
                    "source_group_id": group["group_id"],
                    "source_file": group["source_file"],
                }
            )

            if len(selected_windows) >= target_window_count:
                break

        group_index += 1

        if group_index >= len(shuffled_groups):
            group_index = 0
            random_generator.shuffle(shuffled_groups)

    return selected_windows


def build_scenarios_from_test_groups(
    test_groups,
    seed,
):
    random_generator = random.Random(seed)

    pool = build_group_pool_by_quadrant_label(test_groups)
    valid_quadrants = get_available_quadrants_for_scenarios(pool)

    scenarios = []
    scenario_counter = 0

    for quadrant in valid_quadrants:
        for repetition in range(SCENARIOS_PER_QUADRANT_PER_SEED):
            scenario_id = f"seed{seed}_{quadrant}_scenario{repetition + 1}"

            scenario_windows = []
            stream_window_index = 0

            for segment_index, segment_label in enumerate(SEGMENT_SEQUENCE):
                groups = pool[(quadrant, segment_label)]

                segment_windows = collect_segment_windows(
                    groups,
                    TARGET_WINDOWS_PER_SEGMENT,
                    random_generator,
                )

                for window_in_segment, item in enumerate(segment_windows):
                    scenario_windows.append(
                        {
                            "seed": seed,
                            "scenario_id": scenario_id,
                            "quadrant": quadrant,
                            "segment_index": segment_index,
                            "segment_label": segment_label,
                            "window_in_segment": window_in_segment,
                            "stream_window_index": stream_window_index,
                            "source_group_id": item["source_group_id"],
                            "source_file": item["source_file"],
                            "sample": item["sample"],
                        }
                    )

                    stream_window_index += 1

            if scenario_windows:
                scenario_counter += 1
                scenarios.append(
                    {
                        "scenario_id": scenario_id,
                        "seed": seed,
                        "quadrant": quadrant,
                        "windows": scenario_windows,
                    }
                )

    return scenarios


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

    if strategy["type"] == "vote_then_state_machine":
        voted_labels = apply_majority_vote(
            raw_labels,
            strategy["vote_window"],
        )

        return apply_state_machine(
            voted_labels,
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


def get_transition_boundaries(true_labels):
    boundaries = []

    previous_label = true_labels[0]

    for index in range(1, len(true_labels)):
        current_label = true_labels[index]

        if current_label != previous_label:
            boundaries.append(
                {
                    "index": index,
                    "previous_label": previous_label,
                    "new_label": current_label,
                }
            )

        previous_label = current_label

    return boundaries


def compute_transition_delay(true_labels, predicted_labels):
    boundaries = get_transition_boundaries(true_labels)

    if not boundaries:
        return {
            "transition_count": 0,
            "transition_success_count": 0,
            "transition_success_rate": 0.0,
            "transition_delay_mean": 0.0,
        }

    delays = []
    success_count = 0

    for boundary in boundaries:
        start_index = boundary["index"]
        new_label = boundary["new_label"]

        end_index = len(true_labels)

        for next_boundary in boundaries:
            if next_boundary["index"] > start_index:
                end_index = next_boundary["index"]
                break

        found = False

        for index in range(start_index, end_index):
            if predicted_labels[index] == new_label:
                delays.append(index - start_index)
                success_count += 1
                found = True
                break

        if not found:
            delays.append(end_index - start_index)

    return {
        "transition_count": len(boundaries),
        "transition_success_count": success_count,
        "transition_success_rate": safe_division(success_count, len(boundaries)),
        "transition_delay_mean": mean(delays),
    }


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


def evaluate_scenario_strategy(raw_records, strategy):
    raw_labels = []
    true_labels = []

    for record in raw_records:
        raw_labels.append(record["raw_predicted_label"])
        true_labels.append(record["true_label"])

    final_labels = apply_temporal_strategy(
        raw_labels,
        strategy,
    )

    metrics = compute_metrics(
        true_labels,
        final_labels,
    )

    state_changes = count_state_changes(final_labels)
    true_state_changes = count_state_changes(true_labels)
    extra_state_changes = max(0, state_changes - true_state_changes)
    missing_state_changes = max(0, true_state_changes - state_changes)

    transition_info = compute_transition_delay(
        true_labels,
        final_labels,
    )

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
            }
        )

    summary_rows = sorted(
        summary_rows,
        key=lambda item: (
            item["macro_f1_mean"],
            item["accuracy_mean"],
            item["movement_f1_mean"],
            -item["extra_state_changes_mean"],
            -item["transition_delay_mean"],
        ),
        reverse=True,
    )

    return summary_rows


# ================= OUTPUT =================

def get_experiment_paths():
    run_dir = RESULTS_DIR / "runs" / EXPERIMENT_FOLDER_NAME
    tables_dir = run_dir / "tables"

    tables_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "tables_dir": tables_dir,

        "scenario_results": tables_dir / "scenario_results.csv",
        "strategy_summary": tables_dir / "scenario_strategy_summary.csv",
        "window_predictions": tables_dir / "scenario_window_predictions.csv",
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
        "description": "Scenario-based realtime stream simulation with state-machine strategies",
        "model": "gradient_boosting_depth3_20 + temporal state machine",
        "validation_method": "Artificial scenario streams assembled from file-based holdout test groups",
        "top_k": MODEL_TOP_K,
        "max_depth": MODEL_MAX_DEPTH,
        "min_samples_split": "",
        "accuracy": f"{best_summary['accuracy_mean']:.6f}",
        "macro_f1": f"{best_summary['macro_f1_mean']:.6f}",
        "weighted_f1": f"{best_summary['weighted_f1_mean']:.6f}",
        "main_observation": (
            f"Best scenario strategy: {best_summary['strategy']}. "
            "This is an architecture simulation, not final validation."
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

    lines.append("Experiment 021 - Scenario-based realtime stream simulation")
    lines.append("")
    lines.append("Objective:")
    lines.append(
        "Build artificial continuous streams to evaluate how temporal "
        "decision strategies behave during class transitions."
    )
    lines.append("")
    lines.append("Scenario sequence:")
    lines.append(" -> ".join(SEGMENT_SEQUENCE))
    lines.append("")
    lines.append("Model:")
    lines.append(f"- Gradient Boosting, n_estimators={MODEL_N_ESTIMATORS}")
    lines.append(f"- max_depth={MODEL_MAX_DEPTH}")
    lines.append(f"- top_k={MODEL_TOP_K}")
    lines.append("")
    lines.append("Dataset:")
    lines.append(f"- Total samples/windows: {len(feature_dataset)}")
    lines.append(f"- Features per sample: {len(feature_dataset[0]['features'])}")
    lines.append("")
    lines.append("Simulation parameters:")
    lines.append(f"- Target windows per segment: {TARGET_WINDOWS_PER_SEGMENT}")
    lines.append(f"- Scenarios per quadrant per seed: {SCENARIOS_PER_QUADRANT_PER_SEED}")
    lines.append(f"- Generated scenario evaluations: {len(scenario_results)}")
    lines.append("")
    lines.append("Best strategy:")
    lines.append(f"- strategy: {best['strategy']}")
    lines.append(f"- accuracy mean: {best['accuracy_mean']:.6f}")
    lines.append(f"- macro F1 mean: {best['macro_f1_mean']:.6f}")
    lines.append(f"- movement F1 mean: {best['movement_f1_mean']:.6f}")
    lines.append(f"- final state changes mean: {best['final_state_changes_mean']:.6f}")
    lines.append(f"- extra state changes mean: {best['extra_state_changes_mean']:.6f}")
    lines.append(f"- missing state changes mean: {best['missing_state_changes_mean']:.6f}")
    lines.append(f"- transition success rate mean: {best['transition_success_rate_mean']:.6f}")
    lines.append(f"- transition delay mean: {best['transition_delay_mean']:.6f}")
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

    print()
    print("Scenario-based Realtime Stream Simulation")
    print("=" * 70)
    print("Experiment:", EXPERIMENT_ID)
    print("Feature dataset:", FEATURE_DATASET_FILE)
    print("Samples/windows:", len(feature_dataset))
    print("Seeds:", SEEDS)
    print()

    all_scenario_results = []
    all_window_prediction_rows = []

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

        selected_train_dataset, selected_indices = select_top_k_from_training(
            train_dataset,
            MODEL_TOP_K,
        )

        model = train_gradient_boosting(
            selected_train_dataset,
            seed,
        )

        scenarios = build_scenarios_from_test_groups(
            test_groups,
            seed,
        )

        print(f"Generated scenarios: {len(scenarios)}")

        for scenario in scenarios:
            raw_records = predict_stream_windows(
                model,
                scenario["windows"],
                selected_indices,
            )

            for strategy in TEMPORAL_STRATEGIES:
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
    print("Scenario stream simulation finished.")
    print("=" * 70)
    print("Best strategy:")
    print(f"strategy: {strategy_summary[0]['strategy']}")
    print(f"accuracy_mean: {strategy_summary[0]['accuracy_mean']:.6f}")
    print(f"macro_f1_mean: {strategy_summary[0]['macro_f1_mean']:.6f}")
    print(f"movement_f1_mean: {strategy_summary[0]['movement_f1_mean']:.6f}")
    print(f"transition_delay_mean: {strategy_summary[0]['transition_delay_mean']:.6f}")
    print()
    print("Results saved to:")
    print(paths["run_dir"])


def main():
    run_experiment()


if __name__ == "__main__":
    main()