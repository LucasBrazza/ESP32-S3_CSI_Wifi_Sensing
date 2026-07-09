"""
Dataset Split and Data Quality Diagnostic

Experiment 019

This script diagnoses whether the current dataset and file-based splits
are causing unstable model performance.

Purpose:

    Identify whether performance instability is caused by:
    - specific files
    - specific quadrants
    - class imbalance
    - short files with few generated windows
    - weak cross-quadrant generalization

The model used here is the best stable candidate from Experiment 018:

    GradientBoostingClassifier
    n_estimators = 20
    max_depth = 3
    top_k = 231

This experiment is diagnostic. It is not intended to tune the final model.
"""

import csv
import random
from collections import Counter

from Tools.common.config import MIN_SAMPLES_SPLIT
from Tools.common.io_utils import load_pickle
from Tools.common.project_paths import FEATURE_DATASET_FILE, RESULTS_DIR
from Tools.preprocessing.feature_selection import rank_features_by_fisher_score


try:
    from sklearn.ensemble import GradientBoostingClassifier
except ImportError as error:
    raise ImportError(
        "scikit-learn is required for experiment 019. "
        "Install it with: pip install scikit-learn"
    ) from error


CLASS_ORDER = [
    "empty",
    "static_presence",
    "movement",
]

EXPERIMENT_ID = "019"
EXPERIMENT_FOLDER_NAME = "019_dataset_split_quality_diagnostic"

TEST_SIZE = 0.20
EXPECTED_FEATURES_PER_SAMPLE = 231

MODEL_TOP_K = 231
MODEL_N_ESTIMATORS = 20
MODEL_MAX_DEPTH = 3
MODEL_LEARNING_RATE = 0.10

LOW_WINDOW_COUNT_THRESHOLD = 3

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


# ================= GROUPING =================

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


def get_unique_quadrants(groups):
    quadrants = []

    for group in groups:
        quadrant = group["quadrant"]

        if quadrant not in quadrants:
            quadrants.append(quadrant)

    quadrants.sort()

    return quadrants


def count_groups_by_label(groups):
    counter = Counter()

    for group in groups:
        counter[group["label"]] += 1

    return counter


def count_groups_by_quadrant(groups):
    counter = Counter()

    for group in groups:
        counter[group["quadrant"]] += 1

    return counter


def count_samples_by_label(dataset):
    counter = Counter()

    for sample in dataset:
        counter[sample["label"]] += 1

    return counter


def count_samples_by_quadrant(dataset):
    counter = Counter()

    for sample in dataset:
        counter[get_quadrant(sample)] += 1

    return counter


def count_samples_by_label_quadrant(dataset):
    counter = Counter()

    for sample in dataset:
        key = (
            sample["label"],
            get_quadrant(sample),
        )

        counter[key] += 1

    return counter


# ================= SPLITTING =================

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
    random_generator.shuffle(test_dataset)

    return train_dataset, test_dataset, train_groups, test_groups


def quadrant_holdout_split(dataset, test_quadrant):
    groups = group_samples_by_file(dataset)

    train_dataset = []
    test_dataset = []
    train_groups = []
    test_groups = []

    for group in groups:
        if group["quadrant"] == test_quadrant:
            test_groups.append(group)
            test_dataset.extend(group["samples"])
        else:
            train_groups.append(group)
            train_dataset.extend(group["samples"])

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


def compute_metrics(true_labels, predicted_labels, labels):
    matrix = build_confusion_matrix(
        true_labels,
        predicted_labels,
        labels,
    )

    per_class = []

    total_samples = 0
    total_correct = 0

    for row_index in range(len(labels)):
        total_correct += matrix[row_index][row_index]

        for value in matrix[row_index]:
            total_samples += value

    for index, label in enumerate(labels):
        true_positive = matrix[index][index]

        false_positive = 0
        false_negative = 0

        for row_index in range(len(labels)):
            if row_index != index:
                false_positive += matrix[row_index][index]

        for column_index in range(len(labels)):
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

    supported_classes = []

    for item in per_class:
        if item["support"] > 0:
            supported_classes.append(item)

    macro_f1_all_classes = safe_division(
        sum(item["f1_score"] for item in per_class),
        len(per_class),
    )

    macro_f1_supported_classes = safe_division(
        sum(item["f1_score"] for item in supported_classes),
        len(supported_classes),
    )

    weighted_f1 = safe_division(
        sum(item["f1_score"] * item["support"] for item in per_class),
        total_samples,
    )

    result = {
        "accuracy": accuracy,
        "macro_f1": macro_f1_all_classes,
        "macro_f1_supported": macro_f1_supported_classes,
        "weighted_f1": weighted_f1,
    }

    for item in per_class:
        label = item["label"]

        result[f"{label}_f1"] = item["f1_score"]
        result[f"{label}_support"] = item["support"]

    return result


# ================= MODEL =================

def train_and_predict_gradient_boosting(train_dataset, test_dataset, seed):
    x_train, y_train = dataset_to_xy(train_dataset)
    x_test, _ = dataset_to_xy(test_dataset)

    model = GradientBoostingClassifier(
        n_estimators=MODEL_N_ESTIMATORS,
        learning_rate=MODEL_LEARNING_RATE,
        max_depth=MODEL_MAX_DEPTH,
        random_state=seed,
    )

    model.fit(x_train, y_train)

    predicted_labels = model.predict(x_test)

    return predicted_labels


def evaluate_split(train_dataset, test_dataset, seed):
    selected_train_dataset, selected_test_dataset, selected_indices = select_top_k_from_training(
        train_dataset,
        test_dataset,
        MODEL_TOP_K,
    )

    predicted_labels = train_and_predict_gradient_boosting(
        selected_train_dataset,
        selected_test_dataset,
        seed,
    )

    true_labels = []

    for sample in selected_test_dataset:
        true_labels.append(sample["label"])

    metrics = compute_metrics(
        true_labels,
        predicted_labels,
        CLASS_ORDER,
    )

    predictions = []

    for sample, predicted_label in zip(selected_test_dataset, predicted_labels):
        predictions.append(
            {
                "group_id": get_sample_group_id(sample),
                "label": sample["label"],
                "quadrant": get_quadrant(sample),
                "file_name": get_file_name(sample),
                "source_file": get_source_file(sample),
                "window_index": sample.get("window_index", ""),
                "true_label": sample["label"],
                "predicted_label": predicted_label,
                "correct": sample["label"] == predicted_label,
            }
        )

    return metrics, predictions, selected_indices


# ================= OUTPUT PATHS =================

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

        "dataset_file_groups": tables_dir / "dataset_file_groups.csv",
        "split_quality_by_seed": tables_dir / "split_quality_by_seed.csv",
        "file_test_performance": tables_dir / "file_test_performance_summary.csv",
        "quadrant_performance": tables_dir / "quadrant_performance_summary.csv",
        "class_quadrant_performance": tables_dir / "class_quadrant_performance_summary.csv",
        "cross_quadrant_holdout": tables_dir / "cross_quadrant_holdout_results.csv",

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


# ================= DIAGNOSTIC TABLES =================

def build_file_group_rows(groups):
    rows = []

    for group in groups:
        rows.append(
            {
                "group_id": group["group_id"],
                "label": group["label"],
                "quadrant": group["quadrant"],
                "file_name": group["file_name"],
                "source_file": group["source_file"],
                "window_count": group["sample_count"],
                "low_window_count": group["sample_count"] < LOW_WINDOW_COUNT_THRESHOLD,
            }
        )

    rows = sorted(
        rows,
        key=lambda item: (
            item["label"],
            item["quadrant"],
            item["window_count"],
            item["group_id"],
        ),
    )

    return rows


def build_split_quality_row(
    seed,
    train_dataset,
    test_dataset,
    train_groups,
    test_groups,
    metrics,
    quadrants,
):
    train_label_count = count_samples_by_label(train_dataset)
    test_label_count = count_samples_by_label(test_dataset)

    train_quadrant_count = count_samples_by_quadrant(train_dataset)
    test_quadrant_count = count_samples_by_quadrant(test_dataset)

    train_group_label_count = count_groups_by_label(train_groups)
    test_group_label_count = count_groups_by_label(test_groups)

    train_group_quadrant_count = count_groups_by_quadrant(train_groups)
    test_group_quadrant_count = count_groups_by_quadrant(test_groups)

    test_sample_counts = []

    for label in CLASS_ORDER:
        test_sample_counts.append(test_label_count[label])

    minimum_test_class_samples = min(test_sample_counts)

    row = {
        "seed": seed,

        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "train_groups": len(train_groups),
        "test_groups": len(test_groups),

        "minimum_test_class_samples": minimum_test_class_samples,

        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "empty_f1": metrics.get("empty_f1", 0.0),
        "static_presence_f1": metrics.get("static_presence_f1", 0.0),
        "movement_f1": metrics.get("movement_f1", 0.0),
    }

    for label in CLASS_ORDER:
        row[f"train_samples_{label}"] = train_label_count[label]
        row[f"test_samples_{label}"] = test_label_count[label]
        row[f"train_groups_{label}"] = train_group_label_count[label]
        row[f"test_groups_{label}"] = test_group_label_count[label]

    for quadrant in quadrants:
        row[f"train_samples_{quadrant}"] = train_quadrant_count[quadrant]
        row[f"test_samples_{quadrant}"] = test_quadrant_count[quadrant]
        row[f"train_groups_{quadrant}"] = train_group_quadrant_count[quadrant]
        row[f"test_groups_{quadrant}"] = test_group_quadrant_count[quadrant]

    return row


def summarize_file_test_performance(all_predictions, file_group_rows):
    file_info = {}

    for row in file_group_rows:
        file_info[row["group_id"]] = row

    performance = {}

    for prediction in all_predictions:
        group_id = prediction["group_id"]

        if group_id not in performance:
            performance[group_id] = {
                "group_id": group_id,
                "test_appearances": 0,
                "tested_windows": 0,
                "correct_windows": 0,
                "predicted_empty": 0,
                "predicted_static_presence": 0,
                "predicted_movement": 0,
            }

        performance[group_id]["tested_windows"] += 1

        if prediction["correct"]:
            performance[group_id]["correct_windows"] += 1

        predicted_label = prediction["predicted_label"]

        if predicted_label == "empty":
            performance[group_id]["predicted_empty"] += 1

        if predicted_label == "static_presence":
            performance[group_id]["predicted_static_presence"] += 1

        if predicted_label == "movement":
            performance[group_id]["predicted_movement"] += 1

    groups_seen_per_seed = set()

    for prediction in all_predictions:
        key = (
            prediction["seed"],
            prediction["group_id"],
        )

        groups_seen_per_seed.add(key)

    for seed, group_id in groups_seen_per_seed:
        performance[group_id]["test_appearances"] += 1

    rows = []

    for group_id in performance:
        item = performance[group_id]
        info = file_info.get(group_id, {})

        tested_windows = item["tested_windows"]
        correct_windows = item["correct_windows"]

        rows.append(
            {
                "group_id": group_id,
                "label": info.get("label", ""),
                "quadrant": info.get("quadrant", ""),
                "file_name": info.get("file_name", ""),
                "source_file": info.get("source_file", ""),
                "original_window_count": info.get("window_count", ""),
                "low_window_count": info.get("low_window_count", ""),
                "test_appearances": item["test_appearances"],
                "tested_windows": tested_windows,
                "correct_windows": correct_windows,
                "accuracy": safe_division(correct_windows, tested_windows),
                "predicted_empty": item["predicted_empty"],
                "predicted_static_presence": item["predicted_static_presence"],
                "predicted_movement": item["predicted_movement"],
            }
        )

    rows = sorted(
        rows,
        key=lambda item: (
            item["accuracy"],
            -item["tested_windows"],
            item["label"],
            item["quadrant"],
            item["group_id"],
        ),
    )

    return rows


def summarize_quadrant_performance(all_predictions):
    aggregate = {}

    for prediction in all_predictions:
        quadrant = prediction["quadrant"]

        if quadrant not in aggregate:
            aggregate[quadrant] = {
                "quadrant": quadrant,
                "tested_windows": 0,
                "correct_windows": 0,
            }

        aggregate[quadrant]["tested_windows"] += 1

        if prediction["correct"]:
            aggregate[quadrant]["correct_windows"] += 1

    rows = []

    for quadrant in aggregate:
        item = aggregate[quadrant]

        rows.append(
            {
                "quadrant": quadrant,
                "tested_windows": item["tested_windows"],
                "correct_windows": item["correct_windows"],
                "accuracy": safe_division(
                    item["correct_windows"],
                    item["tested_windows"],
                ),
            }
        )

    rows = sorted(
        rows,
        key=lambda item: (
            item["accuracy"],
            item["quadrant"],
        ),
    )

    return rows


def summarize_class_quadrant_performance(all_predictions):
    aggregate = {}

    for prediction in all_predictions:
        key = (
            prediction["label"],
            prediction["quadrant"],
        )

        if key not in aggregate:
            aggregate[key] = {
                "label": prediction["label"],
                "quadrant": prediction["quadrant"],
                "tested_windows": 0,
                "correct_windows": 0,
            }

        aggregate[key]["tested_windows"] += 1

        if prediction["correct"]:
            aggregate[key]["correct_windows"] += 1

    rows = []

    for key in aggregate:
        item = aggregate[key]

        rows.append(
            {
                "label": item["label"],
                "quadrant": item["quadrant"],
                "tested_windows": item["tested_windows"],
                "correct_windows": item["correct_windows"],
                "accuracy": safe_division(
                    item["correct_windows"],
                    item["tested_windows"],
                ),
            }
        )

    rows = sorted(
        rows,
        key=lambda item: (
            item["accuracy"],
            item["label"],
            item["quadrant"],
        ),
    )

    return rows


def run_cross_quadrant_holdout(feature_dataset, quadrants):
    rows = []

    for quadrant in quadrants:
        train_dataset, test_dataset, train_groups, test_groups = quadrant_holdout_split(
            feature_dataset,
            quadrant,
        )

        if not train_dataset or not test_dataset:
            continue

        metrics, predictions, _ = evaluate_split(
            train_dataset,
            test_dataset,
            seed=42,
        )

        train_label_count = count_samples_by_label(train_dataset)
        test_label_count = count_samples_by_label(test_dataset)

        row = {
            "test_quadrant": quadrant,
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),

            "accuracy": metrics["accuracy"],
            "macro_f1_all_classes": metrics["macro_f1"],
            "macro_f1_supported_classes": metrics["macro_f1_supported"],
            "weighted_f1": metrics["weighted_f1"],

            "empty_f1": metrics.get("empty_f1", 0.0),
            "static_presence_f1": metrics.get("static_presence_f1", 0.0),
            "movement_f1": metrics.get("movement_f1", 0.0),
        }

        for label in CLASS_ORDER:
            row[f"train_samples_{label}"] = train_label_count[label]
            row[f"test_samples_{label}"] = test_label_count[label]

        rows.append(row)

    rows = sorted(
        rows,
        key=lambda item: (
            item["accuracy"],
            item["test_quadrant"],
        ),
    )

    return rows


# ================= EXPERIMENT INDEX =================

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
        "description": "Dataset split and data quality diagnostic",
        "model": "gradient_boosting_depth3_20",
        "validation_method": "Repeated file-based holdout plus cross-quadrant holdout",
        "top_k": MODEL_TOP_K,
        "max_depth": MODEL_MAX_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "accuracy": f"{best_summary['accuracy_mean']:.6f}",
        "macro_f1": f"{best_summary['macro_f1_mean']:.6f}",
        "weighted_f1": f"{best_summary['weighted_f1_mean']:.6f}",
        "main_observation": (
            "Diagnostic experiment focused on split instability, weak files, "
            "quadrant effects, and cross-quadrant generalization"
        ),
    }

    existing_rows.append(new_row)

    with open(index_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in existing_rows:
            writer.writerow(row)


# ================= NOTES =================

def build_metric_summary(split_quality_rows):
    accuracy_values = []
    macro_f1_values = []
    weighted_f1_values = []
    movement_f1_values = []

    for row in split_quality_rows:
        accuracy_values.append(float(row["accuracy"]))
        macro_f1_values.append(float(row["macro_f1"]))
        weighted_f1_values.append(float(row["weighted_f1"]))
        movement_f1_values.append(float(row["movement_f1"]))

    return {
        "accuracy_mean": mean(accuracy_values),
        "accuracy_std": std(accuracy_values),
        "accuracy_min": min(accuracy_values),
        "accuracy_max": max(accuracy_values),

        "macro_f1_mean": mean(macro_f1_values),
        "macro_f1_std": std(macro_f1_values),
        "macro_f1_min": min(macro_f1_values),
        "macro_f1_max": max(macro_f1_values),

        "weighted_f1_mean": mean(weighted_f1_values),
        "weighted_f1_std": std(weighted_f1_values),

        "movement_f1_mean": mean(movement_f1_values),
        "movement_f1_std": std(movement_f1_values),
        "movement_f1_min": min(movement_f1_values),
        "movement_f1_max": max(movement_f1_values),
    }


def save_experiment_notes(
    output_path,
    feature_dataset,
    groups,
    file_group_rows,
    split_quality_rows,
    file_performance_rows,
    quadrant_performance_rows,
    class_quadrant_rows,
    cross_quadrant_rows,
    metric_summary,
):
    window_counts = []

    for group in groups:
        window_counts.append(group["sample_count"])

    low_window_groups = []

    for row in file_group_rows:
        if row["low_window_count"]:
            low_window_groups.append(row)

    worst_files = file_performance_rows[:10]
    worst_quadrants = quadrant_performance_rows[:5]
    worst_class_quadrants = class_quadrant_rows[:10]

    lines = []

    lines.append("Experiment 019 - Dataset split and data quality diagnostic")
    lines.append("")
    lines.append("Objective:")
    lines.append(
        "Diagnose why model performance changes significantly across "
        "different file-based holdout splits."
    )
    lines.append("")
    lines.append("Model used for diagnostic:")
    lines.append(f"- Gradient Boosting, n_estimators={MODEL_N_ESTIMATORS}")
    lines.append(f"- max_depth={MODEL_MAX_DEPTH}")
    lines.append(f"- top_k={MODEL_TOP_K}")
    lines.append("")
    lines.append("Dataset:")
    lines.append(f"- Total samples/windows: {len(feature_dataset)}")
    lines.append(f"- Features per sample: {len(feature_dataset[0]['features'])}")
    lines.append(f"- Total file groups: {len(groups)}")
    lines.append(f"- Minimum windows per file: {min(window_counts)}")
    lines.append(f"- Maximum windows per file: {max(window_counts)}")
    lines.append(f"- Mean windows per file: {mean(window_counts):.6f}")
    lines.append(f"- Groups with fewer than {LOW_WINDOW_COUNT_THRESHOLD} windows: {len(low_window_groups)}")
    lines.append("")
    lines.append("Repeated split performance:")
    lines.append(f"- accuracy mean: {metric_summary['accuracy_mean']:.6f}")
    lines.append(f"- accuracy std: {metric_summary['accuracy_std']:.6f}")
    lines.append(f"- accuracy min: {metric_summary['accuracy_min']:.6f}")
    lines.append(f"- accuracy max: {metric_summary['accuracy_max']:.6f}")
    lines.append(f"- macro F1 mean: {metric_summary['macro_f1_mean']:.6f}")
    lines.append(f"- macro F1 std: {metric_summary['macro_f1_std']:.6f}")
    lines.append(f"- macro F1 min: {metric_summary['macro_f1_min']:.6f}")
    lines.append(f"- macro F1 max: {metric_summary['macro_f1_max']:.6f}")
    lines.append(f"- movement F1 mean: {metric_summary['movement_f1_mean']:.6f}")
    lines.append(f"- movement F1 std: {metric_summary['movement_f1_std']:.6f}")
    lines.append("")
    lines.append("Worst files/groups by test accuracy:")
    for row in worst_files:
        lines.append(
            f"- label={row['label']}, quadrant={row['quadrant']}, "
            f"windows={row['original_window_count']}, "
            f"tested_windows={row['tested_windows']}, "
            f"accuracy={float(row['accuracy']):.6f}, "
            f"group_id={row['group_id']}"
        )

    lines.append("")
    lines.append("Worst quadrants by test accuracy:")
    for row in worst_quadrants:
        lines.append(
            f"- quadrant={row['quadrant']}, "
            f"tested_windows={row['tested_windows']}, "
            f"accuracy={float(row['accuracy']):.6f}"
        )

    lines.append("")
    lines.append("Worst class/quadrant pairs by test accuracy:")
    for row in worst_class_quadrants:
        lines.append(
            f"- label={row['label']}, quadrant={row['quadrant']}, "
            f"tested_windows={row['tested_windows']}, "
            f"accuracy={float(row['accuracy']):.6f}"
        )

    lines.append("")
    lines.append("Cross-quadrant holdout:")
    for row in cross_quadrant_rows:
        lines.append(
            f"- test_quadrant={row['test_quadrant']}, "
            f"accuracy={float(row['accuracy']):.6f}, "
            f"macro_f1_supported={float(row['macro_f1_supported_classes']):.6f}, "
            f"test_samples={row['test_samples']}"
        )

    lines.append("")
    lines.append("Interpretation rule:")
    lines.append(
        "If specific files, quadrants, or class/quadrant pairs show much lower "
        "accuracy, the next action should focus on data quality and collection "
        "protocol rather than classifier tuning."
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
    quadrants = get_unique_quadrants(groups)

    print()
    print("Dataset Split and Data Quality Diagnostic")
    print("=" * 70)
    print("Experiment:", EXPERIMENT_ID)
    print("Total samples/windows:", len(feature_dataset))
    print("Total file groups:", len(groups))
    print("Quadrants:", quadrants)
    print()

    file_group_rows = build_file_group_rows(groups)

    split_quality_rows = []
    all_predictions = []

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

        metrics, predictions, selected_indices = evaluate_split(
            train_dataset,
            test_dataset,
            seed,
        )

        for prediction in predictions:
            prediction["seed"] = seed
            all_predictions.append(prediction)

        row = build_split_quality_row(
            seed,
            train_dataset,
            test_dataset,
            train_groups,
            test_groups,
            metrics,
            quadrants,
        )

        split_quality_rows.append(row)

        print(f"train_samples={len(train_dataset)}")
        print(f"test_samples={len(test_dataset)}")
        print(f"accuracy={metrics['accuracy']:.6f}")
        print(f"macro_f1={metrics['macro_f1']:.6f}")
        print(f"movement_f1={metrics.get('movement_f1', 0.0):.6f}")

    file_performance_rows = summarize_file_test_performance(
        all_predictions,
        file_group_rows,
    )

    quadrant_performance_rows = summarize_quadrant_performance(
        all_predictions,
    )

    class_quadrant_rows = summarize_class_quadrant_performance(
        all_predictions,
    )

    cross_quadrant_rows = run_cross_quadrant_holdout(
        feature_dataset,
        quadrants,
    )

    metric_summary = build_metric_summary(
        split_quality_rows,
    )

    paths = get_experiment_paths()

    save_csv(file_group_rows, paths["dataset_file_groups"])
    save_csv(split_quality_rows, paths["split_quality_by_seed"])
    save_csv(file_performance_rows, paths["file_test_performance"])
    save_csv(quadrant_performance_rows, paths["quadrant_performance"])
    save_csv(class_quadrant_rows, paths["class_quadrant_performance"])
    save_csv(cross_quadrant_rows, paths["cross_quadrant_holdout"])

    save_experiment_notes(
        paths["experiment_notes"],
        feature_dataset,
        groups,
        file_group_rows,
        split_quality_rows,
        file_performance_rows,
        quadrant_performance_rows,
        class_quadrant_rows,
        cross_quadrant_rows,
        metric_summary,
    )

    update_experiment_index(metric_summary)

    print()
    print("=" * 70)
    print("Dataset split and quality diagnostic finished.")
    print("=" * 70)
    print("Repeated split summary:")
    print(f"accuracy_mean: {metric_summary['accuracy_mean']:.6f}")
    print(f"accuracy_std: {metric_summary['accuracy_std']:.6f}")
    print(f"macro_f1_mean: {metric_summary['macro_f1_mean']:.6f}")
    print(f"macro_f1_std: {metric_summary['macro_f1_std']:.6f}")
    print(f"movement_f1_mean: {metric_summary['movement_f1_mean']:.6f}")
    print(f"movement_f1_std: {metric_summary['movement_f1_std']:.6f}")
    print()
    print("Results saved to:")
    print(paths["run_dir"])


def main():
    run_experiment()


if __name__ == "__main__":
    main()