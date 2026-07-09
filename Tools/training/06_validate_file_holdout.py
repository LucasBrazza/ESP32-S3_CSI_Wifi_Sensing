"""
File-Based Holdout Validation

This script validates the tuned decision tree classifier using a stricter
holdout strategy.

Instead of splitting individual windows randomly, this script splits the
dataset by source file / acquisition file.

This means:

    all windows from the same acquisition file go either to training
    or to testing

This avoids data leakage between train and test sets, since windows from
the same acquisition tend to be very similar.

Experiment:

    004_file_holdout_tuned_depth6_min5_top30

Expected current classifier configuration:

    MAX_TREE_DEPTH = 6
    MIN_SAMPLES_SPLIT = 5
"""

import csv
import random
from collections import Counter

import matplotlib.pyplot as plt

from Tools.common.config import (
    MODEL,
    MAX_TREE_DEPTH,
    MIN_SAMPLES_SPLIT,
    TOP_K_FEATURES,
)

from Tools.common.io_utils import load_pickle

from Tools.common.project_paths import (
    SELECTED_FEATURE_DATASET_FILE,
    RESULTS_DIR,
)

from Tools.classification import decision_tree


CLASS_ORDER = [
    "empty",
    "static_presence",
    "movement",
]

TEST_SIZE = 0.20
RANDOM_SEED = 42

EXPERIMENT_ID = "006"
EXPERIMENT_FOLDER_NAME = "006_file_holdout_topk70_depth6_min5"


# ================= DATASET HELPERS =================

def get_labels(dataset):
    labels_found = []

    for item in dataset:
        label = item["label"]

        if label not in labels_found:
            labels_found.append(label)

    ordered_labels = []

    for label in CLASS_ORDER:
        if label in labels_found:
            ordered_labels.append(label)

    for label in labels_found:
        if label not in ordered_labels:
            ordered_labels.append(label)

    return ordered_labels


def get_sample_group_id(sample):
    """
    Return a unique identifier for the acquisition file.

    Priority:
        1. source_file
        2. combination of label, quadrant and file_name

    The fallback avoids collisions if different classes or quadrants
    happen to use the same file name.
    """
    source_file = sample.get("source_file", "")

    if source_file:
        return str(source_file)

    label = sample.get("label", "unknown_label")
    quadrant = sample.get("quadrant", "unknown_quadrant")
    file_name = sample.get("file_name", "unknown_file")

    return f"{label}|{quadrant}|{file_name}"


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

        for sample in samples:
            label = sample["label"]

            if label not in labels:
                labels.append(label)

        if len(labels) != 1:
            raise ValueError(
                "A file group contains more than one label. "
                f"group_id={group_id}, labels={labels}"
            )

        representative = samples[0]

        grouped_items.append(
            {
                "group_id": group_id,
                "label": labels[0],
                "quadrant": representative.get("quadrant", "unknown"),
                "file_name": representative.get("file_name", ""),
                "source_file": representative.get("source_file", ""),
                "samples": samples,
                "sample_count": len(samples),
            }
        )

    return grouped_items


def file_stratified_holdout_split(dataset, test_size=0.20, seed=42):
    """
    Split the dataset by acquisition file while preserving class balance
    as much as possible.

    The split is stratified by file label, not by individual window.

    Because different files can generate different numbers of windows,
    the final train/test sample ratio may not be exactly 80/20.
    """
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


def count_by_label(dataset):
    counter = Counter()

    for item in dataset:
        counter[item["label"]] += 1

    return counter


def count_groups_by_label(groups):
    counter = Counter()

    for group in groups:
        counter[group["label"]] += 1

    return counter


# ================= METRICS =================

def safe_division(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def predict_dataset(tree, dataset):
    predictions = []

    for sample_index, sample in enumerate(dataset):
        predicted_label = decision_tree.predict_one(
            tree,
            sample["features"],
        )

        true_label = sample["label"]

        predictions.append(
            {
                "sample_index": sample_index,
                "sample_id": sample.get("sample_id", ""),
                "group_id": get_sample_group_id(sample),
                "quadrant": sample.get("quadrant", "unknown"),
                "file_name": sample.get("file_name", ""),
                "source_file": sample.get("source_file", ""),
                "window_index": sample.get("window_index", ""),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": true_label == predicted_label,
            }
        )

    return predictions


def build_confusion_matrix(predictions, labels):
    label_to_index = {}

    for index, label in enumerate(labels):
        label_to_index[label] = index

    matrix = []

    for _ in labels:
        row = []

        for _ in labels:
            row.append(0)

        matrix.append(row)

    for item in predictions:
        true_label = item["true_label"]
        predicted_label = item["predicted_label"]

        true_index = label_to_index[true_label]
        predicted_index = label_to_index[predicted_label]

        matrix[true_index][predicted_index] += 1

    return matrix


def compute_class_metrics(matrix, labels):
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

    macro_precision = safe_division(
        sum(item["precision"] for item in per_class),
        len(per_class),
    )

    macro_recall = safe_division(
        sum(item["recall"] for item in per_class),
        len(per_class),
    )

    macro_f1 = safe_division(
        sum(item["f1_score"] for item in per_class),
        len(per_class),
    )

    weighted_precision = safe_division(
        sum(item["precision"] * item["support"] for item in per_class),
        total_samples,
    )

    weighted_recall = safe_division(
        sum(item["recall"] * item["support"] for item in per_class),
        total_samples,
    )

    weighted_f1 = safe_division(
        sum(item["f1_score"] * item["support"] for item in per_class),
        total_samples,
    )

    return {
        "per_class": per_class,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


def compute_quadrant_metrics(predictions):
    grouped = {}

    for item in predictions:
        quadrant = item.get("quadrant", "unknown")

        if quadrant not in grouped:
            grouped[quadrant] = {
                "quadrant": quadrant,
                "support": 0,
                "correct": 0,
            }

        grouped[quadrant]["support"] += 1

        if item["correct"]:
            grouped[quadrant]["correct"] += 1

    metrics = []

    for quadrant in sorted(grouped):
        item = grouped[quadrant]

        metrics.append(
            {
                "quadrant": quadrant,
                "accuracy": safe_division(
                    item["correct"],
                    item["support"],
                ),
                "correct": item["correct"],
                "support": item["support"],
            }
        )

    return metrics


def compute_file_metrics(predictions):
    grouped = {}

    for item in predictions:
        group_id = item["group_id"]

        if group_id not in grouped:
            grouped[group_id] = {
                "group_id": group_id,
                "file_name": item.get("file_name", ""),
                "source_file": item.get("source_file", ""),
                "quadrant": item.get("quadrant", "unknown"),
                "true_label": item["true_label"],
                "support": 0,
                "correct": 0,
            }

        grouped[group_id]["support"] += 1

        if item["correct"]:
            grouped[group_id]["correct"] += 1

    metrics = []

    for group_id in sorted(grouped):
        item = grouped[group_id]

        metrics.append(
            {
                "group_id": item["group_id"],
                "file_name": item["file_name"],
                "source_file": item["source_file"],
                "quadrant": item["quadrant"],
                "true_label": item["true_label"],
                "accuracy": safe_division(
                    item["correct"],
                    item["support"],
                ),
                "correct": item["correct"],
                "support": item["support"],
            }
        )

    return metrics


def analyze_tree(tree):
    if tree["type"] == "leaf":
        return {
            "nodes": 1,
            "decision_nodes": 0,
            "leaf_nodes": 1,
            "max_depth": tree.get("depth", 0),
        }

    left = analyze_tree(tree["left"])
    right = analyze_tree(tree["right"])

    return {
        "nodes": 1 + left["nodes"] + right["nodes"],
        "decision_nodes": 1 + left["decision_nodes"] + right["decision_nodes"],
        "leaf_nodes": left["leaf_nodes"] + right["leaf_nodes"],
        "max_depth": max(left["max_depth"], right["max_depth"]),
    }


# ================= OUTPUT PATHS =================

def get_experiment_paths():
    run_dir = RESULTS_DIR / "runs" / EXPERIMENT_FOLDER_NAME
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    reports_dir = run_dir / "reports"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "tables_dir": tables_dir,
        "figures_dir": figures_dir,
        "reports_dir": reports_dir,
        "predictions": tables_dir / "classifier_predictions.csv",
        "confusion_matrix": tables_dir / "confusion_matrix.csv",
        "class_metrics": tables_dir / "class_metrics.csv",
        "quadrant_metrics": tables_dir / "quadrant_metrics.csv",
        "file_metrics": tables_dir / "file_metrics.csv",
        "file_split_summary": tables_dir / "file_split_summary.csv",
        "confusion_matrix_plot": figures_dir / "confusion_matrix.png",
        "report": reports_dir / "classifier_validation_report.txt",
        "experiment_notes": run_dir / "experiment_notes.txt",
    }


# ================= SAVE OUTPUTS =================

def save_predictions(predictions, output_path):
    fieldnames = [
        "sample_index",
        "sample_id",
        "group_id",
        "quadrant",
        "file_name",
        "source_file",
        "window_index",
        "true_label",
        "predicted_label",
        "correct",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for item in predictions:
            writer.writerow(item)


def save_confusion_matrix(matrix, labels, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        header = ["true_label/predicted_label"]

        for label in labels:
            header.append(label)

        writer.writerow(header)

        for index, label in enumerate(labels):
            row = [label]

            for value in matrix[index]:
                row.append(value)

            writer.writerow(row)


def save_class_metrics(metrics_result, output_path):
    fieldnames = [
        "label",
        "precision",
        "recall",
        "f1_score",
        "support",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for item in metrics_result["per_class"]:
            writer.writerow(item)


def save_quadrant_metrics(quadrant_metrics, output_path):
    fieldnames = [
        "quadrant",
        "accuracy",
        "correct",
        "support",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for item in quadrant_metrics:
            writer.writerow(item)


def save_file_metrics(file_metrics, output_path):
    fieldnames = [
        "group_id",
        "file_name",
        "source_file",
        "quadrant",
        "true_label",
        "accuracy",
        "correct",
        "support",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for item in file_metrics:
            writer.writerow(item)


def save_file_split_summary(train_groups, test_groups, output_path):
    fieldnames = [
        "split",
        "group_id",
        "label",
        "quadrant",
        "file_name",
        "source_file",
        "sample_count",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for group in train_groups:
            writer.writerow(
                {
                    "split": "train",
                    "group_id": group["group_id"],
                    "label": group["label"],
                    "quadrant": group["quadrant"],
                    "file_name": group["file_name"],
                    "source_file": group["source_file"],
                    "sample_count": group["sample_count"],
                }
            )

        for group in test_groups:
            writer.writerow(
                {
                    "split": "test",
                    "group_id": group["group_id"],
                    "label": group["label"],
                    "quadrant": group["quadrant"],
                    "file_name": group["file_name"],
                    "source_file": group["source_file"],
                    "sample_count": group["sample_count"],
                }
            )


def save_confusion_matrix_plot(matrix, labels, output_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)

    plt.title("Confusion Matrix - File-Based Holdout")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(labels)),
        labels,
    )

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(
                j,
                i,
                str(matrix[i][j]),
                ha="center",
                va="center",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_report(
    output_path,
    dataset,
    train_dataset,
    test_dataset,
    train_groups,
    test_groups,
    labels,
    matrix,
    metrics_result,
    quadrant_metrics,
    file_metrics,
    tree_complexity,
):
    total_class_count = count_by_label(dataset)
    train_class_count = count_by_label(train_dataset)
    test_class_count = count_by_label(test_dataset)

    train_group_count = count_groups_by_label(train_groups)
    test_group_count = count_groups_by_label(test_groups)

    total_groups = train_groups + test_groups
    total_group_count = count_groups_by_label(total_groups)

    lines = []

    lines.append("FILE-BASED HOLDOUT VALIDATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Experiment: {EXPERIMENT_ID} - {EXPERIMENT_FOLDER_NAME}")
    lines.append(f"Model: {MODEL}")
    lines.append("Validation method: File-based stratified holdout 80/20")
    lines.append(f"Random seed: {RANDOM_SEED}")
    lines.append("")
    lines.append("Dataset:")
    lines.append(f"- Total samples/windows: {len(dataset)}")
    lines.append(f"- Train samples/windows: {len(train_dataset)}")
    lines.append(f"- Test samples/windows: {len(test_dataset)}")
    lines.append(f"- Total files/groups: {len(total_groups)}")
    lines.append(f"- Train files/groups: {len(train_groups)}")
    lines.append(f"- Test files/groups: {len(test_groups)}")
    lines.append(f"- Features per sample: {len(dataset[0]['features'])}")
    lines.append("")
    lines.append("Classifier parameters:")
    lines.append(f"- Max tree depth: {MAX_TREE_DEPTH}")
    lines.append(f"- Min samples split: {MIN_SAMPLES_SPLIT}")
    lines.append("")
    lines.append("Tree complexity:")
    lines.append(f"- Total nodes: {tree_complexity['nodes']}")
    lines.append(f"- Decision nodes: {tree_complexity['decision_nodes']}")
    lines.append(f"- Leaf nodes: {tree_complexity['leaf_nodes']}")
    lines.append(f"- Actual max depth: {tree_complexity['max_depth']}")
    lines.append("")

    lines.append("Class distribution by samples/windows")
    lines.append("-" * 70)
    lines.append(f"{'Class':<20}{'Total':>10}{'Train':>10}{'Test':>10}")

    for label in labels:
        lines.append(
            f"{label:<20}"
            f"{total_class_count[label]:>10}"
            f"{train_class_count[label]:>10}"
            f"{test_class_count[label]:>10}"
        )

    lines.append("")
    lines.append("Class distribution by files/groups")
    lines.append("-" * 70)
    lines.append(f"{'Class':<20}{'Total':>10}{'Train':>10}{'Test':>10}")

    for label in labels:
        lines.append(
            f"{label:<20}"
            f"{total_group_count[label]:>10}"
            f"{train_group_count[label]:>10}"
            f"{test_group_count[label]:>10}"
        )

    lines.append("")
    lines.append("Overall metrics")
    lines.append("-" * 70)
    lines.append(f"Accuracy: {metrics_result['accuracy']:.6f}")
    lines.append(f"Macro precision: {metrics_result['macro_precision']:.6f}")
    lines.append(f"Macro recall: {metrics_result['macro_recall']:.6f}")
    lines.append(f"Macro F1-score: {metrics_result['macro_f1']:.6f}")
    lines.append(f"Weighted precision: {metrics_result['weighted_precision']:.6f}")
    lines.append(f"Weighted recall: {metrics_result['weighted_recall']:.6f}")
    lines.append(f"Weighted F1-score: {metrics_result['weighted_f1']:.6f}")
    lines.append("")

    lines.append("Per-class metrics")
    lines.append("-" * 70)

    for item in metrics_result["per_class"]:
        lines.append(
            f"{item['label']}: "
            f"precision={item['precision']:.6f} | "
            f"recall={item['recall']:.6f} | "
            f"f1={item['f1_score']:.6f} | "
            f"support={item['support']}"
        )

    lines.append("")
    lines.append("Per-quadrant metrics")
    lines.append("-" * 70)

    for item in quadrant_metrics:
        lines.append(
            f"{item['quadrant']}: "
            f"accuracy={item['accuracy']:.6f} | "
            f"correct={item['correct']} | "
            f"support={item['support']}"
        )

    lines.append("")
    lines.append("File-level metrics summary")
    lines.append("-" * 70)
    lines.append(f"Test files evaluated: {len(file_metrics)}")

    if file_metrics:
        average_file_accuracy = safe_division(
            sum(item["accuracy"] for item in file_metrics),
            len(file_metrics),
        )

        lines.append(f"Average file accuracy: {average_file_accuracy:.6f}")

    lines.append("")
    lines.append("Confusion matrix")
    lines.append("-" * 70)
    lines.append("Rows = true labels | Columns = predicted labels")
    lines.append("")

    header = " " * 20

    for label in labels:
        header += f"{label:>18}"

    lines.append(header)

    for row_index, label in enumerate(labels):
        row = f"{label:<20}"

        for value in matrix[row_index]:
            row += f"{value:>18}"

        lines.append(row)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def save_experiment_notes(
    output_path,
    metrics_result,
):
    lines = []

    lines.append("Experiment 004 - File-based holdout validation")
    lines.append("")
    lines.append("Description:")
    lines.append(
        "Validation of the tuned decision tree using file-based holdout. "
        "All windows generated from the same acquisition file are kept "
        "together either in training or in testing."
    )
    lines.append("")
    lines.append("Motivation:")
    lines.append(
        "Previous holdout experiments split individual windows randomly. "
        "That strategy is useful as an initial validation, but it can be "
        "optimistic because different windows from the same acquisition "
        "may be very similar. File-based splitting provides a stricter "
        "evaluation."
    )
    lines.append("")
    lines.append("Configuration:")
    lines.append("- Validation method: file-based stratified holdout 80/20")
    lines.append(f"- Random seed: {RANDOM_SEED}")
    lines.append(f"- Max tree depth: {MAX_TREE_DEPTH}")
    lines.append(f"- Min samples split: {MIN_SAMPLES_SPLIT}")
    lines.append(f"- Top-K selected features: {TOP_K_FEATURES}")
    lines.append("")
    lines.append("Results:")
    lines.append(f"- Accuracy: {metrics_result['accuracy']:.6f}")
    lines.append(f"- Macro F1-score: {metrics_result['macro_f1']:.6f}")
    lines.append(f"- Weighted F1-score: {metrics_result['weighted_f1']:.6f}")
    lines.append("")
    lines.append("Main observation:")
    lines.append(
        "This result should be compared with experiment 003. "
        "If performance drops significantly, it indicates that the "
        "window-based holdout was benefiting from similarity between "
        "windows from the same acquisition file."
    )
    lines.append("")
    lines.append("Next step:")
    lines.append(
        "Analyze the difference between experiments 003 and 004, focusing "
        "on whether the model generalizes to unseen acquisition files."
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def update_experiment_index(metrics_result):
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
        "description": "File-based holdout validation using tuned decision tree",
        "model": "decision_tree",
        "validation_method": "File-based stratified holdout 80/20",
        "top_k": TOP_K_FEATURES,
        "max_depth": MAX_TREE_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "accuracy": f"{metrics_result['accuracy']:.6f}",
        "macro_f1": f"{metrics_result['macro_f1']:.6f}",
        "weighted_f1": f"{metrics_result['weighted_f1']:.6f}",
        "main_observation": (
            "Stricter validation by acquisition file to evaluate "
            "generalization to unseen files"
        ),
    }

    existing_rows.append(new_row)

    with open(index_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in existing_rows:
            writer.writerow(row)


# ================= MAIN =================

def main():
    print()
    print("File-Based Holdout Validation")
    print("=" * 70)

    dataset = load_pickle(SELECTED_FEATURE_DATASET_FILE)

    if not dataset:
        raise ValueError("Selected feature dataset is empty.")

    labels = get_labels(dataset)

    print("Model:", MODEL)
    print("Validation method: File-based stratified holdout 80/20")
    print("Samples/windows:", len(dataset))
    print("Features per sample:", len(dataset[0]["features"]))
    print("Labels:", labels)
    print("Max tree depth:", MAX_TREE_DEPTH)
    print("Min samples split:", MIN_SAMPLES_SPLIT)

    train_dataset, test_dataset, train_groups, test_groups = file_stratified_holdout_split(
        dataset,
        test_size=TEST_SIZE,
        seed=RANDOM_SEED,
    )

    print()
    print("Train files/groups:", len(train_groups))
    print("Test files/groups:", len(test_groups))
    print("Train samples/windows:", len(train_dataset))
    print("Test samples/windows:", len(test_dataset))

    tree = decision_tree.build_tree(
        train_dataset,
        max_depth=MAX_TREE_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
    )

    tree_complexity = analyze_tree(tree)

    predictions = predict_dataset(
        tree,
        test_dataset,
    )

    matrix = build_confusion_matrix(
        predictions,
        labels,
    )

    metrics_result = compute_class_metrics(
        matrix,
        labels,
    )

    quadrant_metrics = compute_quadrant_metrics(
        predictions,
    )

    file_metrics = compute_file_metrics(
        predictions,
    )

    paths = get_experiment_paths()

    save_predictions(predictions, paths["predictions"])
    save_confusion_matrix(matrix, labels, paths["confusion_matrix"])
    save_class_metrics(metrics_result, paths["class_metrics"])
    save_quadrant_metrics(quadrant_metrics, paths["quadrant_metrics"])
    save_file_metrics(file_metrics, paths["file_metrics"])
    save_file_split_summary(train_groups, test_groups, paths["file_split_summary"])
    save_confusion_matrix_plot(matrix, labels, paths["confusion_matrix_plot"])

    save_report(
        paths["report"],
        dataset,
        train_dataset,
        test_dataset,
        train_groups,
        test_groups,
        labels,
        matrix,
        metrics_result,
        quadrant_metrics,
        file_metrics,
        tree_complexity,
    )

    save_experiment_notes(
        paths["experiment_notes"],
        metrics_result,
    )

    update_experiment_index(metrics_result)

    print()
    print("Accuracy:", metrics_result["accuracy"])
    print("Macro F1-score:", metrics_result["macro_f1"])
    print("Weighted F1-score:", metrics_result["weighted_f1"])
    print()
    print("Results saved to:")
    print(paths["run_dir"])


if __name__ == "__main__":
    main()