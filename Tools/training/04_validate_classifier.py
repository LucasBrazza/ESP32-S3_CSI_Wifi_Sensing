"""
Classifier Validation

This script validates the decision tree classifier using a stratified
holdout split.

The selected feature dataset is divided into:

    80% training samples
    20% test samples

The split is stratified by class label, which keeps the proportion of
empty, static_presence and movement samples in both sets.

Outputs:

    datasets/results/tables/confusion_matrix.csv
    datasets/results/tables/class_metrics.csv
    datasets/results/tables/quadrant_metrics.csv
    datasets/results/tables/classifier_predictions.csv
    datasets/results/figures/confusion_matrix.png
    datasets/results/reports/classifier_validation_report.txt
"""

import csv
import random
from collections import Counter

import matplotlib.pyplot as plt

from Tools.common.config import (
    MODEL,
    MAX_TREE_DEPTH,
    MIN_SAMPLES_SPLIT,
)

from Tools.common.io_utils import load_pickle

from Tools.common.project_paths import (
    SELECTED_FEATURE_DATASET_FILE,
    TABLES_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
)

from Tools.classification import decision_tree


CLASS_ORDER = [
    "empty",
    "static_presence",
    "movement",
]

TEST_SIZE = 0.20
RANDOM_SEED = 42


# ================= DATASET SPLIT =================

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


def stratified_holdout_split(dataset, test_size=0.20, seed=42):
    """
    Split the dataset into train and test sets while preserving class
    distribution.

    The split is performed independently for each class label:

        80% of empty samples go to train
        20% of empty samples go to test
        80% of static_presence samples go to train
        20% of static_presence samples go to test
        80% of movement samples go to train
        20% of movement samples go to test

    This avoids a test set with missing or poorly represented classes.
    """
    random_generator = random.Random(seed)

    samples_by_label = {}

    for sample in dataset:
        label = sample["label"]

        if label not in samples_by_label:
            samples_by_label[label] = []

        samples_by_label[label].append(sample)

    train_dataset = []
    test_dataset = []

    for label in samples_by_label:
        samples = samples_by_label[label][:]
        random_generator.shuffle(samples)

        test_count = int(round(len(samples) * test_size))

        if test_count < 1 and len(samples) > 1:
            test_count = 1

        if test_count >= len(samples):
            test_count = len(samples) - 1

        test_samples = samples[:test_count]
        train_samples = samples[test_count:]

        test_dataset.extend(test_samples)
        train_dataset.extend(train_samples)

    random_generator.shuffle(train_dataset)
    random_generator.shuffle(test_dataset)

    return train_dataset, test_dataset


# ================= METRICS =================

def safe_division(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def predict_sample(tree, sample):
    """
    Predict one sample using the project decision tree implementation.
    """
    return decision_tree.predict_one(
        tree,
        sample["features"],
    )


def holdout_validation(train_dataset, test_dataset, max_depth, min_samples_split):
    tree = decision_tree.build_tree(
        train_dataset,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
    )

    predictions = []

    for test_index, test_sample in enumerate(test_dataset):
        predicted_label = predict_sample(
            tree,
            test_sample,
        )

        true_label = test_sample["label"]

        predictions.append(
            {
                "sample_index": test_index,
                "sample_id": test_sample.get("sample_id", ""),
                "quadrant": test_sample.get("quadrant", "unknown"),
                "file_name": test_sample.get("file_name", ""),
                "window_index": test_sample.get("window_index", ""),
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


def compute_accuracy(predictions):
    if not predictions:
        return 0.0

    correct = 0

    for item in predictions:
        if item["correct"]:
            correct += 1

    return correct / len(predictions)


def compute_class_metrics(matrix, labels):
    metrics = []

    total_samples = 0
    total_correct = 0

    for i in range(len(labels)):
        total_correct += matrix[i][i]

        for value in matrix[i]:
            total_samples += value

    for i, label in enumerate(labels):
        true_positive = matrix[i][i]

        false_positive = 0
        false_negative = 0

        for row_index in range(len(labels)):
            if row_index != i:
                false_positive += matrix[row_index][i]

        for column_index in range(len(labels)):
            if column_index != i:
                false_negative += matrix[i][column_index]

        support = sum(matrix[i])

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

        metrics.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "support": support,
            }
        )

    macro_precision = safe_division(
        sum(item["precision"] for item in metrics),
        len(metrics),
    )

    macro_recall = safe_division(
        sum(item["recall"] for item in metrics),
        len(metrics),
    )

    macro_f1 = safe_division(
        sum(item["f1_score"] for item in metrics),
        len(metrics),
    )

    weighted_precision = safe_division(
        sum(item["precision"] * item["support"] for item in metrics),
        total_samples,
    )

    weighted_recall = safe_division(
        sum(item["recall"] * item["support"] for item in metrics),
        total_samples,
    )

    weighted_f1 = safe_division(
        sum(item["f1_score"] * item["support"] for item in metrics),
        total_samples,
    )

    accuracy = safe_division(
        total_correct,
        total_samples,
    )

    return {
        "per_class": metrics,
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
        accuracy = safe_division(
            item["correct"],
            item["support"],
        )

        metrics.append(
            {
                "quadrant": quadrant,
                "accuracy": accuracy,
                "correct": item["correct"],
                "support": item["support"],
            }
        )

    return metrics


# ================= SAVE OUTPUTS =================

def save_predictions(predictions):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = TABLES_DIR / "classifier_predictions.csv"

    fieldnames = [
        "sample_index",
        "sample_id",
        "quadrant",
        "file_name",
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

    return output_path


def save_confusion_matrix(matrix, labels):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = TABLES_DIR / "confusion_matrix.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        header = ["true_label/predicted_label"]

        for label in labels:
            header.append(label)

        writer.writerow(header)

        for i, label in enumerate(labels):
            row = [label]

            for value in matrix[i]:
                row.append(value)

            writer.writerow(row)

    return output_path


def save_class_metrics(metrics_result):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = TABLES_DIR / "class_metrics.csv"

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

    return output_path


def save_quadrant_metrics(quadrant_metrics):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = TABLES_DIR / "quadrant_metrics.csv"

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

    return output_path


def save_confusion_matrix_plot(matrix, labels):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / "confusion_matrix.png"

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)

    plt.title("Confusion Matrix - Stratified Holdout 80/20")
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

    return output_path


def count_by_label(dataset):
    counter = Counter()

    for item in dataset:
        counter[item["label"]] += 1

    return counter


def save_report(
    dataset,
    train_dataset,
    test_dataset,
    labels,
    matrix,
    metrics_result,
    quadrant_metrics,
):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / "classifier_validation_report.txt"

    total_class_count = count_by_label(dataset)
    train_class_count = count_by_label(train_dataset)
    test_class_count = count_by_label(test_dataset)

    lines = []

    lines.append("CLASSIFIER VALIDATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Model: {MODEL}")
    lines.append("Validation method: Stratified holdout 80/20")
    lines.append(f"Random seed: {RANDOM_SEED}")
    lines.append(f"Total samples: {len(dataset)}")
    lines.append(f"Train samples: {len(train_dataset)}")
    lines.append(f"Test samples: {len(test_dataset)}")
    lines.append(f"Features per sample: {len(dataset[0]['features'])}")
    lines.append(f"Max tree depth: {MAX_TREE_DEPTH}")
    lines.append(f"Min samples split: {MIN_SAMPLES_SPLIT}")
    lines.append("")

    lines.append("Class distribution")
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
    lines.append("Confusion matrix")
    lines.append("-" * 70)
    lines.append("Rows = true labels | Columns = predicted labels")
    lines.append("")

    header = " " * 20

    for label in labels:
        header += f"{label:>18}"

    lines.append(header)

    for i, label in enumerate(labels):
        row = f"{label:<20}"

        for value in matrix[i]:
            row += f"{value:>18}"

        lines.append(row)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    return output_path


# ================= MAIN =================

def main():
    print()
    print("Classifier Validation")
    print("=" * 70)

    dataset = load_pickle(SELECTED_FEATURE_DATASET_FILE)

    if not dataset:
        raise ValueError("Selected feature dataset is empty.")

    labels = get_labels(dataset)

    print("Model:", MODEL)
    print("Validation method: Stratified holdout 80/20")
    print("Samples:", len(dataset))
    print("Features per sample:", len(dataset[0]["features"]))
    print("Labels:", labels)

    train_dataset, test_dataset = stratified_holdout_split(
        dataset,
        test_size=TEST_SIZE,
        seed=RANDOM_SEED,
    )

    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    predictions = holdout_validation(
        train_dataset,
        test_dataset,
        max_depth=MAX_TREE_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
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

    predictions_path = save_predictions(predictions)
    confusion_matrix_path = save_confusion_matrix(matrix, labels)
    class_metrics_path = save_class_metrics(metrics_result)
    quadrant_metrics_path = save_quadrant_metrics(quadrant_metrics)
    figure_path = save_confusion_matrix_plot(matrix, labels)
    report_path = save_report(
        dataset,
        train_dataset,
        test_dataset,
        labels,
        matrix,
        metrics_result,
        quadrant_metrics,
    )

    print()
    print("Accuracy:", metrics_result["accuracy"])
    print("Macro F1-score:", metrics_result["macro_f1"])
    print("Weighted F1-score:", metrics_result["weighted_f1"])

    print()
    print("Predictions saved:")
    print(predictions_path)

    print()
    print("Confusion matrix saved:")
    print(confusion_matrix_path)

    print()
    print("Class metrics saved:")
    print(class_metrics_path)

    print()
    print("Quadrant metrics saved:")
    print(quadrant_metrics_path)

    print()
    print("Figure saved:")
    print(figure_path)

    print()
    print("Report saved:")
    print(report_path)


if __name__ == "__main__":
    main()