"""
Hierarchical Classifier Validation

Experiment 008

This script evaluates a hierarchical decision tree classifier using
file-based holdout validation.

The hierarchical classifier has two stages:

    Stage A:
        empty vs presence

        where:
            empty -> empty
            static_presence -> presence
            movement -> presence

    Stage B:
        static_presence vs movement

        using only samples with presence.

Final prediction rule:

    if Stage A predicts empty:
        final prediction = empty

    if Stage A predicts presence:
        final prediction = Stage B prediction

Motivation:
    Experiment 007 showed that detecting presence is easier than
    distinguishing static_presence from movement. This experiment tests
    whether a hierarchical classifier improves the final three-class
    classification problem.

Experiment folder:
    Tools/datasets/results/runs/008_hierarchical_classifier_validation/
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

EXPERIMENT_ID = "008"
EXPERIMENT_FOLDER_NAME = "008_hierarchical_classifier_validation"


# ================= DATASET HELPERS =================

def get_sample_group_id(sample):
    source_file = sample.get("source_file", "")

    if source_file:
        return str(source_file)

    label = sample.get("label", "unknown_label")
    quadrant = sample.get("quadrant", "unknown_quadrant")
    file_name = sample.get("file_name", "unknown_file")

    return f"{label}|{quadrant}|{file_name}"


def clone_sample_with_label(sample, new_label):
    cloned_sample = {}

    for key in sample:
        cloned_sample[key] = sample[key]

    cloned_sample["original_label"] = sample["label"]
    cloned_sample["label"] = new_label

    return cloned_sample


def convert_to_empty_vs_presence(dataset):
    converted_dataset = []

    for sample in dataset:
        if sample["label"] == "empty":
            new_label = "empty"
        else:
            new_label = "presence"

        converted_dataset.append(
            clone_sample_with_label(sample, new_label)
        )

    return converted_dataset


def filter_static_vs_movement(dataset):
    filtered_dataset = []

    for sample in dataset:
        if sample["label"] in ["static_presence", "movement"]:
            filtered_dataset.append(sample)

    return filtered_dataset


def get_labels(dataset, preferred_order=None):
    labels_found = []

    for item in dataset:
        label = item["label"]

        if label not in labels_found:
            labels_found.append(label)

    if preferred_order is None:
        return labels_found

    ordered_labels = []

    for label in preferred_order:
        if label in labels_found:
            ordered_labels.append(label)

    for label in labels_found:
        if label not in ordered_labels:
            ordered_labels.append(label)

    return ordered_labels


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
    Split the original three-class dataset by acquisition file.

    This keeps all windows from the same file together and preserves
    the original class distribution as much as possible.

    The split is done before building the hierarchical labels so that
    the final comparison remains compatible with previous file-based
    experiments.
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


# ================= HIERARCHICAL PREDICTION =================

def predict_stage_dataset(tree, dataset):
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
                "original_label": sample.get("original_label", sample["label"]),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": true_label == predicted_label,
            }
        )

    return predictions


def predict_hierarchical(stage_a_tree, stage_b_tree, test_dataset):
    predictions = []

    for sample_index, sample in enumerate(test_dataset):
        stage_a_prediction = decision_tree.predict_one(
            stage_a_tree,
            sample["features"],
        )

        if stage_a_prediction == "empty":
            final_prediction = "empty"
            stage_b_prediction = ""
        else:
            stage_b_prediction = decision_tree.predict_one(
                stage_b_tree,
                sample["features"],
            )
            final_prediction = stage_b_prediction

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
                "stage_a_prediction": stage_a_prediction,
                "stage_b_prediction": stage_b_prediction,
                "predicted_label": final_prediction,
                "correct": true_label == final_prediction,
            }
        )

    return predictions


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
        "final_predictions": tables_dir / "hierarchical_predictions.csv",
        "final_confusion_matrix": tables_dir / "confusion_matrix.csv",
        "final_class_metrics": tables_dir / "class_metrics.csv",
        "final_quadrant_metrics": tables_dir / "quadrant_metrics.csv",
        "stage_a_predictions": tables_dir / "stage_a_empty_vs_presence_predictions.csv",
        "stage_a_confusion_matrix": tables_dir / "stage_a_confusion_matrix.csv",
        "stage_a_class_metrics": tables_dir / "stage_a_class_metrics.csv",
        "stage_b_predictions": tables_dir / "stage_b_static_vs_movement_predictions.csv",
        "stage_b_confusion_matrix": tables_dir / "stage_b_confusion_matrix.csv",
        "stage_b_class_metrics": tables_dir / "stage_b_class_metrics.csv",
        "confusion_matrix_plot": figures_dir / "confusion_matrix.png",
        "report": reports_dir / "hierarchical_classifier_report.txt",
        "experiment_notes": run_dir / "experiment_notes.txt",
    }


# ================= SAVE OUTPUTS =================

def save_final_predictions(predictions, output_path):
    fieldnames = [
        "sample_index",
        "sample_id",
        "group_id",
        "quadrant",
        "file_name",
        "source_file",
        "window_index",
        "true_label",
        "stage_a_prediction",
        "stage_b_prediction",
        "predicted_label",
        "correct",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for item in predictions:
            writer.writerow(item)


def save_stage_predictions(predictions, output_path):
    fieldnames = [
        "sample_index",
        "sample_id",
        "group_id",
        "quadrant",
        "file_name",
        "source_file",
        "window_index",
        "original_label",
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

        for row_index, label in enumerate(labels):
            row = [label]

            for value in matrix[row_index]:
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


def save_confusion_matrix_plot(matrix, labels, output_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)

    plt.title("Confusion Matrix - Hierarchical Classifier")
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
    stage_a_metrics,
    stage_b_metrics,
    final_metrics,
    final_matrix,
    final_labels,
    final_quadrant_metrics,
    stage_a_complexity,
    stage_b_complexity,
):
    total_class_count = count_by_label(dataset)
    train_class_count = count_by_label(train_dataset)
    test_class_count = count_by_label(test_dataset)

    total_group_count = count_groups_by_label(train_groups + test_groups)
    train_group_count = count_groups_by_label(train_groups)
    test_group_count = count_groups_by_label(test_groups)

    lines = []

    lines.append("HIERARCHICAL CLASSIFIER VALIDATION REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Experiment: {EXPERIMENT_ID} - {EXPERIMENT_FOLDER_NAME}")
    lines.append(f"Model: {MODEL}")
    lines.append("Validation method: File-based stratified holdout 80/20")
    lines.append(f"Random seed: {RANDOM_SEED}")
    lines.append("")
    lines.append("Current configuration:")
    lines.append(f"- Top-K selected features: {TOP_K_FEATURES}")
    lines.append(f"- Max tree depth: {MAX_TREE_DEPTH}")
    lines.append(f"- Min samples split: {MIN_SAMPLES_SPLIT}")
    lines.append("")
    lines.append("Hierarchical structure:")
    lines.append("- Stage A: empty vs presence")
    lines.append("- Stage B: static_presence vs movement")
    lines.append("")

    lines.append("Dataset:")
    lines.append(f"- Total samples/windows: {len(dataset)}")
    lines.append(f"- Train samples/windows: {len(train_dataset)}")
    lines.append(f"- Test samples/windows: {len(test_dataset)}")
    lines.append(f"- Total files/groups: {len(train_groups) + len(test_groups)}")
    lines.append(f"- Train files/groups: {len(train_groups)}")
    lines.append(f"- Test files/groups: {len(test_groups)}")
    lines.append(f"- Features per sample: {len(dataset[0]['features'])}")
    lines.append("")

    lines.append("Class distribution by samples/windows")
    lines.append("-" * 70)
    lines.append(f"{'Class':<20}{'Total':>10}{'Train':>10}{'Test':>10}")

    for label in final_labels:
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

    for label in final_labels:
        lines.append(
            f"{label:<20}"
            f"{total_group_count[label]:>10}"
            f"{train_group_count[label]:>10}"
            f"{test_group_count[label]:>10}"
        )

    lines.append("")
    lines.append("Stage A metrics - empty vs presence")
    lines.append("-" * 70)
    lines.append(f"Accuracy: {stage_a_metrics['accuracy']:.6f}")
    lines.append(f"Macro F1-score: {stage_a_metrics['macro_f1']:.6f}")
    lines.append(f"Weighted F1-score: {stage_a_metrics['weighted_f1']:.6f}")

    for item in stage_a_metrics["per_class"]:
        lines.append(
            f"{item['label']}: "
            f"precision={item['precision']:.6f} | "
            f"recall={item['recall']:.6f} | "
            f"f1={item['f1_score']:.6f} | "
            f"support={item['support']}"
        )

    lines.append("")
    lines.append("Stage B metrics - static_presence vs movement")
    lines.append("-" * 70)
    lines.append(f"Accuracy: {stage_b_metrics['accuracy']:.6f}")
    lines.append(f"Macro F1-score: {stage_b_metrics['macro_f1']:.6f}")
    lines.append(f"Weighted F1-score: {stage_b_metrics['weighted_f1']:.6f}")

    for item in stage_b_metrics["per_class"]:
        lines.append(
            f"{item['label']}: "
            f"precision={item['precision']:.6f} | "
            f"recall={item['recall']:.6f} | "
            f"f1={item['f1_score']:.6f} | "
            f"support={item['support']}"
        )

    lines.append("")
    lines.append("Final hierarchical metrics")
    lines.append("-" * 70)
    lines.append(f"Accuracy: {final_metrics['accuracy']:.6f}")
    lines.append(f"Macro precision: {final_metrics['macro_precision']:.6f}")
    lines.append(f"Macro recall: {final_metrics['macro_recall']:.6f}")
    lines.append(f"Macro F1-score: {final_metrics['macro_f1']:.6f}")
    lines.append(f"Weighted precision: {final_metrics['weighted_precision']:.6f}")
    lines.append(f"Weighted recall: {final_metrics['weighted_recall']:.6f}")
    lines.append(f"Weighted F1-score: {final_metrics['weighted_f1']:.6f}")

    lines.append("")
    lines.append("Final per-class metrics")
    lines.append("-" * 70)

    for item in final_metrics["per_class"]:
        lines.append(
            f"{item['label']}: "
            f"precision={item['precision']:.6f} | "
            f"recall={item['recall']:.6f} | "
            f"f1={item['f1_score']:.6f} | "
            f"support={item['support']}"
        )

    lines.append("")
    lines.append("Final per-quadrant metrics")
    lines.append("-" * 70)

    for item in final_quadrant_metrics:
        lines.append(
            f"{item['quadrant']}: "
            f"accuracy={item['accuracy']:.6f} | "
            f"correct={item['correct']} | "
            f"support={item['support']}"
        )

    lines.append("")
    lines.append("Final confusion matrix")
    lines.append("-" * 70)
    lines.append("Rows = true labels | Columns = predicted labels")
    lines.append("")

    header = " " * 20

    for label in final_labels:
        header += f"{label:>18}"

    lines.append(header)

    for row_index, label in enumerate(final_labels):
        row = f"{label:<20}"

        for value in final_matrix[row_index]:
            row += f"{value:>18}"

        lines.append(row)

    lines.append("")
    lines.append("Tree complexity")
    lines.append("-" * 70)
    lines.append("Stage A:")
    lines.append(f"- Total nodes: {stage_a_complexity['nodes']}")
    lines.append(f"- Decision nodes: {stage_a_complexity['decision_nodes']}")
    lines.append(f"- Leaf nodes: {stage_a_complexity['leaf_nodes']}")
    lines.append(f"- Actual max depth: {stage_a_complexity['max_depth']}")
    lines.append("")
    lines.append("Stage B:")
    lines.append(f"- Total nodes: {stage_b_complexity['nodes']}")
    lines.append(f"- Decision nodes: {stage_b_complexity['decision_nodes']}")
    lines.append(f"- Leaf nodes: {stage_b_complexity['leaf_nodes']}")
    lines.append(f"- Actual max depth: {stage_b_complexity['max_depth']}")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def save_experiment_notes(output_path, final_metrics):
    lines = []

    lines.append("Experiment 008 - Hierarchical classifier validation")
    lines.append("")
    lines.append("Description:")
    lines.append(
        "Validation of a hierarchical decision tree classifier using "
        "file-based holdout."
    )
    lines.append("")
    lines.append("Structure:")
    lines.append("- Stage A: empty vs presence")
    lines.append("- Stage B: static_presence vs movement")
    lines.append("")
    lines.append("Motivation:")
    lines.append(
        "Experiment 007 showed that empty vs presence classification was "
        "considerably easier than static_presence vs movement. This "
        "experiment evaluates whether separating the problem into two "
        "decision stages improves the final three-class classification."
    )
    lines.append("")
    lines.append("Configuration:")
    lines.append(f"- Top-K selected features: {TOP_K_FEATURES}")
    lines.append(f"- Max tree depth: {MAX_TREE_DEPTH}")
    lines.append(f"- Min samples split: {MIN_SAMPLES_SPLIT}")
    lines.append("- Validation method: file-based stratified holdout 80/20")
    lines.append(f"- Random seed: {RANDOM_SEED}")
    lines.append("")
    lines.append("Final results:")
    lines.append(f"- Accuracy: {final_metrics['accuracy']:.6f}")
    lines.append(f"- Macro F1-score: {final_metrics['macro_f1']:.6f}")
    lines.append(f"- Weighted F1-score: {final_metrics['weighted_f1']:.6f}")
    lines.append("")
    lines.append("Next step:")
    lines.append(
        "Compare this hierarchical approach with experiment 006. If it "
        "does not improve the result, the next focus should be feature "
        "engineering or window configuration for motion discrimination."
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def update_experiment_index(final_metrics):
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
        "description": "Hierarchical decision tree validation",
        "model": MODEL,
        "validation_method": "File-based stratified holdout 80/20",
        "top_k": TOP_K_FEATURES,
        "max_depth": MAX_TREE_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "accuracy": f"{final_metrics['accuracy']:.6f}",
        "macro_f1": f"{final_metrics['macro_f1']:.6f}",
        "weighted_f1": f"{final_metrics['weighted_f1']:.6f}",
        "main_observation": (
            "Two-stage classifier: empty vs presence followed by "
            "static_presence vs movement"
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
    print("Hierarchical Classifier Validation")
    print("=" * 70)

    dataset = load_pickle(SELECTED_FEATURE_DATASET_FILE)

    if not dataset:
        raise ValueError("Selected feature dataset is empty.")

    print("Model:", MODEL)
    print("Validation method: File-based stratified holdout 80/20")
    print("Samples/windows:", len(dataset))
    print("Features per sample:", len(dataset[0]["features"]))
    print("Top-K selected features:", TOP_K_FEATURES)
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

    # Stage A datasets
    stage_a_train_dataset = convert_to_empty_vs_presence(train_dataset)
    stage_a_test_dataset = convert_to_empty_vs_presence(test_dataset)

    # Stage B datasets
    stage_b_train_dataset = filter_static_vs_movement(train_dataset)
    stage_b_test_dataset = filter_static_vs_movement(test_dataset)

    # Train Stage A
    stage_a_tree = decision_tree.build_tree(
        stage_a_train_dataset,
        max_depth=MAX_TREE_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
    )

    # Train Stage B
    stage_b_tree = decision_tree.build_tree(
        stage_b_train_dataset,
        max_depth=MAX_TREE_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
    )

    # Stage A diagnostics
    stage_a_predictions = predict_stage_dataset(
        stage_a_tree,
        stage_a_test_dataset,
    )

    stage_a_labels = get_labels(
        stage_a_test_dataset,
        preferred_order=["empty", "presence"],
    )

    stage_a_matrix = build_confusion_matrix(
        stage_a_predictions,
        stage_a_labels,
    )

    stage_a_metrics = compute_class_metrics(
        stage_a_matrix,
        stage_a_labels,
    )

    # Stage B diagnostics
    stage_b_predictions = predict_stage_dataset(
        stage_b_tree,
        stage_b_test_dataset,
    )

    stage_b_labels = get_labels(
        stage_b_test_dataset,
        preferred_order=["static_presence", "movement"],
    )

    stage_b_matrix = build_confusion_matrix(
        stage_b_predictions,
        stage_b_labels,
    )

    stage_b_metrics = compute_class_metrics(
        stage_b_matrix,
        stage_b_labels,
    )

    # Final hierarchical prediction
    final_predictions = predict_hierarchical(
        stage_a_tree,
        stage_b_tree,
        test_dataset,
    )

    final_labels = get_labels(
        test_dataset,
        preferred_order=CLASS_ORDER,
    )

    final_matrix = build_confusion_matrix(
        final_predictions,
        final_labels,
    )

    final_metrics = compute_class_metrics(
        final_matrix,
        final_labels,
    )

    final_quadrant_metrics = compute_quadrant_metrics(
        final_predictions,
    )

    stage_a_complexity = analyze_tree(stage_a_tree)
    stage_b_complexity = analyze_tree(stage_b_tree)

    paths = get_experiment_paths()

    save_final_predictions(
        final_predictions,
        paths["final_predictions"],
    )

    save_confusion_matrix(
        final_matrix,
        final_labels,
        paths["final_confusion_matrix"],
    )

    save_class_metrics(
        final_metrics,
        paths["final_class_metrics"],
    )

    save_quadrant_metrics(
        final_quadrant_metrics,
        paths["final_quadrant_metrics"],
    )

    save_stage_predictions(
        stage_a_predictions,
        paths["stage_a_predictions"],
    )

    save_confusion_matrix(
        stage_a_matrix,
        stage_a_labels,
        paths["stage_a_confusion_matrix"],
    )

    save_class_metrics(
        stage_a_metrics,
        paths["stage_a_class_metrics"],
    )

    save_stage_predictions(
        stage_b_predictions,
        paths["stage_b_predictions"],
    )

    save_confusion_matrix(
        stage_b_matrix,
        stage_b_labels,
        paths["stage_b_confusion_matrix"],
    )

    save_class_metrics(
        stage_b_metrics,
        paths["stage_b_class_metrics"],
    )

    save_confusion_matrix_plot(
        final_matrix,
        final_labels,
        paths["confusion_matrix_plot"],
    )

    save_report(
        paths["report"],
        dataset,
        train_dataset,
        test_dataset,
        train_groups,
        test_groups,
        stage_a_metrics,
        stage_b_metrics,
        final_metrics,
        final_matrix,
        final_labels,
        final_quadrant_metrics,
        stage_a_complexity,
        stage_b_complexity,
    )

    save_experiment_notes(
        paths["experiment_notes"],
        final_metrics,
    )

    update_experiment_index(final_metrics)

    print()
    print("Stage A - empty vs presence")
    print("-" * 70)
    print(f"Accuracy: {stage_a_metrics['accuracy']:.6f}")
    print(f"Macro F1-score: {stage_a_metrics['macro_f1']:.6f}")
    print(f"Weighted F1-score: {stage_a_metrics['weighted_f1']:.6f}")

    print()
    print("Stage B - static_presence vs movement")
    print("-" * 70)
    print(f"Accuracy: {stage_b_metrics['accuracy']:.6f}")
    print(f"Macro F1-score: {stage_b_metrics['macro_f1']:.6f}")
    print(f"Weighted F1-score: {stage_b_metrics['weighted_f1']:.6f}")

    print()
    print("Final hierarchical classifier")
    print("-" * 70)
    print(f"Accuracy: {final_metrics['accuracy']:.6f}")
    print(f"Macro F1-score: {final_metrics['macro_f1']:.6f}")
    print(f"Weighted F1-score: {final_metrics['weighted_f1']:.6f}")

    print()
    print("Results saved to:")
    print(paths["run_dir"])


if __name__ == "__main__":
    main()