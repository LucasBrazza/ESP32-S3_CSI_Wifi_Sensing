"""
Binary Task Diagnostics

Experiment 007

This script evaluates two binary classification tasks using the current
selected feature dataset and file-based holdout validation.

Task A:
    empty vs presence

    where:
        empty -> empty
        static_presence -> presence
        movement -> presence

Task B:
    static_presence vs movement

    using only samples originally labeled as static_presence or movement.

Goal:
    Diagnose whether the current limitation is related to detecting
    presence or to distinguishing stationary presence from movement.

Experiment folder:
    Tools/datasets/results/runs/007_binary_task_diagnostics/
"""

import csv
import random
from collections import Counter

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


TEST_SIZE = 0.20
RANDOM_SEED = 42

EXPERIMENT_ID = "007"
EXPERIMENT_FOLDER_NAME = "007_binary_task_diagnostics"


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


def build_empty_vs_presence_dataset(dataset):
    binary_dataset = []

    for sample in dataset:
        original_label = sample["label"]

        if original_label == "empty":
            new_label = "empty"
        else:
            new_label = "presence"

        binary_dataset.append(
            clone_sample_with_label(sample, new_label)
        )

    return binary_dataset


def build_static_vs_movement_dataset(dataset):
    binary_dataset = []

    for sample in dataset:
        original_label = sample["label"]

        if original_label in ["static_presence", "movement"]:
            binary_dataset.append(
                clone_sample_with_label(sample, original_label)
            )

    return binary_dataset


def get_labels(dataset):
    labels = []

    for sample in dataset:
        label = sample["label"]

        if label not in labels:
            labels.append(label)

    return labels


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

    for sample in dataset:
        counter[sample["label"]] += 1

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
                "original_label": sample.get("original_label", ""),
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
    reports_dir = run_dir / "reports"

    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "tables_dir": tables_dir,
        "reports_dir": reports_dir,
        "summary": tables_dir / "binary_task_summary.csv",
        "report": reports_dir / "binary_task_diagnostics_report.txt",
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


def save_summary(task_results, output_path):
    fieldnames = [
        "task",
        "description",
        "train_samples",
        "test_samples",
        "train_files",
        "test_files",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "tree_nodes",
        "tree_decision_nodes",
        "tree_leaf_nodes",
        "tree_actual_max_depth",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for item in task_results:
            writer.writerow(
                {
                    "task": item["task"],
                    "description": item["description"],
                    "train_samples": item["train_samples"],
                    "test_samples": item["test_samples"],
                    "train_files": item["train_files"],
                    "test_files": item["test_files"],
                    "accuracy": item["metrics"]["accuracy"],
                    "macro_precision": item["metrics"]["macro_precision"],
                    "macro_recall": item["metrics"]["macro_recall"],
                    "macro_f1": item["metrics"]["macro_f1"],
                    "weighted_precision": item["metrics"]["weighted_precision"],
                    "weighted_recall": item["metrics"]["weighted_recall"],
                    "weighted_f1": item["metrics"]["weighted_f1"],
                    "tree_nodes": item["tree_complexity"]["nodes"],
                    "tree_decision_nodes": item["tree_complexity"]["decision_nodes"],
                    "tree_leaf_nodes": item["tree_complexity"]["leaf_nodes"],
                    "tree_actual_max_depth": item["tree_complexity"]["max_depth"],
                }
            )


def save_report(task_results, output_path):
    lines = []

    lines.append("BINARY TASK DIAGNOSTICS REPORT")
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

    for result in task_results:
        lines.append(result["task"])
        lines.append("-" * 70)
        lines.append(result["description"])
        lines.append("")
        lines.append("Dataset split:")
        lines.append(f"- Train samples/windows: {result['train_samples']}")
        lines.append(f"- Test samples/windows: {result['test_samples']}")
        lines.append(f"- Train files/groups: {result['train_files']}")
        lines.append(f"- Test files/groups: {result['test_files']}")
        lines.append("")
        lines.append("Metrics:")
        lines.append(f"- Accuracy: {result['metrics']['accuracy']:.6f}")
        lines.append(f"- Macro precision: {result['metrics']['macro_precision']:.6f}")
        lines.append(f"- Macro recall: {result['metrics']['macro_recall']:.6f}")
        lines.append(f"- Macro F1-score: {result['metrics']['macro_f1']:.6f}")
        lines.append(f"- Weighted F1-score: {result['metrics']['weighted_f1']:.6f}")
        lines.append("")
        lines.append("Per-class metrics:")

        for item in result["metrics"]["per_class"]:
            lines.append(
                f"- {item['label']}: "
                f"precision={item['precision']:.6f}, "
                f"recall={item['recall']:.6f}, "
                f"f1={item['f1_score']:.6f}, "
                f"support={item['support']}"
            )

        lines.append("")
        lines.append("Confusion matrix:")
        labels = result["labels"]
        matrix = result["matrix"]

        header = " " * 20

        for label in labels:
            header += f"{label:>18}"

        lines.append(header)

        for row_index, label in enumerate(labels):
            row = f"{label:<20}"

            for value in matrix[row_index]:
                row += f"{value:>18}"

            lines.append(row)

        lines.append("")
        lines.append("Tree complexity:")
        lines.append(f"- Total nodes: {result['tree_complexity']['nodes']}")
        lines.append(f"- Decision nodes: {result['tree_complexity']['decision_nodes']}")
        lines.append(f"- Leaf nodes: {result['tree_complexity']['leaf_nodes']}")
        lines.append(f"- Actual max depth: {result['tree_complexity']['max_depth']}")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def save_experiment_notes(task_results, output_path):
    task_a = task_results[0]
    task_b = task_results[1]

    lines = []

    lines.append("Experiment 007 - Binary task diagnostics")
    lines.append("")
    lines.append("Description:")
    lines.append(
        "Diagnostic experiment separating the original three-class problem "
        "into two binary classification tasks."
    )
    lines.append("")
    lines.append("Task A:")
    lines.append("- empty vs presence")
    lines.append("- presence combines static_presence and movement")
    lines.append("")
    lines.append("Task B:")
    lines.append("- static_presence vs movement")
    lines.append("- only samples with presence are used")
    lines.append("")
    lines.append("Motivation:")
    lines.append(
        "Experiment 006 showed improved detection of the empty class, but "
        "the model still confused static_presence and movement. This "
        "experiment checks whether the main limitation is presence detection "
        "or motion discrimination."
    )
    lines.append("")
    lines.append("Results:")
    lines.append(
        f"- Task A accuracy: {task_a['metrics']['accuracy']:.6f}, "
        f"macro F1-score: {task_a['metrics']['macro_f1']:.6f}"
    )
    lines.append(
        f"- Task B accuracy: {task_b['metrics']['accuracy']:.6f}, "
        f"macro F1-score: {task_b['metrics']['macro_f1']:.6f}"
    )
    lines.append("")
    lines.append("Main observation:")
    lines.append(
        "If Task A performs clearly better than Task B, the main limitation "
        "is not detecting presence, but distinguishing stationary presence "
        "from movement."
    )
    lines.append("")
    lines.append("Next step:")
    lines.append(
        "Use these results to decide whether a hierarchical classifier is "
        "worth testing."
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


# ================= TASK EXECUTION =================

def run_binary_task(task_name, description, dataset):
    labels = get_labels(dataset)

    train_dataset, test_dataset, train_groups, test_groups = file_stratified_holdout_split(
        dataset,
        test_size=TEST_SIZE,
        seed=RANDOM_SEED,
    )

    tree = decision_tree.build_tree(
        train_dataset,
        max_depth=MAX_TREE_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
    )

    predictions = predict_dataset(
        tree,
        test_dataset,
    )

    matrix = build_confusion_matrix(
        predictions,
        labels,
    )

    metrics = compute_class_metrics(
        matrix,
        labels,
    )

    quadrant_metrics = compute_quadrant_metrics(
        predictions,
    )

    tree_complexity = analyze_tree(tree)

    return {
        "task": task_name,
        "description": description,
        "labels": labels,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "train_files": len(train_groups),
        "test_files": len(test_groups),
        "predictions": predictions,
        "matrix": matrix,
        "metrics": metrics,
        "quadrant_metrics": quadrant_metrics,
        "tree_complexity": tree_complexity,
    }


def save_task_artifacts(task_result, paths, prefix):
    save_predictions(
        task_result["predictions"],
        paths["tables_dir"] / f"{prefix}_predictions.csv",
    )

    save_confusion_matrix(
        task_result["matrix"],
        task_result["labels"],
        paths["tables_dir"] / f"{prefix}_confusion_matrix.csv",
    )

    save_class_metrics(
        task_result["metrics"],
        paths["tables_dir"] / f"{prefix}_class_metrics.csv",
    )

    save_quadrant_metrics(
        task_result["quadrant_metrics"],
        paths["tables_dir"] / f"{prefix}_quadrant_metrics.csv",
    )


# ================= MAIN =================

def main():
    print()
    print("Binary Task Diagnostics")
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

    empty_vs_presence_dataset = build_empty_vs_presence_dataset(dataset)
    static_vs_movement_dataset = build_static_vs_movement_dataset(dataset)

    task_a = run_binary_task(
        "Task A - empty vs presence",
        "Binary classification between empty and presence.",
        empty_vs_presence_dataset,
    )

    task_b = run_binary_task(
        "Task B - static_presence vs movement",
        "Binary classification between stationary presence and movement.",
        static_vs_movement_dataset,
    )

    task_results = [
        task_a,
        task_b,
    ]

    paths = get_experiment_paths()

    save_task_artifacts(
        task_a,
        paths,
        "task_a_empty_vs_presence",
    )

    save_task_artifacts(
        task_b,
        paths,
        "task_b_static_vs_movement",
    )

    save_summary(
        task_results,
        paths["summary"],
    )

    save_report(
        task_results,
        paths["report"],
    )

    save_experiment_notes(
        task_results,
        paths["experiment_notes"],
    )

    print()
    print("Task A - empty vs presence")
    print("-" * 70)
    print(f"Accuracy: {task_a['metrics']['accuracy']:.6f}")
    print(f"Macro F1-score: {task_a['metrics']['macro_f1']:.6f}")
    print(f"Weighted F1-score: {task_a['metrics']['weighted_f1']:.6f}")

    print()
    print("Task B - static_presence vs movement")
    print("-" * 70)
    print(f"Accuracy: {task_b['metrics']['accuracy']:.6f}")
    print(f"Macro F1-score: {task_b['metrics']['macro_f1']:.6f}")
    print(f"Weighted F1-score: {task_b['metrics']['weighted_f1']:.6f}")

    print()
    print("Results saved to:")
    print(paths["run_dir"])


if __name__ == "__main__":
    main()