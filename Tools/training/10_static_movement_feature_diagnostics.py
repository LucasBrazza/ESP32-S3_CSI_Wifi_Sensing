"""
Static Presence vs Movement Feature Diagnostics

Experiment 009

This script focuses only on the hardest part observed so far:

    static_presence vs movement

It uses the original feature dataset and recomputes Fisher Score ranking
only for these two classes. Then, it evaluates different top_k values
using file-based holdout validation.

Goal:
    Check whether the current feature selection is adequate for
    distinguishing stationary presence from movement.

Experiment folder:
    Tools/datasets/results/runs/009_static_movement_feature_diagnostics/
"""

import csv
import random
from collections import Counter

from Tools.common.config import (
    MODEL,
    MAX_TREE_DEPTH,
    MIN_SAMPLES_SPLIT,
)

from Tools.common.io_utils import load_pickle

from Tools.common.project_paths import (
    FEATURE_DATASET_FILE,
    RESULTS_DIR,
)

from Tools.preprocessing.feature_selection import (
    rank_features_by_fisher_score,
)

from Tools.classification import decision_tree


TEST_SIZE = 0.20
RANDOM_SEED = 42

EXPERIMENT_ID = "009"
EXPERIMENT_FOLDER_NAME = "009_static_movement_feature_diagnostics"

TOP_K_VALUES = [
    5,
    10,
    20,
    30,
    40,
    50,
    70,
    100,
    126,
]


# ================= DATASET HELPERS =================

def filter_static_vs_movement(dataset):
    filtered = []

    for sample in dataset:
        if sample["label"] in ["static_presence", "movement"]:
            filtered.append(sample)

    return filtered


def select_top_features_preserving_metadata(feature_dataset, ranking, top_k):
    selected_indices = []

    for item in ranking[:top_k]:
        selected_indices.append(item["feature_index"])

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

    return selected_dataset, selected_indices


def get_sample_group_id(sample):
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


def get_labels(dataset):
    labels = []

    preferred_order = [
        "static_presence",
        "movement",
    ]

    labels_found = []

    for sample in dataset:
        label = sample["label"]

        if label not in labels_found:
            labels_found.append(label)

    for label in preferred_order:
        if label in labels_found:
            labels.append(label)

    for label in labels_found:
        if label not in labels:
            labels.append(label)

    return labels


def predict_dataset(tree, dataset):
    predictions = []

    for sample in dataset:
        predicted_label = decision_tree.predict_one(
            tree,
            sample["features"],
        )

        true_label = sample["label"]

        predictions.append(
            {
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

    result = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }

    for item in per_class:
        label = item["label"]

        result[f"{label}_precision"] = item["precision"]
        result[f"{label}_recall"] = item["recall"]
        result[f"{label}_f1"] = item["f1_score"]
        result[f"{label}_support"] = item["support"]

    return result, per_class


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
        "results": tables_dir / "static_movement_topk_results.csv",
        "best_predictions": tables_dir / "best_predictions.csv",
        "best_confusion_matrix": tables_dir / "best_confusion_matrix.csv",
        "best_class_metrics": tables_dir / "best_class_metrics.csv",
        "best_report": reports_dir / "best_static_movement_config.txt",
        "experiment_notes": run_dir / "experiment_notes.txt",
    }


def save_results(results, output_path):
    fieldnames = list(results[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for item in results:
            writer.writerow(item)


def save_predictions(predictions, output_path):
    fieldnames = [
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

        for row_index, label in enumerate(labels):
            row = [label]

            for value in matrix[row_index]:
                row.append(value)

            writer.writerow(row)


def save_class_metrics(per_class_metrics, output_path):
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

        for item in per_class_metrics:
            writer.writerow(item)


def select_best_result(results):
    ordered_results = sorted(
        results,
        key=lambda item: (
            item["macro_f1"],
            item["accuracy"],
            -item["top_k"],
            -item["tree_nodes"],
        ),
        reverse=True,
    )

    return ordered_results[0]


def save_best_report(best_result, output_path):
    lines = []

    lines.append("BEST STATIC PRESENCE VS MOVEMENT CONFIGURATION")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Experiment: {EXPERIMENT_ID} - {EXPERIMENT_FOLDER_NAME}")
    lines.append("")
    lines.append("Selection criterion:")
    lines.append("Highest macro F1-score, then highest accuracy, then smaller top_k.")
    lines.append("")
    lines.append("Best configuration:")
    lines.append(f"- top_k: {best_result['top_k']}")
    lines.append(f"- max_depth: {best_result['max_depth']}")
    lines.append(f"- min_samples_split: {best_result['min_samples_split']}")
    lines.append("")
    lines.append("Validation metrics:")
    lines.append(f"- accuracy: {best_result['accuracy']:.6f}")
    lines.append(f"- macro_f1: {best_result['macro_f1']:.6f}")
    lines.append(f"- weighted_f1: {best_result['weighted_f1']:.6f}")
    lines.append("")
    lines.append("Per-class metrics:")
    lines.append(f"- static_presence recall: {best_result['static_presence_recall']:.6f}")
    lines.append(f"- static_presence f1: {best_result['static_presence_f1']:.6f}")
    lines.append(f"- movement recall: {best_result['movement_recall']:.6f}")
    lines.append(f"- movement f1: {best_result['movement_f1']:.6f}")
    lines.append("")
    lines.append("Tree complexity:")
    lines.append(f"- total nodes: {best_result['tree_nodes']}")
    lines.append(f"- decision nodes: {best_result['tree_decision_nodes']}")
    lines.append(f"- leaf nodes: {best_result['tree_leaf_nodes']}")
    lines.append(f"- actual max depth: {best_result['tree_actual_max_depth']}")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def save_experiment_notes(output_path, best_result):
    lines = []

    lines.append("Experiment 009 - Static presence vs movement feature diagnostics")
    lines.append("")
    lines.append("Description:")
    lines.append(
        "Diagnostic experiment focused only on distinguishing "
        "static_presence from movement."
    )
    lines.append("")
    lines.append("Motivation:")
    lines.append(
        "Previous experiments showed that the main limitation is not "
        "presence detection, but the separation between stationary "
        "presence and movement."
    )
    lines.append("")
    lines.append("Method:")
    lines.append("- Use only static_presence and movement samples")
    lines.append("- Recompute Fisher Score ranking for this binary task")
    lines.append("- Test multiple top_k values")
    lines.append("- Validate using file-based stratified holdout")
    lines.append("")
    lines.append("Best result:")
    lines.append(f"- top_k: {best_result['top_k']}")
    lines.append(f"- accuracy: {best_result['accuracy']:.6f}")
    lines.append(f"- macro F1-score: {best_result['macro_f1']:.6f}")
    lines.append(f"- weighted F1-score: {best_result['weighted_f1']:.6f}")
    lines.append(f"- static_presence F1-score: {best_result['static_presence_f1']:.6f}")
    lines.append(f"- movement F1-score: {best_result['movement_f1']:.6f}")
    lines.append("")
    lines.append("Next step:")
    lines.append(
        "If this binary task remains weak, the next improvement should "
        "focus on temporal/window features rather than only feature count."
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


# ================= MAIN =================

def main():
    print()
    print("Static Presence vs Movement Feature Diagnostics")
    print("=" * 70)

    full_dataset = load_pickle(FEATURE_DATASET_FILE)

    if not full_dataset:
        raise ValueError("Feature dataset is empty.")

    dataset = filter_static_vs_movement(full_dataset)

    if not dataset:
        raise ValueError("No static_presence or movement samples found.")

    labels = get_labels(dataset)

    print("Model:", MODEL)
    print("Validation method: File-based stratified holdout 80/20")
    print("Samples/windows:", len(dataset))
    print("Original features per sample:", len(dataset[0]["features"]))
    print("Labels:", labels)
    print("Max tree depth:", MAX_TREE_DEPTH)
    print("Min samples split:", MIN_SAMPLES_SPLIT)

    print()
    print("Ranking features using Fisher Score for static vs movement...")
    ranking = rank_features_by_fisher_score(dataset)

    valid_top_k_values = []

    for top_k in TOP_K_VALUES:
        if top_k <= len(ranking):
            valid_top_k_values.append(top_k)

    results = []
    artifacts_by_top_k = {}

    for index, top_k in enumerate(valid_top_k_values, start=1):
        print()
        print(f"[{index}/{len(valid_top_k_values)}] Evaluating top_k={top_k}")

        selected_dataset, selected_indices = select_top_features_preserving_metadata(
            dataset,
            ranking,
            top_k,
        )

        train_dataset, test_dataset, train_groups, test_groups = file_stratified_holdout_split(
            selected_dataset,
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

        metrics, per_class_metrics = compute_class_metrics(
            matrix,
            labels,
        )

        tree_complexity = analyze_tree(tree)

        result = {
            "top_k": top_k,
            "max_depth": MAX_TREE_DEPTH,
            "min_samples_split": MIN_SAMPLES_SPLIT,
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "train_files": len(train_groups),
            "test_files": len(test_groups),
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
            "weighted_f1": metrics["weighted_f1"],
            "static_presence_precision": metrics.get("static_presence_precision", 0.0),
            "static_presence_recall": metrics.get("static_presence_recall", 0.0),
            "static_presence_f1": metrics.get("static_presence_f1", 0.0),
            "movement_precision": metrics.get("movement_precision", 0.0),
            "movement_recall": metrics.get("movement_recall", 0.0),
            "movement_f1": metrics.get("movement_f1", 0.0),
            "tree_nodes": tree_complexity["nodes"],
            "tree_decision_nodes": tree_complexity["decision_nodes"],
            "tree_leaf_nodes": tree_complexity["leaf_nodes"],
            "tree_actual_max_depth": tree_complexity["max_depth"],
            "selected_indices": ";".join(str(item) for item in selected_indices),
        }

        results.append(result)

        artifacts_by_top_k[top_k] = {
            "predictions": predictions,
            "matrix": matrix,
            "per_class_metrics": per_class_metrics,
        }

        print(f"accuracy: {metrics['accuracy']:.6f}")
        print(f"macro_f1: {metrics['macro_f1']:.6f}")
        print(f"static_presence_f1: {metrics.get('static_presence_f1', 0.0):.6f}")
        print(f"movement_f1: {metrics.get('movement_f1', 0.0):.6f}")

    results = sorted(
        results,
        key=lambda item: (
            item["macro_f1"],
            item["accuracy"],
            -item["top_k"],
            -item["tree_nodes"],
        ),
        reverse=True,
    )

    best_result = select_best_result(results)
    best_top_k = best_result["top_k"]
    best_artifacts = artifacts_by_top_k[best_top_k]

    paths = get_experiment_paths()

    save_results(
        results,
        paths["results"],
    )

    save_predictions(
        best_artifacts["predictions"],
        paths["best_predictions"],
    )

    save_confusion_matrix(
        best_artifacts["matrix"],
        labels,
        paths["best_confusion_matrix"],
    )

    save_class_metrics(
        best_artifacts["per_class_metrics"],
        paths["best_class_metrics"],
    )

    save_best_report(
        best_result,
        paths["best_report"],
    )

    save_experiment_notes(
        paths["experiment_notes"],
        best_result,
    )

    print()
    print("Static vs Movement diagnostics finished.")
    print("=" * 70)
    print("Best configuration:")
    print(f"top_k: {best_result['top_k']}")
    print(f"accuracy: {best_result['accuracy']:.6f}")
    print(f"macro_f1: {best_result['macro_f1']:.6f}")
    print(f"weighted_f1: {best_result['weighted_f1']:.6f}")
    print(f"static_presence_f1: {best_result['static_presence_f1']:.6f}")
    print(f"movement_f1: {best_result['movement_f1']:.6f}")
    print(f"tree_nodes: {best_result['tree_nodes']}")
    print()
    print("Results saved to:")
    print(paths["run_dir"])


if __name__ == "__main__":
    main()