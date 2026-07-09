"""
Decision Tree Tuning Experiment

This script runs a controlled tuning experiment for the decision tree
classifier.

The goal is to evaluate how max_depth and min_samples_split affect:

    - accuracy
    - macro F1-score
    - weighted F1-score
    - recall and F1-score per class
    - tree size, considering future embedded deployment

The experiment uses the same validation strategy adopted in the
baseline:

    Stratified holdout 80/20
    Random seed = 42

Results are saved as a new experiment run:

    Tools/datasets/results/runs/002_tree_tuning_depth_min_samples/
"""

import csv
import random
from collections import Counter

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

EXPERIMENT_ID = "002"
EXPERIMENT_FOLDER_NAME = "002_tree_tuning_depth_min_samples"

MAX_DEPTH_VALUES = [
    3,
    4,
    5,
    6,
    8,
]

MIN_SAMPLES_SPLIT_VALUES = [
    2,
    5,
    10,
    20,
]


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


def predict_dataset(tree, dataset):
    predictions = []

    for sample in dataset:
        predicted_label = decision_tree.predict_one(
            tree,
            sample["features"],
        )

        predictions.append(
            {
                "true_label": sample["label"],
                "predicted_label": predicted_label,
                "quadrant": sample.get("quadrant", "unknown"),
                "file_name": sample.get("file_name", ""),
                "window_index": sample.get("window_index", ""),
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
    metrics = []

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

        metrics.append(
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

    result = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }

    for item in metrics:
        label = item["label"]

        result[f"{label}_precision"] = item["precision"]
        result[f"{label}_recall"] = item["recall"]
        result[f"{label}_f1"] = item["f1_score"]
        result[f"{label}_support"] = item["support"]

    return result


def analyze_tree(tree):
    """
    Compute simple tree complexity metrics.

    These values are relevant because the final model should later be
    ported to an embedded environment.
    """
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


def count_by_label(dataset):
    counter = Counter()

    for item in dataset:
        counter[item["label"]] += 1

    return counter


# ================= EXPERIMENT OUTPUTS =================

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
        "tuning_results": tables_dir / "decision_tree_tuning_results.csv",
        "best_result": reports_dir / "best_decision_tree_config.txt",
        "experiment_notes": run_dir / "experiment_notes.txt",
    }


def save_tuning_results(results, output_path):
    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for item in results:
            writer.writerow(item)


def select_best_result(results):
    """
    Select the best configuration.

    Priority:
        1. Higher macro F1-score
        2. Higher accuracy
        3. Smaller number of nodes

    Macro F1 is prioritized because the classes are not perfectly
    balanced and the baseline showed poor behavior for empty.
    """
    ordered_results = sorted(
        results,
        key=lambda item: (
            item["macro_f1"],
            item["accuracy"],
            -item["tree_nodes"],
        ),
        reverse=True,
    )

    return ordered_results[0]


def save_best_result(best_result, output_path):
    lines = []

    lines.append("BEST DECISION TREE CONFIGURATION")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Experiment: {EXPERIMENT_ID} - {EXPERIMENT_FOLDER_NAME}")
    lines.append("")
    lines.append("Selection criterion:")
    lines.append("Highest macro F1-score, then highest accuracy, then smaller tree.")
    lines.append("")
    lines.append("Best parameters:")
    lines.append(f"- max_depth: {best_result['max_depth']}")
    lines.append(f"- min_samples_split: {best_result['min_samples_split']}")
    lines.append("")
    lines.append("Validation metrics:")
    lines.append(f"- accuracy: {best_result['accuracy']:.6f}")
    lines.append(f"- macro_f1: {best_result['macro_f1']:.6f}")
    lines.append(f"- weighted_f1: {best_result['weighted_f1']:.6f}")
    lines.append("")
    lines.append("Per-class F1-score:")
    lines.append(f"- empty: {best_result['empty_f1']:.6f}")
    lines.append(f"- static_presence: {best_result['static_presence_f1']:.6f}")
    lines.append(f"- movement: {best_result['movement_f1']:.6f}")
    lines.append("")
    lines.append("Per-class recall:")
    lines.append(f"- empty: {best_result['empty_recall']:.6f}")
    lines.append(f"- static_presence: {best_result['static_presence_recall']:.6f}")
    lines.append(f"- movement: {best_result['movement_recall']:.6f}")
    lines.append("")
    lines.append("Tree complexity:")
    lines.append(f"- total nodes: {best_result['tree_nodes']}")
    lines.append(f"- decision nodes: {best_result['tree_decision_nodes']}")
    lines.append(f"- leaf nodes: {best_result['tree_leaf_nodes']}")
    lines.append(f"- actual max depth: {best_result['tree_actual_max_depth']}")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def save_experiment_notes(
    output_path,
    dataset,
    train_dataset,
    test_dataset,
    best_result,
):
    total_class_count = count_by_label(dataset)
    train_class_count = count_by_label(train_dataset)
    test_class_count = count_by_label(test_dataset)

    lines = []

    lines.append("Experiment 002 - Decision tree tuning")
    lines.append("")
    lines.append("Description:")
    lines.append(
        "Controlled tuning experiment for the decision tree classifier, "
        "varying max_depth and min_samples_split."
    )
    lines.append("")
    lines.append("Validation:")
    lines.append("- Stratified holdout 80/20")
    lines.append(f"- Random seed: {RANDOM_SEED}")
    lines.append("")
    lines.append("Dataset:")
    lines.append(f"- Total samples: {len(dataset)}")
    lines.append(f"- Train samples: {len(train_dataset)}")
    lines.append(f"- Test samples: {len(test_dataset)}")
    lines.append(f"- Features per sample: {len(dataset[0]['features'])}")
    lines.append("")
    lines.append("Class distribution:")
    lines.append(f"{'Class':<20}{'Total':>10}{'Train':>10}{'Test':>10}")

    labels = get_labels(dataset)

    for label in labels:
        lines.append(
            f"{label:<20}"
            f"{total_class_count[label]:>10}"
            f"{train_class_count[label]:>10}"
            f"{test_class_count[label]:>10}"
        )

    lines.append("")
    lines.append("Tested values:")
    lines.append(f"- max_depth: {MAX_DEPTH_VALUES}")
    lines.append(f"- min_samples_split: {MIN_SAMPLES_SPLIT_VALUES}")
    lines.append("")
    lines.append("Best result:")
    lines.append(f"- max_depth: {best_result['max_depth']}")
    lines.append(f"- min_samples_split: {best_result['min_samples_split']}")
    lines.append(f"- accuracy: {best_result['accuracy']:.6f}")
    lines.append(f"- macro F1-score: {best_result['macro_f1']:.6f}")
    lines.append(f"- weighted F1-score: {best_result['weighted_f1']:.6f}")
    lines.append(f"- empty recall: {best_result['empty_recall']:.6f}")
    lines.append(f"- empty F1-score: {best_result['empty_f1']:.6f}")
    lines.append("")
    lines.append("Main observation:")
    lines.append(
        "This experiment compares different tree sizes while preserving "
        "the same dataset and validation split used in the baseline. "
        "The goal is to improve the balance between classification "
        "performance and model complexity for future embedded inference."
    )
    lines.append("")
    lines.append("Next step:")
    lines.append(
        "Use the best configuration to run a full validation report and "
        "compare it directly with the baseline experiment 001."
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def update_experiment_index(best_result):
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
        "description": "Decision tree tuning over max_depth and min_samples_split",
        "model": "decision_tree",
        "validation_method": "Stratified holdout 80/20",
        "top_k": 30,
        "max_depth": best_result["max_depth"],
        "min_samples_split": best_result["min_samples_split"],
        "accuracy": f"{best_result['accuracy']:.6f}",
        "macro_f1": f"{best_result['macro_f1']:.6f}",
        "weighted_f1": f"{best_result['weighted_f1']:.6f}",
        "main_observation": (
            "Best configuration selected by macro F1 while considering "
            "tree size for embedded deployment"
        ),
    }

    existing_rows.append(new_row)

    with open(index_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in existing_rows:
            writer.writerow(row)


# ================= MAIN EXPERIMENT =================

def run_experiment():
    dataset = load_pickle(SELECTED_FEATURE_DATASET_FILE)

    if not dataset:
        raise ValueError("Selected feature dataset is empty.")

    labels = get_labels(dataset)

    train_dataset, test_dataset = stratified_holdout_split(
        dataset,
        test_size=TEST_SIZE,
        seed=RANDOM_SEED,
    )

    results = []

    total_experiments = len(MAX_DEPTH_VALUES) * len(MIN_SAMPLES_SPLIT_VALUES)
    current_experiment = 0

    for max_depth in MAX_DEPTH_VALUES:
        for min_samples_split in MIN_SAMPLES_SPLIT_VALUES:
            current_experiment += 1

            print(
                f"[{current_experiment}/{total_experiments}] "
                f"Training tree with "
                f"max_depth={max_depth}, "
                f"min_samples_split={min_samples_split}"
            )

            tree = decision_tree.build_tree(
                train_dataset,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
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

            complexity = analyze_tree(tree)

            result = {
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "train_samples": len(train_dataset),
                "test_samples": len(test_dataset),
                "features_per_sample": len(dataset[0]["features"]),
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_precision": metrics["weighted_precision"],
                "weighted_recall": metrics["weighted_recall"],
                "weighted_f1": metrics["weighted_f1"],
                "empty_precision": metrics.get("empty_precision", 0.0),
                "empty_recall": metrics.get("empty_recall", 0.0),
                "empty_f1": metrics.get("empty_f1", 0.0),
                "static_presence_precision": metrics.get("static_presence_precision", 0.0),
                "static_presence_recall": metrics.get("static_presence_recall", 0.0),
                "static_presence_f1": metrics.get("static_presence_f1", 0.0),
                "movement_precision": metrics.get("movement_precision", 0.0),
                "movement_recall": metrics.get("movement_recall", 0.0),
                "movement_f1": metrics.get("movement_f1", 0.0),
                "tree_nodes": complexity["nodes"],
                "tree_decision_nodes": complexity["decision_nodes"],
                "tree_leaf_nodes": complexity["leaf_nodes"],
                "tree_actual_max_depth": complexity["max_depth"],
            }

            results.append(result)

    results = sorted(
        results,
        key=lambda item: (
            item["macro_f1"],
            item["accuracy"],
            -item["tree_nodes"],
        ),
        reverse=True,
    )

    best_result = select_best_result(results)

    paths = get_experiment_paths()

    save_tuning_results(
        results,
        paths["tuning_results"],
    )

    save_best_result(
        best_result,
        paths["best_result"],
    )

    save_experiment_notes(
        paths["experiment_notes"],
        dataset,
        train_dataset,
        test_dataset,
        best_result,
    )

    update_experiment_index(best_result)

    print()
    print("Decision tree tuning finished.")
    print("=" * 70)
    print("Best configuration:")
    print(f"max_depth: {best_result['max_depth']}")
    print(f"min_samples_split: {best_result['min_samples_split']}")
    print(f"accuracy: {best_result['accuracy']:.6f}")
    print(f"macro_f1: {best_result['macro_f1']:.6f}")
    print(f"weighted_f1: {best_result['weighted_f1']:.6f}")
    print(f"empty_recall: {best_result['empty_recall']:.6f}")
    print(f"empty_f1: {best_result['empty_f1']:.6f}")
    print(f"tree_nodes: {best_result['tree_nodes']}")
    print()
    print("Results saved to:")
    print(paths["run_dir"])


def main():
    print()
    print("Decision Tree Tuning Experiment")
    print("=" * 70)
    run_experiment()


if __name__ == "__main__":
    main()