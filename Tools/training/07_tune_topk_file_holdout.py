"""
Top-K Feature Tuning with File-Based Holdout

Experiment 012

This script evaluates different values of top_k for Fisher Score feature
selection using file-based holdout validation.

The dataset is first split by acquisition file. Then, Fisher Score ranking
is computed using only the training set. The selected feature indices are
then applied to both training and test sets.

This avoids using test data during feature selection.

Results are saved under:

    Tools/datasets/results/runs/012_topk_train_only_file_holdout/

Fixed classifier configuration:
    MAX_TREE_DEPTH = 6
    MIN_SAMPLES_SPLIT = 5
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


CLASS_ORDER = [
    "empty",
    "static_presence",
    "movement",
]

TEST_SIZE = 0.20
RANDOM_SEED = 42

EXPERIMENT_ID = "012"
EXPERIMENT_FOLDER_NAME = "012_topk_train_only_file_holdout"

TOP_K_VALUES = [
    30,
    70,
    100,
    126,
]


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

    for sample in dataset:
        predicted_label = decision_tree.predict_one(
            tree,
            sample["features"],
        )

        predictions.append(
            {
                "sample_id": sample.get("sample_id", ""),
                "group_id": get_sample_group_id(sample),
                "quadrant": sample.get("quadrant", "unknown"),
                "file_name": sample.get("file_name", ""),
                "source_file": sample.get("source_file", ""),
                "window_index": sample.get("window_index", ""),
                "true_label": sample["label"],
                "predicted_label": predicted_label,
                "correct": sample["label"] == predicted_label,
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
        "topk_results": tables_dir / "topk_tuning_results.csv",
        "best_confusion_matrix": tables_dir / "best_confusion_matrix.csv",
        "best_class_metrics": tables_dir / "best_class_metrics.csv",
        "best_quadrant_metrics": tables_dir / "best_quadrant_metrics.csv",
        "best_predictions": tables_dir / "best_classifier_predictions.csv",
        "best_report": reports_dir / "best_topk_config.txt",
        "experiment_notes": run_dir / "experiment_notes.txt",
    }


# ================= SAVE OUTPUTS =================

def save_topk_results(results, output_path):
    if not results:
        return

    fieldnames = list(results[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for item in results:
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


def select_best_result(results):
    """
    Selection criterion:
        1. Highest macro F1-score
        2. Highest accuracy
        3. Smaller number of selected features
        4. Smaller tree
    """
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

    lines.append("BEST TOP-K CONFIGURATION - FILE-BASED HOLDOUT")
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
    feature_dataset,
    train_dataset,
    test_dataset,
    train_groups,
    test_groups,
    best_result,
):
    total_class_count = count_by_label(feature_dataset)
    train_class_count = count_by_label(train_dataset)
    test_class_count = count_by_label(test_dataset)

    total_group_count = count_groups_by_label(train_groups + test_groups)
    train_group_count = count_groups_by_label(train_groups)
    test_group_count = count_groups_by_label(test_groups)

    labels = get_labels(feature_dataset)

    lines = []

    lines.append("Experiment 012 - Train-only Top-K tuning with file-based holdout")
    lines.append("")
    lines.append("Description:")
    lines.append(
        "Controlled experiment varying the number of Fisher Score selected "
        "features while keeping the tuned decision tree parameters fixed."
    )
    lines.append("")
    lines.append("Motivation:")
    lines.append(
        "Previous top_k experiments computed the Fisher Score ranking before "
        "the train/test split. This experiment corrects the methodology by "
        "splitting the dataset by acquisition file first, computing the Fisher "
        "Score ranking only on the training set, and then applying the selected "
        "feature indices to both training and test sets."
    )
    lines.append("")
    lines.append("Validation:")
    lines.append("- File-based stratified holdout 80/20")
    lines.append(f"- Random seed: {RANDOM_SEED}")
    lines.append("")
    lines.append("Feature selection:")
    lines.append("- Fisher Score ranking computed only on training data")
    lines.append("- Test data is not used to rank or select features")
    lines.append("- Selected feature indices are applied to both train and test sets")
    lines.append("")
    lines.append("Fixed classifier parameters:")
    lines.append(f"- Max tree depth: {MAX_TREE_DEPTH}")
    lines.append(f"- Min samples split: {MIN_SAMPLES_SPLIT}")
    lines.append("")
    lines.append("Tested top_k values:")
    lines.append(str(TOP_K_VALUES))
    lines.append("")
    lines.append("Dataset:")
    lines.append(f"- Total samples/windows: {len(feature_dataset)}")
    lines.append(f"- Train samples/windows: {len(train_dataset)}")
    lines.append(f"- Test samples/windows: {len(test_dataset)}")
    lines.append(f"- Total files/groups: {len(train_groups) + len(test_groups)}")
    lines.append(f"- Train files/groups: {len(train_groups)}")
    lines.append(f"- Test files/groups: {len(test_groups)}")
    lines.append(f"- Original features per sample: {len(feature_dataset[0]['features'])}")
    lines.append(f"- Selected features in best result: {best_result['top_k']}")
    lines.append("")
    lines.append("Class distribution by samples/windows:")
    lines.append(f"{'Class':<20}{'Total':>10}{'Train':>10}{'Test':>10}")

    for label in labels:
        lines.append(
            f"{label:<20}"
            f"{total_class_count[label]:>10}"
            f"{train_class_count[label]:>10}"
            f"{test_class_count[label]:>10}"
        )

    lines.append("")
    lines.append("Class distribution by files/groups:")
    lines.append(f"{'Class':<20}{'Total':>10}{'Train':>10}{'Test':>10}")

    for label in labels:
        lines.append(
            f"{label:<20}"
            f"{total_group_count[label]:>10}"
            f"{train_group_count[label]:>10}"
            f"{test_group_count[label]:>10}"
        )

    lines.append("")
    lines.append("Best result:")
    lines.append(f"- top_k: {best_result['top_k']}")
    lines.append(f"- accuracy: {best_result['accuracy']:.6f}")
    lines.append(f"- macro F1-score: {best_result['macro_f1']:.6f}")
    lines.append(f"- weighted F1-score: {best_result['weighted_f1']:.6f}")
    lines.append(f"- empty F1-score: {best_result['empty_f1']:.6f}")
    lines.append(f"- static_presence F1-score: {best_result['static_presence_f1']:.6f}")
    lines.append(f"- movement F1-score: {best_result['movement_f1']:.6f}")
    lines.append("")
    lines.append("Main observation:")
    lines.append(
        "This experiment provides a cleaner estimate of top_k performance, "
        "because the feature ranking is learned only from the training data. "
        "The result should be compared with experiments 006 and 011, which "
        "used feature selections obtained before this methodological correction."
    )
    lines.append("")
    lines.append("Next step:")
    lines.append(
        "If the result remains close to the previous best configuration, the "
        "pipeline can proceed to temporal feature engineering. If performance "
        "drops significantly, previous results should be treated as optimistic."
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
        "description": "Top-K feature tuning using file-based holdout",
        "model": MODEL,
        "validation_method": "File-based stratified holdout 80/20",
        "top_k": best_result["top_k"],
        "max_depth": best_result["max_depth"],
        "min_samples_split": best_result["min_samples_split"],
        "accuracy": f"{best_result['accuracy']:.6f}",
        "macro_f1": f"{best_result['macro_f1']:.6f}",
        "weighted_f1": f"{best_result['weighted_f1']:.6f}",
        "main_observation": (
            "Best top_k selected by macro F1 under file-based validation"
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
    feature_dataset = load_pickle(FEATURE_DATASET_FILE)

    if not feature_dataset:
        raise ValueError("Feature dataset is empty.")

    labels = get_labels(feature_dataset)

    print("Splitting dataset by file before feature selection...")

    full_train_dataset, full_test_dataset, train_groups, test_groups = file_stratified_holdout_split(
        feature_dataset,
        test_size=TEST_SIZE,
        seed=RANDOM_SEED,
    )

    print("Ranking features using Fisher Score on training data only...")
    ranking = rank_features_by_fisher_score(full_train_dataset)

    valid_top_k_values = []

    for top_k in TOP_K_VALUES:
        if top_k <= len(ranking):
            valid_top_k_values.append(top_k)

    if not valid_top_k_values:
        raise ValueError("No valid top_k values for current feature vector.")

    results = []
    artifacts_by_top_k = {}

    total_experiments = len(valid_top_k_values)

    for experiment_index, top_k in enumerate(valid_top_k_values, start=1):
        print()
        print(
            f"[{experiment_index}/{total_experiments}] "
            f"Evaluating top_k={top_k}"
        )

        selected_indices = []

        for item in ranking[:top_k]:
            selected_indices.append(item["feature_index"])

        train_dataset = select_features_by_indices(
            full_train_dataset,
            selected_indices,
        )

        test_dataset = select_features_by_indices(
            full_test_dataset,
            selected_indices,
        )
        
        selected_dataset = train_dataset + test_dataset

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

        quadrant_metrics = compute_quadrant_metrics(
            predictions,
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
            "empty_precision": metrics.get("empty_precision", 0.0),
            "empty_recall": metrics.get("empty_recall", 0.0),
            "empty_f1": metrics.get("empty_f1", 0.0),
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
            "selected_indices": ";".join(str(index) for index in selected_indices),
        }

        results.append(result)

        artifacts_by_top_k[top_k] = {
            "selected_dataset": selected_dataset,
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "train_groups": train_groups,
            "test_groups": test_groups,
            "predictions": predictions,
            "matrix": matrix,
            "per_class_metrics": per_class_metrics,
            "quadrant_metrics": quadrant_metrics,
        }

        print(f"accuracy: {metrics['accuracy']:.6f}")
        print(f"macro_f1: {metrics['macro_f1']:.6f}")
        print(f"weighted_f1: {metrics['weighted_f1']:.6f}")
        print(f"empty_f1: {metrics.get('empty_f1', 0.0):.6f}")
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

    save_topk_results(
        results,
        paths["topk_results"],
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

    save_quadrant_metrics(
        best_artifacts["quadrant_metrics"],
        paths["best_quadrant_metrics"],
    )

    save_predictions(
        best_artifacts["predictions"],
        paths["best_predictions"],
    )

    save_best_report(
        best_result,
        paths["best_report"],
    )

    save_experiment_notes(
        paths["experiment_notes"],
        feature_dataset,
        best_artifacts["train_dataset"],
        best_artifacts["test_dataset"],
        best_artifacts["train_groups"],
        best_artifacts["test_groups"],
        best_result,
    )

    update_experiment_index(best_result)

    print()
    print("Top-K tuning finished.")
    print("=" * 70)
    print("Best configuration:")
    print(f"top_k: {best_result['top_k']}")
    print(f"max_depth: {best_result['max_depth']}")
    print(f"min_samples_split: {best_result['min_samples_split']}")
    print(f"accuracy: {best_result['accuracy']:.6f}")
    print(f"macro_f1: {best_result['macro_f1']:.6f}")
    print(f"weighted_f1: {best_result['weighted_f1']:.6f}")
    print(f"empty_f1: {best_result['empty_f1']:.6f}")
    print(f"static_presence_f1: {best_result['static_presence_f1']:.6f}")
    print(f"movement_f1: {best_result['movement_f1']:.6f}")
    print(f"tree_nodes: {best_result['tree_nodes']}")
    print()
    print("Results saved to:")
    print(paths["run_dir"])


def main():
    print()
    print("Top-K Feature Tuning with File-Based Holdout")
    print("=" * 70)
    print("Model:", MODEL)
    print("Max tree depth:", MAX_TREE_DEPTH)
    print("Min samples split:", MIN_SAMPLES_SPLIT)
    print("Tested top_k values:", TOP_K_VALUES)
    print()
    run_experiment()


if __name__ == "__main__":
    main()