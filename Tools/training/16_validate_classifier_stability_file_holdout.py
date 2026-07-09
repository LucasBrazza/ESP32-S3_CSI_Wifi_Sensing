"""
Classifier Stability and Capacity Validation

Experiment 018

This script evaluates whether the best compact ensemble found in
Experiment 017 is stable across multiple file-based holdout splits.

Purpose:

    Verify whether the compact Gradient Boosting model is consistently
    competitive or whether the result from a single split was accidental.

This experiment compares:

    - official single-tree baseline
    - deeper single-tree candidate
    - compact Gradient Boosting candidates
    - stronger reference models

Feature selection is recomputed using Fisher Score on the training split
only for each seed.
"""

import csv
import random
from collections import Counter

from Tools.common.config import (
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


try:
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        ExtraTreesClassifier,
    )
except ImportError as error:
    raise ImportError(
        "scikit-learn is required for experiment 018. "
        "Install it with: pip install scikit-learn"
    ) from error


CLASS_ORDER = [
    "empty",
    "static_presence",
    "movement",
]

EXPERIMENT_ID = "018"
EXPERIMENT_FOLDER_NAME = "018_classifier_stability_validation"

TEST_SIZE = 0.20

EXPECTED_FEATURES_PER_SAMPLE = 231

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


MODEL_CONFIGS = [
    {
        "name": "custom_tree_depth6",
        "type": "custom_tree",
        "top_k": 126,
        "max_depth": 6,
        "estimated_model_units": 1,
    },
    {
        "name": "custom_tree_depth8",
        "type": "custom_tree",
        "top_k": 150,
        "max_depth": 8,
        "estimated_model_units": 1,
    },
    {
        "name": "gradient_boosting_depth3_5",
        "type": "gradient_boosting",
        "top_k": 231,
        "n_estimators": 5,
        "learning_rate": 0.10,
        "max_depth": 3,
        "estimated_model_units": 5,
    },
    {
        "name": "gradient_boosting_depth3_10",
        "type": "gradient_boosting",
        "top_k": 231,
        "n_estimators": 10,
        "learning_rate": 0.10,
        "max_depth": 3,
        "estimated_model_units": 10,
    },
    {
        "name": "gradient_boosting_depth3_20",
        "type": "gradient_boosting",
        "top_k": 231,
        "n_estimators": 20,
        "learning_rate": 0.10,
        "max_depth": 3,
        "estimated_model_units": 20,
    },
    {
        "name": "gradient_boosting_full_100",
        "type": "gradient_boosting",
        "top_k": 126,
        "n_estimators": 100,
        "learning_rate": 0.10,
        "max_depth": 3,
        "estimated_model_units": 100,
    },
    {
        "name": "extra_trees_100",
        "type": "extra_trees",
        "top_k": 180,
        "n_estimators": 100,
        "max_depth": None,
        "estimated_model_units": 100,
    },
]


# ================= DATASET HELPERS =================

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


def dataset_to_xy(dataset):
    x_values = []
    y_values = []

    for sample in dataset:
        x_values.append(sample["features"])
        y_values.append(sample["label"])

    return x_values, y_values


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

    weighted_f1 = safe_division(
        sum(item["f1_score"] * item["support"] for item in per_class),
        total_samples,
    )

    result = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    for item in per_class:
        label = item["label"]

        result[f"{label}_precision"] = item["precision"]
        result[f"{label}_recall"] = item["recall"]
        result[f"{label}_f1"] = item["f1_score"]
        result[f"{label}_support"] = item["support"]

    return result


# ================= MODELS =================

def predict_custom_tree(train_dataset, test_dataset, max_depth):
    tree = decision_tree.build_tree(
        train_dataset,
        max_depth=max_depth,
        min_samples_split=MIN_SAMPLES_SPLIT,
    )

    predicted_labels = []

    for sample in test_dataset:
        predicted_labels.append(
            decision_tree.predict_one(
                tree,
                sample["features"],
            )
        )

    return predicted_labels


def build_sklearn_model(config, seed):
    if config["type"] == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            random_state=seed,
        )

    if config["type"] == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=MIN_SAMPLES_SPLIT,
            class_weight="balanced",
            random_state=seed,
        )

    raise ValueError(f"Unknown model type: {config['type']}")


def evaluate_model(config, train_dataset, test_dataset, seed):
    if config["type"] == "custom_tree":
        predicted_labels = predict_custom_tree(
            train_dataset,
            test_dataset,
            max_depth=config["max_depth"],
        )
    else:
        x_train, y_train = dataset_to_xy(train_dataset)
        x_test, _ = dataset_to_xy(test_dataset)

        model = build_sklearn_model(
            config,
            seed,
        )

        model.fit(x_train, y_train)
        predicted_labels = model.predict(x_test)

    true_labels = []

    for sample in test_dataset:
        true_labels.append(sample["label"])

    matrix = build_confusion_matrix(
        true_labels,
        predicted_labels,
        CLASS_ORDER,
    )

    metrics = compute_class_metrics(
        matrix,
        CLASS_ORDER,
    )

    return metrics


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
        "seed_results": tables_dir / "stability_results_by_seed.csv",
        "summary_results": tables_dir / "stability_summary.csv",
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


def summarize_results(seed_results):
    results_by_model = {}

    for row in seed_results:
        classifier = row["classifier"]

        if classifier not in results_by_model:
            results_by_model[classifier] = []

        results_by_model[classifier].append(row)

    summary_rows = []

    for classifier in results_by_model:
        rows = results_by_model[classifier]

        accuracy_values = [float(row["accuracy"]) for row in rows]
        macro_f1_values = [float(row["macro_f1"]) for row in rows]
        weighted_f1_values = [float(row["weighted_f1"]) for row in rows]
        empty_f1_values = [float(row["empty_f1"]) for row in rows]
        static_f1_values = [float(row["static_presence_f1"]) for row in rows]
        movement_f1_values = [float(row["movement_f1"]) for row in rows]

        reference = rows[0]

        summary_rows.append(
            {
                "classifier": classifier,
                "top_k": reference["top_k"],
                "estimated_model_units": reference["estimated_model_units"],
                "max_depth": reference["max_depth"],
                "seeds": len(rows),

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

                "empty_f1_mean": mean(empty_f1_values),
                "static_presence_f1_mean": mean(static_f1_values),
                "movement_f1_mean": mean(movement_f1_values),

                "movement_f1_std": std(movement_f1_values),
                "movement_f1_min": min(movement_f1_values),
                "movement_f1_max": max(movement_f1_values),
            }
        )

    summary_rows = sorted(
        summary_rows,
        key=lambda item: (
            item["macro_f1_mean"],
            item["accuracy_mean"],
            item["movement_f1_mean"],
            -int(item["estimated_model_units"]),
        ),
        reverse=True,
    )

    return summary_rows


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
        "description": "Classifier stability validation across multiple file-based splits",
        "model": best_result["classifier"],
        "validation_method": f"File-based holdout repeated across {best_result['seeds']} seeds",
        "top_k": best_result["top_k"],
        "max_depth": best_result["max_depth"],
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "accuracy": f"{best_result['accuracy_mean']:.6f}",
        "macro_f1": f"{best_result['macro_f1_mean']:.6f}",
        "weighted_f1": f"{best_result['weighted_f1_mean']:.6f}",
        "main_observation": (
            "Best model selected by mean macro F1 across multiple seeds"
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
    seed_results,
    summary_results,
):
    best_result = summary_results[0]

    lines = []

    lines.append("Experiment 018 - Classifier stability and capacity validation")
    lines.append("")
    lines.append("Objective:")
    lines.append(
        "Repeat file-based holdout validation across multiple seeds to verify "
        "whether the compact ensemble result from Experiment 017 is stable."
    )
    lines.append("")
    lines.append("Dataset:")
    lines.append(f"- Total samples/windows: {len(feature_dataset)}")
    lines.append(f"- Features per sample: {len(feature_dataset[0]['features'])}")
    lines.append("")
    lines.append("Validation:")
    lines.append(f"- File-based stratified holdout: {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}")
    lines.append(f"- Seeds: {SEEDS}")
    lines.append("- Fisher Score ranking recomputed using training data only for each seed")
    lines.append("")
    lines.append("Compared models:")

    for config in MODEL_CONFIGS:
        lines.append(
            f"- {config['name']}: top_k={config['top_k']}, "
            f"units={config['estimated_model_units']}, "
            f"max_depth={config.get('max_depth', '')}"
        )

    lines.append("")
    lines.append("Best mean result:")
    lines.append(f"- classifier: {best_result['classifier']}")
    lines.append(f"- top_k: {best_result['top_k']}")
    lines.append(f"- estimated model units: {best_result['estimated_model_units']}")
    lines.append(f"- max depth: {best_result['max_depth']}")
    lines.append(f"- accuracy mean: {best_result['accuracy_mean']:.6f}")
    lines.append(f"- accuracy std: {best_result['accuracy_std']:.6f}")
    lines.append(f"- macro F1 mean: {best_result['macro_f1_mean']:.6f}")
    lines.append(f"- macro F1 std: {best_result['macro_f1_std']:.6f}")
    lines.append(f"- movement F1 mean: {best_result['movement_f1_mean']:.6f}")
    lines.append(f"- movement F1 std: {best_result['movement_f1_std']:.6f}")
    lines.append("")
    lines.append("Interpretation rule:")
    lines.append(
        "If the compact model remains close to or better than larger models "
        "on mean macro F1, it is a strong candidate for future embedded "
        "implementation. If larger models consistently outperform it, the "
        "compact ensemble may still be too simple."
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

    seed_results = []

    for seed in SEEDS:
        print()
        print("=" * 70)
        print(f"Seed {seed}")
        print("=" * 70)

        full_train_dataset, full_test_dataset, train_groups, test_groups = file_stratified_holdout_split(
            feature_dataset,
            test_size=TEST_SIZE,
            seed=seed,
        )

        ranking = rank_features_by_fisher_score(full_train_dataset)

        selected_datasets_by_top_k = {}

        unique_top_k_values = []

        for config in MODEL_CONFIGS:
            top_k = config["top_k"]

            if top_k not in unique_top_k_values:
                unique_top_k_values.append(top_k)

        for top_k in unique_top_k_values:
            if top_k > len(ranking):
                raise ValueError(
                    f"Invalid top_k={top_k}. "
                    f"Ranking has only {len(ranking)} features."
                )

            selected_indices = []

            for item in ranking[:top_k]:
                selected_indices.append(item["feature_index"])

            selected_train_dataset = select_features_by_indices(
                full_train_dataset,
                selected_indices,
            )

            selected_test_dataset = select_features_by_indices(
                full_test_dataset,
                selected_indices,
            )

            selected_datasets_by_top_k[top_k] = (
                selected_train_dataset,
                selected_test_dataset,
            )

        train_count = count_by_label(full_train_dataset)
        test_count = count_by_label(full_test_dataset)

        print("Train samples:", dict(train_count))
        print("Test samples:", dict(test_count))

        for config in MODEL_CONFIGS:
            top_k = config["top_k"]

            train_dataset, test_dataset = selected_datasets_by_top_k[top_k]

            metrics = evaluate_model(
                config,
                train_dataset,
                test_dataset,
                seed,
            )

            row = {
                "seed": seed,
                "classifier": config["name"],
                "top_k": top_k,
                "estimated_model_units": config["estimated_model_units"],
                "max_depth": config.get("max_depth", ""),
                "train_samples": len(train_dataset),
                "test_samples": len(test_dataset),

                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],

                "empty_f1": metrics.get("empty_f1", 0.0),
                "static_presence_f1": metrics.get("static_presence_f1", 0.0),
                "movement_f1": metrics.get("movement_f1", 0.0),
            }

            seed_results.append(row)

            print(
                f"{config['name']:<30} "
                f"accuracy={metrics['accuracy']:.6f} "
                f"macro_f1={metrics['macro_f1']:.6f} "
                f"movement_f1={metrics.get('movement_f1', 0.0):.6f}"
            )

    summary_results = summarize_results(seed_results)

    paths = get_experiment_paths()

    save_csv(
        seed_results,
        paths["seed_results"],
    )

    save_csv(
        summary_results,
        paths["summary_results"],
    )

    save_experiment_notes(
        paths["experiment_notes"],
        feature_dataset,
        seed_results,
        summary_results,
    )

    update_experiment_index(
        summary_results[0],
    )

    print()
    print("=" * 70)
    print("Classifier stability validation finished.")
    print("=" * 70)
    print("Best mean result:")
    print(f"classifier: {summary_results[0]['classifier']}")
    print(f"top_k: {summary_results[0]['top_k']}")
    print(f"accuracy_mean: {summary_results[0]['accuracy_mean']:.6f}")
    print(f"accuracy_std: {summary_results[0]['accuracy_std']:.6f}")
    print(f"macro_f1_mean: {summary_results[0]['macro_f1_mean']:.6f}")
    print(f"macro_f1_std: {summary_results[0]['macro_f1_std']:.6f}")
    print(f"movement_f1_mean: {summary_results[0]['movement_f1_mean']:.6f}")
    print(f"movement_f1_std: {summary_results[0]['movement_f1_std']:.6f}")
    print()
    print("Results saved to:")
    print(paths["run_dir"])


def main():
    print()
    print("Classifier Stability and Capacity Validation")
    print("=" * 70)
    print("Experiment:", EXPERIMENT_ID)
    print("Feature dataset:", FEATURE_DATASET_FILE)
    print("Seeds:", SEEDS)
    print()

    run_experiment()


if __name__ == "__main__":
    main()