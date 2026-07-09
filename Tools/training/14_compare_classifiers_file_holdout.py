"""
Classifier Capacity Diagnostic

Experiment 016

This script compares multiple classifiers using the same file-based
holdout strategy and train-only Fisher Score feature ranking.

Purpose:

    Determine whether the current decision tree is too simple for the
    CSI classification problem.

This experiment is diagnostic only. Some tested models are not intended
to be embedded directly on the ESP32-S3.
"""

import csv
import random
from collections import Counter

from Tools.common.config import (
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


try:
    from sklearn.ensemble import (
        RandomForestClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError as error:
    raise ImportError(
        "scikit-learn is required for experiment 016. "
        "Install it with: pip install scikit-learn"
    ) from error


CLASS_ORDER = [
    "empty",
    "static_presence",
    "movement",
]

EXPERIMENT_ID = "016"
EXPERIMENT_FOLDER_NAME = "016_classifier_capacity_diagnostic"

TEST_SIZE = 0.20
RANDOM_SEED = 42

EXPECTED_FEATURES_PER_SAMPLE = 231

TOP_K_VALUES = [
    70,
    100,
    126,
    150,
    180,
    231,
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

    return result, per_class


def build_prediction_records(test_dataset, predicted_labels):
    predictions = []

    for sample, predicted_label in zip(test_dataset, predicted_labels):
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


# ================= MODELS =================

def predict_custom_decision_tree(train_dataset, test_dataset):
    tree = decision_tree.build_tree(
        train_dataset,
        max_depth=MAX_TREE_DEPTH,
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


def get_sklearn_classifiers():
    classifiers = []

    classifiers.append(
        {
            "name": "random_forest_100",
            "model": RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_SEED,
                class_weight="balanced",
            ),
            "use_scaled_features": False,
        }
    )

    classifiers.append(
        {
            "name": "extra_trees_100",
            "model": ExtraTreesClassifier(
                n_estimators=100,
                random_state=RANDOM_SEED,
                class_weight="balanced",
            ),
            "use_scaled_features": False,
        }
    )

    classifiers.append(
        {
            "name": "gradient_boosting",
            "model": GradientBoostingClassifier(
                random_state=RANDOM_SEED,
            ),
            "use_scaled_features": False,
        }
    )

    classifiers.append(
        {
            "name": "knn_3",
            "model": KNeighborsClassifier(
                n_neighbors=3,
            ),
            "use_scaled_features": True,
        }
    )

    classifiers.append(
        {
            "name": "knn_5",
            "model": KNeighborsClassifier(
                n_neighbors=5,
            ),
            "use_scaled_features": True,
        }
    )

    classifiers.append(
        {
            "name": "linear_svm",
            "model": LinearSVC(
                class_weight="balanced",
                random_state=RANDOM_SEED,
                max_iter=20000,
            ),
            "use_scaled_features": True,
        }
    )

    classifiers.append(
        {
            "name": "rbf_svm",
            "model": SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
            "use_scaled_features": True,
        }
    )

    classifiers.append(
        {
            "name": "logistic_regression",
            "model": LogisticRegression(
                class_weight="balanced",
                random_state=RANDOM_SEED,
                max_iter=20000,
            ),
            "use_scaled_features": True,
        }
    )

    return classifiers


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
        "comparison_results": tables_dir / "classifier_comparison_results.csv",
        "best_confusion_matrix": tables_dir / "best_confusion_matrix.csv",
        "best_class_metrics": tables_dir / "best_class_metrics.csv",
        "best_predictions": tables_dir / "best_classifier_predictions.csv",
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
    ordered_results = sorted(
        results,
        key=lambda item: (
            item["macro_f1"],
            item["accuracy"],
            item["movement_f1"],
            -item["top_k"],
        ),
        reverse=True,
    )

    return ordered_results[0]


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
        "description": "Classifier capacity diagnostic using file-based holdout",
        "model": best_result["classifier"],
        "validation_method": "File-based stratified holdout 80/20",
        "top_k": best_result["top_k"],
        "max_depth": MAX_TREE_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "accuracy": f"{best_result['accuracy']:.6f}",
        "macro_f1": f"{best_result['macro_f1']:.6f}",
        "weighted_f1": f"{best_result['weighted_f1']:.6f}",
        "main_observation": (
            "Best classifier selected by macro F1 under file-based validation"
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
    train_dataset,
    test_dataset,
    train_groups,
    test_groups,
    results,
    best_result,
):
    total_class_count = count_by_label(feature_dataset)
    train_class_count = count_by_label(train_dataset)
    test_class_count = count_by_label(test_dataset)

    total_group_count = count_groups_by_label(train_groups + test_groups)
    train_group_count = count_groups_by_label(train_groups)
    test_group_count = count_groups_by_label(test_groups)

    lines = []

    lines.append("Experiment 016 - Classifier capacity diagnostic")
    lines.append("")
    lines.append("Objective:")
    lines.append(
        "Compare stronger classifiers on the current CSI feature dataset to "
        "verify whether the custom decision tree is too simple for the problem."
    )
    lines.append("")
    lines.append("Validation:")
    lines.append("- File-based stratified holdout 80/20")
    lines.append(f"- Random seed: {RANDOM_SEED}")
    lines.append("")
    lines.append("Feature selection:")
    lines.append("- Fisher Score ranking computed only on training data")
    lines.append("- Same selected feature indices applied to train and test sets")
    lines.append("")
    lines.append("Dataset:")
    lines.append(f"- Total samples/windows: {len(feature_dataset)}")
    lines.append(f"- Train samples/windows: {len(train_dataset)}")
    lines.append(f"- Test samples/windows: {len(test_dataset)}")
    lines.append(f"- Total files/groups: {len(train_groups) + len(test_groups)}")
    lines.append(f"- Train files/groups: {len(train_groups)}")
    lines.append(f"- Test files/groups: {len(test_groups)}")
    lines.append(f"- Original features per sample: {len(feature_dataset[0]['features'])}")
    lines.append("")
    lines.append("Class distribution by samples/windows:")
    lines.append(f"{'Class':<20}{'Total':>10}{'Train':>10}{'Test':>10}")

    for label in CLASS_ORDER:
        lines.append(
            f"{label:<20}"
            f"{total_class_count[label]:>10}"
            f"{train_class_count[label]:>10}"
            f"{test_class_count[label]:>10}"
        )

    lines.append("")
    lines.append("Class distribution by files/groups:")
    lines.append(f"{'Class':<20}{'Total':>10}{'Train':>10}{'Test':>10}")

    for label in CLASS_ORDER:
        lines.append(
            f"{label:<20}"
            f"{total_group_count[label]:>10}"
            f"{train_group_count[label]:>10}"
            f"{test_group_count[label]:>10}"
        )

    lines.append("")
    lines.append("Best result:")
    lines.append(f"- classifier: {best_result['classifier']}")
    lines.append(f"- top_k: {best_result['top_k']}")
    lines.append(f"- accuracy: {best_result['accuracy']:.6f}")
    lines.append(f"- macro F1-score: {best_result['macro_f1']:.6f}")
    lines.append(f"- weighted F1-score: {best_result['weighted_f1']:.6f}")
    lines.append(f"- empty F1-score: {best_result['empty_f1']:.6f}")
    lines.append(f"- static_presence F1-score: {best_result['static_presence_f1']:.6f}")
    lines.append(f"- movement F1-score: {best_result['movement_f1']:.6f}")
    lines.append("")
    lines.append("Interpretation rule:")
    lines.append(
        "If stronger classifiers substantially outperform the custom decision "
        "tree, the current decision tree is likely too simple. If all models "
        "remain close to the same performance range, the main bottleneck is "
        "more likely related to dataset quality, feature representation, or "
        "preprocessing."
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
            f"but found {features_per_sample}. "
            "Restore the 11-feature pipeline before running experiment 016."
        )

    print("Splitting dataset by file...")
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
    artifacts = {}

    classifiers = get_sklearn_classifiers()

    for top_k in valid_top_k_values:
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

        print()
        print("=" * 70)
        print(f"Evaluating top_k={top_k}")
        print("=" * 70)

        # Custom embedded-oriented tree
        classifier_name = "custom_decision_tree"

        predicted_labels = predict_custom_decision_tree(
            train_dataset,
            test_dataset,
        )

        predictions = build_prediction_records(
            test_dataset,
            predicted_labels,
        )

        matrix = build_confusion_matrix(
            predictions,
            CLASS_ORDER,
        )

        metrics, per_class_metrics = compute_class_metrics(
            matrix,
            CLASS_ORDER,
        )

        result = {
            "classifier": classifier_name,
            "top_k": top_k,
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

        results.append(result)

        artifacts[(classifier_name, top_k)] = {
            "predictions": predictions,
            "matrix": matrix,
            "per_class_metrics": per_class_metrics,
        }

        print(
            f"{classifier_name:<25} "
            f"accuracy={metrics['accuracy']:.6f} "
            f"macro_f1={metrics['macro_f1']:.6f} "
            f"movement_f1={metrics.get('movement_f1', 0.0):.6f}"
        )

        x_train_raw, y_train = dataset_to_xy(train_dataset)
        x_test_raw, y_test = dataset_to_xy(test_dataset)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_raw)
        x_test_scaled = scaler.transform(x_test_raw)

        for classifier in classifiers:
            classifier_name = classifier["name"]
            model = classifier["model"]

            if classifier["use_scaled_features"]:
                x_train = x_train_scaled
                x_test = x_test_scaled
            else:
                x_train = x_train_raw
                x_test = x_test_raw

            model.fit(x_train, y_train)
            predicted_labels = model.predict(x_test)

            predictions = build_prediction_records(
                test_dataset,
                predicted_labels,
            )

            matrix = build_confusion_matrix(
                predictions,
                CLASS_ORDER,
            )

            metrics, per_class_metrics = compute_class_metrics(
                matrix,
                CLASS_ORDER,
            )

            result = {
                "classifier": classifier_name,
                "top_k": top_k,
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

            results.append(result)

            artifacts[(classifier_name, top_k)] = {
                "predictions": predictions,
                "matrix": matrix,
                "per_class_metrics": per_class_metrics,
            }

            print(
                f"{classifier_name:<25} "
                f"accuracy={metrics['accuracy']:.6f} "
                f"macro_f1={metrics['macro_f1']:.6f} "
                f"movement_f1={metrics.get('movement_f1', 0.0):.6f}"
            )

    results = sorted(
        results,
        key=lambda item: (
            item["macro_f1"],
            item["accuracy"],
            item["movement_f1"],
            -item["top_k"],
        ),
        reverse=True,
    )

    best_result = select_best_result(results)
    best_key = (
        best_result["classifier"],
        best_result["top_k"],
    )

    paths = get_experiment_paths()

    save_csv(
        results,
        paths["comparison_results"],
    )

    save_confusion_matrix(
        artifacts[best_key]["matrix"],
        CLASS_ORDER,
        paths["best_confusion_matrix"],
    )

    save_class_metrics(
        artifacts[best_key]["per_class_metrics"],
        paths["best_class_metrics"],
    )

    save_predictions(
        artifacts[best_key]["predictions"],
        paths["best_predictions"],
    )

    save_experiment_notes(
        paths["experiment_notes"],
        feature_dataset,
        full_train_dataset,
        full_test_dataset,
        train_groups,
        test_groups,
        results,
        best_result,
    )

    update_experiment_index(best_result)

    print()
    print("=" * 70)
    print("Classifier capacity diagnostic finished.")
    print("=" * 70)
    print("Best result:")
    print(f"classifier: {best_result['classifier']}")
    print(f"top_k: {best_result['top_k']}")
    print(f"accuracy: {best_result['accuracy']:.6f}")
    print(f"macro_f1: {best_result['macro_f1']:.6f}")
    print(f"weighted_f1: {best_result['weighted_f1']:.6f}")
    print(f"empty_f1: {best_result['empty_f1']:.6f}")
    print(f"static_presence_f1: {best_result['static_presence_f1']:.6f}")
    print(f"movement_f1: {best_result['movement_f1']:.6f}")
    print()
    print("Results saved to:")
    print(paths["run_dir"])


def main():
    print()
    print("Classifier Capacity Diagnostic")
    print("=" * 70)
    print("Experiment:", EXPERIMENT_ID)
    print("Feature dataset:", FEATURE_DATASET_FILE)
    print("Top-K values:", TOP_K_VALUES)
    print()

    run_experiment()


if __name__ == "__main__":
    main()