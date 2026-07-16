"""Feature-budget comparison for compact multiclass boosting models.

Experiment 025

Compares the current Gradient Boosting and XGBoost candidates under different
Fisher Score Top-K feature budgets. Every ranking is computed using training
samples only, and both classifiers receive the same file-based split and the
same selected feature indices for each seed.

Main question:
    What is the smallest feature budget that preserves classification quality,
    especially for the movement class, while reducing embedded cost?
"""

import csv
import math
import re

import numpy as np
from collections import defaultdict
from statistics import mean, pstdev
from time import perf_counter

from Tools.common.classification_metrics import (
    align_probability_columns,
    evaluate_multiclass_predictions,
)
from Tools.common.file_holdout import (
    count_by_label,
    dataset_to_xy,
    file_stratified_holdout_split,
    select_features_by_indices,
)
from Tools.common.io_utils import load_pickle
from Tools.common.project_paths import FEATURE_DATASET_FILE, RESULTS_DIR
from Tools.preprocessing.feature_selection import rank_features_by_fisher_score

try:
    from sklearn.ensemble import GradientBoostingClassifier
except ImportError as error:
    raise ImportError(
        "scikit-learn is required for experiment 025. "
        "Install it with: python -m pip install scikit-learn"
    ) from error

try:
    from xgboost import XGBClassifier
except ImportError as error:
    raise ImportError(
        "XGBoost is required for experiment 025. "
        "Install it with: python -m pip install xgboost"
    ) from error


CLASS_ORDER = [
    "empty",
    "static_presence",
    "movement",
]

CLASS_TO_INDEX = {
    label: index
    for index, label in enumerate(CLASS_ORDER)
}

INDEX_TO_CLASS = {
    index: label
    for label, index in CLASS_TO_INDEX.items()
}

EXPERIMENT_ID = "025"
EXPERIMENT_FOLDER_NAME = "025_feature_budget_comparison"

TEST_SIZE = 0.20
EXPECTED_FEATURES_PER_SAMPLE = 231
MAX_MACRO_F1_DROP = 0.02
MAX_MOVEMENT_F1_DROP = 0.02

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

TOP_K_VALUES = [
    30,
    50,
    70,
    100,
    126,
    160,
    231,
]

MODEL_FAMILIES = [
    {
        "family": "gradient_boosting",
        "display_name": "Gradient Boosting",
        "n_estimators": 20,
        "learning_rate": 0.10,
        "max_depth": 3,
    },
    {
        "family": "xgboost",
        "display_name": "XGBoost",
        "n_estimators": 20,
        "learning_rate": 0.10,
        "max_depth": 3,
    },
]

SUMMARY_METRICS = [
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "roc_auc_ovr_macro",
    "roc_auc_ovr_weighted",
    "average_precision_macro",
    "average_precision_weighted",
    "fit_time_ms",
    "prediction_time_ms",
    "actual_tree_count",
    "actual_node_count",
    "used_feature_count",
    "mean_comparisons_per_prediction",
]

PER_CLASS_METRICS = [
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
]


# ================= MODEL BUILDING =================


def model_name(family, top_k):
    return f"{family}_top{top_k}"


def build_model(config, seed):
    family = config["family"]

    if family == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            random_state=seed,
        )

    if family == "xgboost":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=len(CLASS_ORDER),
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            min_child_weight=1,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )

    raise ValueError(f"Unknown model family: {family}")


def _gradient_boosting_complexity(model, x_test):
    estimators = model.estimators_.ravel()
    tree_count = len(estimators)
    node_count = sum(estimator.tree_.node_count for estimator in estimators)

    used_selected_indices = set()
    comparisons_by_sample = np.zeros(len(x_test), dtype=float)

    for estimator in estimators:
        for feature_index in estimator.tree_.feature:
            feature_index = int(feature_index)

            if feature_index >= 0:
                used_selected_indices.add(feature_index)

        decision_path = estimator.decision_path(x_test)
        comparisons_by_sample += decision_path.getnnz(axis=1) - 1

    mean_comparisons = (
        float(np.mean(comparisons_by_sample))
        if len(comparisons_by_sample)
        else 0.0
    )

    return {
        "actual_tree_count": tree_count,
        "actual_node_count": node_count,
        "used_selected_indices": used_selected_indices,
        "mean_comparisons_per_prediction": mean_comparisons,
    }


def _xgboost_complexity(model, x_test):
    dumps = model.get_booster().get_dump(dump_format="text")
    tree_count = len(dumps)
    node_count = 0
    used_selected_indices = set()
    feature_pattern = re.compile(r"\[f(\d+)<")
    leaf_depth_by_tree = []

    for tree_dump in dumps:
        lines = [line for line in tree_dump.splitlines() if line.strip()]
        node_count += len(lines)
        leaf_depths = {}

        for line in lines:
            depth = len(line) - len(line.lstrip("\t"))
            stripped_line = line.strip()
            node_id_text = stripped_line.split(":", 1)[0]

            try:
                node_id = int(node_id_text)
            except ValueError:
                node_id = None

            if node_id is not None and ":leaf=" in stripped_line:
                leaf_depths[node_id] = depth

            match = feature_pattern.search(line)

            if match:
                used_selected_indices.add(int(match.group(1)))

        leaf_depth_by_tree.append(leaf_depths)

    leaf_indices = np.asarray(model.apply(x_test))

    if leaf_indices.ndim == 3:
        leaf_indices = leaf_indices.reshape(leaf_indices.shape[0], -1)
    elif leaf_indices.ndim == 1:
        leaf_indices = leaf_indices.reshape(-1, 1)

    comparisons_by_sample = np.zeros(leaf_indices.shape[0], dtype=float)

    for tree_index in range(min(tree_count, leaf_indices.shape[1])):
        depth_map = leaf_depth_by_tree[tree_index]

        for sample_index, leaf_index in enumerate(leaf_indices[:, tree_index]):
            comparisons_by_sample[sample_index] += depth_map.get(
                int(leaf_index),
                0,
            )

    mean_comparisons = (
        float(np.mean(comparisons_by_sample))
        if len(comparisons_by_sample)
        else 0.0
    )

    return {
        "actual_tree_count": tree_count,
        "actual_node_count": node_count,
        "used_selected_indices": used_selected_indices,
        "mean_comparisons_per_prediction": mean_comparisons,
    }


def inspect_model_complexity(
    model,
    family,
    selected_original_indices,
    x_test,
):
    if family == "gradient_boosting":
        complexity = _gradient_boosting_complexity(model, x_test)
    elif family == "xgboost":
        complexity = _xgboost_complexity(model, x_test)
    else:
        raise ValueError(f"Unsupported family for complexity analysis: {family}")

    used_original_indices = sorted(
        selected_original_indices[selected_index]
        for selected_index in complexity.pop("used_selected_indices")
    )

    complexity["used_feature_count"] = len(used_original_indices)
    complexity["used_original_feature_indices"] = "|".join(
        str(index)
        for index in used_original_indices
    )

    return complexity


def fit_and_predict(
    config,
    train_dataset,
    test_dataset,
    selected_original_indices,
    seed,
):
    x_train, y_train_labels = dataset_to_xy(train_dataset)
    x_test, y_test_labels = dataset_to_xy(test_dataset)

    model = build_model(config, seed)

    if config["family"] == "xgboost":
        y_train = [CLASS_TO_INDEX[label] for label in y_train_labels]
    else:
        y_train = y_train_labels

    fit_started = perf_counter()
    model.fit(x_train, y_train)
    fit_time_ms = (perf_counter() - fit_started) * 1000.0

    prediction_started = perf_counter()
    raw_predictions = model.predict(x_test)
    raw_probabilities = model.predict_proba(x_test)
    prediction_time_ms = (perf_counter() - prediction_started) * 1000.0

    if config["family"] == "xgboost":
        predicted_labels = [
            INDEX_TO_CLASS[int(index)]
            for index in raw_predictions
        ]
        probabilities = raw_probabilities
    else:
        predicted_labels = [str(label) for label in raw_predictions]
        probabilities = align_probability_columns(
            raw_probabilities,
            model.classes_,
            CLASS_ORDER,
        )

    evaluation = evaluate_multiclass_predictions(
        true_labels=y_test_labels,
        predicted_labels=predicted_labels,
        class_order=CLASS_ORDER,
        probabilities=probabilities,
    )

    complexity = inspect_model_complexity(
        model,
        config["family"],
        selected_original_indices,
        x_test,
    )

    evaluation["metrics"].update(
        {
            "fit_time_ms": fit_time_ms,
            "prediction_time_ms": prediction_time_ms,
            **complexity,
        }
    )

    return evaluation


# ================= SUMMARIES =================


def finite_values(values):
    return [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def summarize_scalar_results(seed_results):
    rows_by_configuration = defaultdict(list)

    for row in seed_results:
        key = (row["family"], int(row["top_k"]))
        rows_by_configuration[key].append(row)

    summaries = []

    for config in MODEL_FAMILIES:
        for top_k in TOP_K_VALUES:
            rows = rows_by_configuration[(config["family"], top_k)]
            summary = {
                "classifier": model_name(config["family"], top_k),
                "family": config["family"],
                "top_k": top_k,
                "n_estimators": config["n_estimators"],
                "max_depth": config["max_depth"],
                "seeds": len(rows),
            }

            for metric_name in SUMMARY_METRICS:
                values = finite_values(row[metric_name] for row in rows)

                if values:
                    summary[f"{metric_name}_mean"] = mean(values)
                    summary[f"{metric_name}_std"] = pstdev(values)
                    summary[f"{metric_name}_min"] = min(values)
                    summary[f"{metric_name}_max"] = max(values)
                else:
                    summary[f"{metric_name}_mean"] = math.nan
                    summary[f"{metric_name}_std"] = math.nan
                    summary[f"{metric_name}_min"] = math.nan
                    summary[f"{metric_name}_max"] = math.nan

            summaries.append(summary)

    summaries.sort(
        key=lambda item: (
            item["macro_f1_mean"],
            item["weighted_f1_mean"],
            item["roc_auc_ovr_macro_mean"],
            -item["top_k"],
        ),
        reverse=True,
    )

    return summaries


def summarize_per_class_results(per_class_seed_rows):
    grouped_rows = defaultdict(list)

    for row in per_class_seed_rows:
        key = (row["family"], int(row["top_k"]), row["label"])
        grouped_rows[key].append(row)

    summaries = []

    for config in MODEL_FAMILIES:
        for top_k in TOP_K_VALUES:
            for label in CLASS_ORDER:
                rows = grouped_rows[(config["family"], top_k, label)]
                summary = {
                    "classifier": model_name(config["family"], top_k),
                    "family": config["family"],
                    "top_k": top_k,
                    "label": label,
                    "seeds": len(rows),
                }

                for metric_name in PER_CLASS_METRICS:
                    values = finite_values(row[metric_name] for row in rows)
                    summary[f"{metric_name}_mean"] = (
                        mean(values) if values else math.nan
                    )
                    summary[f"{metric_name}_std"] = (
                        pstdev(values) if values else math.nan
                    )

                summary["support_total"] = sum(
                    int(row["support"])
                    for row in rows
                )
                summaries.append(summary)

    return summaries


def aggregate_confusion_matrices(confusion_records):
    aggregate_counts = defaultdict(int)

    for record in confusion_records:
        key = (
            record["family"],
            int(record["top_k"]),
            record["true_label"],
            record["predicted_label"],
        )
        aggregate_counts[key] += int(record["count"])

    rows = []

    for config in MODEL_FAMILIES:
        family = config["family"]

        for top_k in TOP_K_VALUES:
            true_totals = {}

            for true_label in CLASS_ORDER:
                true_totals[true_label] = sum(
                    aggregate_counts[
                        (family, top_k, true_label, predicted_label)
                    ]
                    for predicted_label in CLASS_ORDER
                )

            for true_label in CLASS_ORDER:
                for predicted_label in CLASS_ORDER:
                    count = aggregate_counts[
                        (family, top_k, true_label, predicted_label)
                    ]
                    total = true_totals[true_label]

                    rows.append(
                        {
                            "classifier": model_name(family, top_k),
                            "family": family,
                            "top_k": top_k,
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                            "count": count,
                            "normalized_by_true_class": (
                                count / total if total else 0.0
                            ),
                        }
                    )

    return rows


def summarize_feature_selection(feature_selection_records):
    grouped = defaultdict(list)

    for row in feature_selection_records:
        key = (int(row["top_k"]), int(row["feature_index"]))
        grouped[key].append(row)

    summaries = []

    for top_k in TOP_K_VALUES:
        for feature_index in range(EXPECTED_FEATURES_PER_SAMPLE):
            rows = grouped[(top_k, feature_index)]

            if not rows:
                continue

            ranks = [int(row["rank"]) for row in rows]
            scores = [float(row["fisher_score"]) for row in rows]

            summaries.append(
                {
                    "top_k": top_k,
                    "feature_index": feature_index,
                    "selection_count": len(rows),
                    "selection_rate": len(rows) / len(SEEDS),
                    "mean_rank_when_selected": mean(ranks),
                    "mean_fisher_score_when_selected": mean(scores),
                }
            )

    summaries.sort(
        key=lambda item: (
            item["top_k"],
            -item["selection_count"],
            item["mean_rank_when_selected"],
        )
    )

    return summaries


def movement_f1_lookup(per_class_summary):
    return {
        (row["family"], int(row["top_k"])): row["f1_mean"]
        for row in per_class_summary
        if row["label"] == "movement"
    }


def select_compact_recommendations(summary_results, per_class_summary):
    movement_lookup = movement_f1_lookup(per_class_summary)
    recommendations = []

    for config in MODEL_FAMILIES:
        family = config["family"]
        family_rows = [
            row
            for row in summary_results
            if row["family"] == family
        ]

        best_macro_row = max(
            family_rows,
            key=lambda row: (
                row["macro_f1_mean"],
                row["weighted_f1_mean"],
                -int(row["top_k"]),
            ),
        )
        best_macro_f1 = best_macro_row["macro_f1_mean"]
        reference_movement_f1 = movement_lookup[
            (family, int(best_macro_row["top_k"]))
        ]

        eligible = [
            row
            for row in family_rows
            if row["macro_f1_mean"] >= best_macro_f1 - MAX_MACRO_F1_DROP
            and movement_lookup[(family, int(row["top_k"]))]
            >= reference_movement_f1 - MAX_MOVEMENT_F1_DROP
        ]

        selected = min(
            eligible,
            key=lambda row: (
                int(row["top_k"]),
                -row["macro_f1_mean"],
                row["actual_node_count_mean"],
            ),
        )

        recommendations.append(
            {
                **selected,
                "movement_f1_mean": movement_lookup[
                    (family, int(selected["top_k"]))
                ],
                "family_best_macro_f1": best_macro_f1,
                "reference_movement_f1": reference_movement_f1,
                "macro_f1_drop_from_family_best": (
                    best_macro_f1 - selected["macro_f1_mean"]
                ),
                "movement_f1_drop_from_macro_f1_reference": (
                    reference_movement_f1
                    - movement_lookup[(family, int(selected["top_k"]))]
                ),
            }
        )

    recommendations.sort(
        key=lambda row: (
            row["macro_f1_mean"],
            row["movement_f1_mean"],
            -int(row["top_k"]),
        ),
        reverse=True,
    )

    return recommendations


# ================= OUTPUT =================


def get_experiment_paths():
    run_dir = RESULTS_DIR / "runs" / EXPERIMENT_FOLDER_NAME
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "tables_dir": tables_dir,
        "seed_results": tables_dir / "feature_budget_results_by_seed.csv",
        "summary_results": tables_dir / "feature_budget_summary.csv",
        "per_class_seed_results": tables_dir / "per_class_results_by_seed.csv",
        "per_class_summary": tables_dir / "per_class_summary.csv",
        "confusion_by_seed": tables_dir / "confusion_matrices_by_seed.csv",
        "aggregated_confusion": tables_dir / "aggregated_confusion_matrices.csv",
        "feature_selection_frequency": tables_dir / "feature_selection_frequency.csv",
        "compact_recommendations": tables_dir / "compact_recommendations.csv",
        "experiment_notes": run_dir / "experiment_notes.txt",
    }


def save_csv(rows, output_path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_experiment_index(highest_mean_result):
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

    existing_rows.append(
        {
            "experiment_id": EXPERIMENT_ID,
            "folder": EXPERIMENT_FOLDER_NAME,
            "description": (
                "Feature-budget comparison for compact Gradient Boosting "
                "and XGBoost candidates"
            ),
            "model": highest_mean_result["classifier"],
            "validation_method": (
                f"File-based holdout repeated across "
                f"{highest_mean_result['seeds']} seeds"
            ),
            "top_k": highest_mean_result["top_k"],
            "max_depth": highest_mean_result["max_depth"],
            "min_samples_split": "",
            "accuracy": f"{highest_mean_result['accuracy_mean']:.6f}",
            "macro_f1": f"{highest_mean_result['macro_f1_mean']:.6f}",
            "weighted_f1": f"{highest_mean_result['weighted_f1_mean']:.6f}",
            "main_observation": (
                "Highest mean Macro F1 is recorded in the index. Final "
                "embedded selection must also consider movement F1, nodes, "
                "used features and the compact recommendation table."
            ),
        }
    )

    with open(index_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)


def save_experiment_notes(
    output_path,
    feature_dataset,
    summary_results,
    compact_recommendations,
):
    highest_mean = summary_results[0]

    lines = [
        "Experiment 025 - Feature-budget comparison",
        "",
        "Objective:",
        (
            "Find the smallest Fisher Score Top-K feature budget that "
            "preserves multiclass performance, especially movement F1, "
            "for compact Gradient Boosting and XGBoost models."
        ),
        "",
        "Dataset:",
        f"- Samples/windows: {len(feature_dataset)}",
        f"- Features per sample: {len(feature_dataset[0]['features'])}",
        "",
        "Validation:",
        f"- File-based stratified holdout: {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}",
        f"- Seeds: {SEEDS}",
        f"- Top-K values: {TOP_K_VALUES}",
        "- Fisher Score ranking computed using training data only for every seed",
        "- Both models receive the same split and feature indices within each seed",
        "",
        "Models:",
    ]

    for config in MODEL_FAMILIES:
        lines.append(f"- {config['display_name']}: {config}")

    lines.extend(
        [
            "",
            "Highest mean Macro F1 configuration:",
            f"- classifier: {highest_mean['classifier']}",
            f"- top_k: {highest_mean['top_k']}",
            f"- accuracy mean: {highest_mean['accuracy_mean']:.6f}",
            f"- macro F1 mean: {highest_mean['macro_f1_mean']:.6f}",
            f"- macro F1 std: {highest_mean['macro_f1_std']:.6f}",
            f"- weighted F1 mean: {highest_mean['weighted_f1_mean']:.6f}",
            f"- actual trees mean: {highest_mean['actual_tree_count_mean']:.2f}",
            f"- actual nodes mean: {highest_mean['actual_node_count_mean']:.2f}",
            f"- used features mean: {highest_mean['used_feature_count_mean']:.2f}",
            "",
            "Compact-selection tolerances:",
            f"- Maximum Macro F1 drop: {MAX_MACRO_F1_DROP:.3f}",
            f"- Maximum movement F1 drop: {MAX_MOVEMENT_F1_DROP:.3f}",
            "",
            "Compact recommendations by family:",
        ]
    )

    for row in compact_recommendations:
        lines.extend(
            [
                f"- {row['family']}:",
                f"  - top_k: {row['top_k']}",
                f"  - macro F1 mean: {row['macro_f1_mean']:.6f}",
                f"  - movement F1 mean: {row['movement_f1_mean']:.6f}",
                (
                    "  - Macro F1 drop from family best: "
                    f"{row['macro_f1_drop_from_family_best']:.6f}"
                ),
                (
                    "  - movement F1 drop from highest-Macro-F1 reference: "
                    f"{row['movement_f1_drop_from_macro_f1_reference']:.6f}"
                ),
                f"  - nodes mean: {row['actual_node_count_mean']:.2f}",
                f"  - used features mean: {row['used_feature_count_mean']:.2f}",
            ]
        )

    lines.extend(
        [
            "",
            "Interpretation rule:",
            (
                "The highest mean Macro F1 is not automatically the final "
                "embedded model. Prefer the smallest Top-K that stays within "
                "the stated Macro F1 and movement F1 tolerances, then inspect "
                "stability, confusion patterns, node count and used features."
            ),
        ]
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def save_partial_outputs(
    paths,
    seed_results,
    per_class_seed_rows,
    confusion_records,
    feature_selection_records,
):
    summary_results = summarize_scalar_results(seed_results)
    per_class_summary = summarize_per_class_results(per_class_seed_rows)
    aggregated_confusion = aggregate_confusion_matrices(confusion_records)
    feature_selection_frequency = summarize_feature_selection(
        feature_selection_records
    )

    save_csv(seed_results, paths["seed_results"])
    save_csv(summary_results, paths["summary_results"])
    save_csv(per_class_seed_rows, paths["per_class_seed_results"])
    save_csv(per_class_summary, paths["per_class_summary"])
    save_csv(confusion_records, paths["confusion_by_seed"])
    save_csv(aggregated_confusion, paths["aggregated_confusion"])
    save_csv(
        feature_selection_frequency,
        paths["feature_selection_frequency"],
    )


# ================= MAIN =================


def run_experiment():
    feature_dataset = load_pickle(FEATURE_DATASET_FILE)

    if not feature_dataset:
        raise ValueError("Feature dataset is empty.")

    features_per_sample = len(feature_dataset[0]["features"])

    if features_per_sample != EXPECTED_FEATURES_PER_SAMPLE:
        raise ValueError(
            "Experiment 025 expects the current 231-feature dataset. "
            f"Found {features_per_sample} features per sample."
        )

    if max(TOP_K_VALUES) > features_per_sample:
        raise ValueError(
            "TOP_K_VALUES contains a value greater than the feature count."
        )

    seed_results = []
    per_class_seed_rows = []
    confusion_records = []
    feature_selection_records = []
    paths = get_experiment_paths()

    for seed in SEEDS:
        print()
        print("=" * 78)
        print(f"Seed {seed}")
        print("=" * 78)

        full_train_dataset, full_test_dataset, _, _ = file_stratified_holdout_split(
            feature_dataset,
            test_size=TEST_SIZE,
            seed=seed,
        )

        print("Train samples:", dict(count_by_label(full_train_dataset)))
        print("Test samples:", dict(count_by_label(full_test_dataset)))

        ranking = rank_features_by_fisher_score(full_train_dataset)
        selected_datasets = {}

        for top_k in TOP_K_VALUES:
            selected_indices = [
                item["feature_index"]
                for item in ranking[:top_k]
            ]

            selected_datasets[top_k] = {
                "indices": selected_indices,
                "train": select_features_by_indices(
                    full_train_dataset,
                    selected_indices,
                ),
                "test": select_features_by_indices(
                    full_test_dataset,
                    selected_indices,
                ),
            }

            for rank_position, item in enumerate(ranking[:top_k], start=1):
                feature_selection_records.append(
                    {
                        "seed": seed,
                        "top_k": top_k,
                        "feature_index": item["feature_index"],
                        "rank": rank_position,
                        "fisher_score": item["score"],
                    }
                )

        for top_k in TOP_K_VALUES:
            selected = selected_datasets[top_k]

            for config in MODEL_FAMILIES:
                evaluation = fit_and_predict(
                    config=config,
                    train_dataset=selected["train"],
                    test_dataset=selected["test"],
                    selected_original_indices=selected["indices"],
                    seed=seed,
                )

                metrics = evaluation["metrics"]
                classifier = model_name(config["family"], top_k)

                result_row = {
                    "seed": seed,
                    "classifier": classifier,
                    "family": config["family"],
                    "top_k": top_k,
                    "n_estimators": config["n_estimators"],
                    "max_depth": config["max_depth"],
                    "train_samples": len(selected["train"]),
                    "test_samples": len(selected["test"]),
                }
                result_row.update(metrics)
                seed_results.append(result_row)

                for class_row in evaluation["per_class"]:
                    per_class_seed_rows.append(
                        {
                            "seed": seed,
                            "classifier": classifier,
                            "family": config["family"],
                            "top_k": top_k,
                            **class_row,
                        }
                    )

                matrix = evaluation["confusion_matrix"]
                normalized_matrix = evaluation["normalized_confusion_matrix"]

                for true_index, true_label in enumerate(CLASS_ORDER):
                    for predicted_index, predicted_label in enumerate(CLASS_ORDER):
                        confusion_records.append(
                            {
                                "seed": seed,
                                "classifier": classifier,
                                "family": config["family"],
                                "top_k": top_k,
                                "true_label": true_label,
                                "predicted_label": predicted_label,
                                "count": matrix[true_index][predicted_index],
                                "normalized_by_true_class": normalized_matrix[
                                    true_index
                                ][predicted_index],
                            }
                        )

                print(
                    f"{classifier:<34} "
                    f"macro_f1={metrics['macro_f1']:.6f} "
                    f"movement_f1={metrics['movement_f1']:.6f} "
                    f"nodes={metrics['actual_node_count']} "
                    f"used_features={metrics['used_feature_count']}"
                )

        save_partial_outputs(
            paths,
            seed_results,
            per_class_seed_rows,
            confusion_records,
            feature_selection_records,
        )

    summary_results = summarize_scalar_results(seed_results)
    per_class_summary = summarize_per_class_results(per_class_seed_rows)
    aggregated_confusion = aggregate_confusion_matrices(confusion_records)
    feature_selection_frequency = summarize_feature_selection(
        feature_selection_records
    )
    compact_recommendations = select_compact_recommendations(
        summary_results,
        per_class_summary,
    )

    save_csv(seed_results, paths["seed_results"])
    save_csv(summary_results, paths["summary_results"])
    save_csv(per_class_seed_rows, paths["per_class_seed_results"])
    save_csv(per_class_summary, paths["per_class_summary"])
    save_csv(confusion_records, paths["confusion_by_seed"])
    save_csv(aggregated_confusion, paths["aggregated_confusion"])
    save_csv(
        feature_selection_frequency,
        paths["feature_selection_frequency"],
    )
    save_csv(compact_recommendations, paths["compact_recommendations"])
    save_experiment_notes(
        paths["experiment_notes"],
        feature_dataset,
        summary_results,
        compact_recommendations,
    )
    update_experiment_index(summary_results[0])

    highest_mean = summary_results[0]

    print()
    print("=" * 78)
    print("Experiment 025 finished")
    print("=" * 78)
    print("Highest mean Macro F1:", highest_mean["classifier"])
    print(f"Macro F1 mean: {highest_mean['macro_f1_mean']:.6f}")
    print(f"Macro F1 std: {highest_mean['macro_f1_std']:.6f}")
    print()
    print("Compact recommendations:")

    for row in compact_recommendations:
        print(
            f"- {row['family']}: top_k={row['top_k']}, "
            f"macro_f1={row['macro_f1_mean']:.6f}, "
            f"movement_f1={row['movement_f1_mean']:.6f}, "
            f"nodes={row['actual_node_count_mean']:.2f}"
        )

    print("Results saved to:", paths["run_dir"])


def main():
    print()
    print("Feature Budget Comparison")
    print("=" * 78)
    print("Experiment:", EXPERIMENT_ID)
    print("Feature dataset:", FEATURE_DATASET_FILE)
    print("Seeds:", SEEDS)
    print("Top-K values:", TOP_K_VALUES)
    print("Model families:", len(MODEL_FAMILIES))

    run_experiment()


if __name__ == "__main__":
    main()
