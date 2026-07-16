"""Professor suggestion classifier comparison.

Experiment 024

Compares the current compact Gradient Boosting candidate against Logistic
Regression and compact XGBoost alternatives. All models are evaluated using
the same repeated file-based holdout splits and train-only Fisher ranking.

Reported metrics:
    - accuracy
    - macro F1
    - weighted F1
    - per-class precision, recall and F1
    - multiclass ROC-AUC One-vs-Rest (macro and weighted)
    - multiclass Average Precision / PR summary (macro and weighted)
    - per-class ROC-AUC and Average Precision
    - aggregated confusion matrices
    - training and inference time
"""

import csv
import math
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as error:
    raise ImportError(
        "scikit-learn is required for experiment 024. "
        "Install it with: python -m pip install scikit-learn"
    ) from error

try:
    from xgboost import XGBClassifier
except ImportError as error:
    raise ImportError(
        "XGBoost is required for experiment 024. "
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

EXPERIMENT_ID = "024"
EXPERIMENT_FOLDER_NAME = "024_professor_suggestions_comparison"

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
        "name": "gradient_boosting_depth3_20",
        "family": "gradient_boosting",
        "top_k": 231,
        "n_estimators": 20,
        "learning_rate": 0.10,
        "max_depth": 3,
        "estimated_model_units": 20,
    },
    {
        "name": "logistic_regression_c0_1",
        "family": "logistic_regression",
        "top_k": 231,
        "c": 0.1,
        "class_weight": None,
        "estimated_model_units": 1,
    },
    {
        "name": "logistic_regression_c1",
        "family": "logistic_regression",
        "top_k": 231,
        "c": 1.0,
        "class_weight": None,
        "estimated_model_units": 1,
    },
    {
        "name": "logistic_regression_c10",
        "family": "logistic_regression",
        "top_k": 231,
        "c": 10.0,
        "class_weight": None,
        "estimated_model_units": 1,
    },
    {
        "name": "logistic_regression_c1_balanced",
        "family": "logistic_regression",
        "top_k": 231,
        "c": 1.0,
        "class_weight": "balanced",
        "estimated_model_units": 1,
    },
    {
        "name": "xgboost_depth2_10",
        "family": "xgboost",
        "top_k": 231,
        "n_estimators": 10,
        "learning_rate": 0.10,
        "max_depth": 2,
        "estimated_model_units": 10,
    },
    {
        "name": "xgboost_depth2_20",
        "family": "xgboost",
        "top_k": 231,
        "n_estimators": 20,
        "learning_rate": 0.10,
        "max_depth": 2,
        "estimated_model_units": 20,
    },
    {
        "name": "xgboost_depth3_20",
        "family": "xgboost",
        "top_k": 231,
        "n_estimators": 20,
        "learning_rate": 0.10,
        "max_depth": 3,
        "estimated_model_units": 20,
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
]

PER_CLASS_METRICS = [
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
]


# ================= MODEL BUILDING =================


def build_model(config, seed):
    family = config["family"]

    if family == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            random_state=seed,
        )

    if family == "logistic_regression":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=config["c"],
                        class_weight=config["class_weight"],
                        max_iter=20000,
                        random_state=seed,
                    ),
                ),
            ]
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


def fit_and_predict(config, train_dataset, test_dataset, seed):
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

    evaluation["metrics"]["fit_time_ms"] = fit_time_ms
    evaluation["metrics"]["prediction_time_ms"] = prediction_time_ms

    return evaluation


# ================= SUMMARIES =================


def finite_values(values):
    return [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def summarize_scalar_results(seed_results):
    rows_by_model = defaultdict(list)

    for row in seed_results:
        rows_by_model[row["classifier"]].append(row)

    summaries = []

    for config in MODEL_CONFIGS:
        model_name = config["name"]
        rows = rows_by_model[model_name]

        summary = {
            "classifier": model_name,
            "family": config["family"],
            "top_k": config["top_k"],
            "estimated_model_units": config["estimated_model_units"],
            "max_depth": config.get("max_depth", ""),
            "n_estimators": config.get("n_estimators", ""),
            "c": config.get("c", ""),
            "class_weight": config.get("class_weight", ""),
            "seeds": len(rows),
        }

        for metric_name in SUMMARY_METRICS:
            values = finite_values(
                row[metric_name]
                for row in rows
            )

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
            item["average_precision_macro_mean"],
            -int(item["estimated_model_units"]),
        ),
        reverse=True,
    )

    return summaries


def summarize_per_class_results(per_class_seed_rows):
    grouped_rows = defaultdict(list)

    for row in per_class_seed_rows:
        key = (row["classifier"], row["label"])
        grouped_rows[key].append(row)

    summaries = []

    for config in MODEL_CONFIGS:
        for label in CLASS_ORDER:
            rows = grouped_rows[(config["name"], label)]
            summary = {
                "classifier": config["name"],
                "label": label,
                "seeds": len(rows),
            }

            for metric_name in PER_CLASS_METRICS:
                values = finite_values(
                    row[metric_name]
                    for row in rows
                )

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
            record["classifier"],
            record["true_label"],
            record["predicted_label"],
        )
        aggregate_counts[key] += int(record["count"])

    rows = []

    for config in MODEL_CONFIGS:
        classifier = config["name"]

        true_totals = {}

        for true_label in CLASS_ORDER:
            true_totals[true_label] = sum(
                aggregate_counts[(classifier, true_label, predicted_label)]
                for predicted_label in CLASS_ORDER
            )

        for true_label in CLASS_ORDER:
            for predicted_label in CLASS_ORDER:
                count = aggregate_counts[
                    (classifier, true_label, predicted_label)
                ]
                total = true_totals[true_label]
                normalized = count / total if total else 0.0

                rows.append(
                    {
                        "classifier": classifier,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": count,
                        "normalized_by_true_class": normalized,
                    }
                )

    return rows


# ================= OUTPUT =================


def get_experiment_paths():
    run_dir = RESULTS_DIR / "runs" / EXPERIMENT_FOLDER_NAME
    tables_dir = run_dir / "tables"

    tables_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "tables_dir": tables_dir,
        "seed_results": tables_dir / "classifier_results_by_seed.csv",
        "summary_results": tables_dir / "classifier_summary.csv",
        "per_class_seed_results": tables_dir / "per_class_results_by_seed.csv",
        "per_class_summary": tables_dir / "per_class_summary.csv",
        "confusion_by_seed": tables_dir / "confusion_matrices_by_seed.csv",
        "aggregated_confusion": tables_dir / "aggregated_confusion_matrices.csv",
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

    existing_rows.append(
        {
            "experiment_id": EXPERIMENT_ID,
            "folder": EXPERIMENT_FOLDER_NAME,
            "description": (
                "Professor suggestions: Logistic Regression, XGBoost and "
                "probability-based multiclass metrics"
            ),
            "model": best_result["classifier"],
            "validation_method": (
                f"File-based holdout repeated across {best_result['seeds']} seeds"
            ),
            "top_k": best_result["top_k"],
            "max_depth": best_result["max_depth"],
            "min_samples_split": "",
            "accuracy": f"{best_result['accuracy_mean']:.6f}",
            "macro_f1": f"{best_result['macro_f1_mean']:.6f}",
            "weighted_f1": f"{best_result['weighted_f1_mean']:.6f}",
            "main_observation": (
                "Models compared by repeated holdout using F1, ROC-AUC, "
                "Average Precision and aggregated confusion matrices"
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
):
    best_result = summary_results[0]

    lines = [
        "Experiment 024 - Professor suggestion classifier comparison",
        "",
        "Objective:",
        (
            "Compare Logistic Regression and compact XGBoost alternatives "
            "against the current Gradient Boosting candidate."
        ),
        "",
        "Dataset:",
        f"- Samples/windows: {len(feature_dataset)}",
        f"- Features per sample: {len(feature_dataset[0]['features'])}",
        "",
        "Validation:",
        f"- File-based stratified holdout: {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}",
        f"- Seeds: {SEEDS}",
        "- Fisher Score ranking computed using training data only for every seed",
        "- The same split is used by every classifier within each seed",
        "",
        "Metrics:",
        "- Accuracy",
        "- Macro F1 and Weighted F1",
        "- Per-class precision, recall and F1",
        "- Multiclass ROC-AUC One-vs-Rest, macro and weighted",
        "- Multiclass Average Precision, macro and weighted",
        "- Per-class ROC-AUC and Average Precision",
        "- Aggregated confusion matrices",
        "- Training and inference time",
        "",
        "Compared models:",
    ]

    for config in MODEL_CONFIGS:
        lines.append(f"- {config['name']}: {config}")

    lines.extend(
        [
            "",
            "Best result by mean Macro F1:",
            f"- classifier: {best_result['classifier']}",
            f"- accuracy mean: {best_result['accuracy_mean']:.6f}",
            f"- macro F1 mean: {best_result['macro_f1_mean']:.6f}",
            f"- macro F1 std: {best_result['macro_f1_std']:.6f}",
            f"- weighted F1 mean: {best_result['weighted_f1_mean']:.6f}",
            f"- ROC-AUC OvR macro mean: {best_result['roc_auc_ovr_macro_mean']:.6f}",
            f"- Average Precision macro mean: {best_result['average_precision_macro_mean']:.6f}",
            "",
            "Selection rule:",
            (
                "Macro F1 remains the main ranking metric. Weighted F1, "
                "ROC-AUC, Average Precision, confusion patterns, stability "
                "and embedded cost are used as complementary evidence."
            ),
        ]
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
            "Experiment 024 expects the current 231-feature dataset. "
            f"Found {features_per_sample} features per sample."
        )

    seed_results = []
    per_class_seed_rows = []
    confusion_records = []
    paths = get_experiment_paths()

    unique_top_k_values = sorted(
        {config["top_k"] for config in MODEL_CONFIGS}
    )

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

        for top_k in unique_top_k_values:
            if top_k > len(ranking):
                raise ValueError(
                    f"Invalid top_k={top_k}; ranking has {len(ranking)} features."
                )

            selected_indices = [
                item["feature_index"]
                for item in ranking[:top_k]
            ]

            selected_datasets[top_k] = (
                select_features_by_indices(
                    full_train_dataset,
                    selected_indices,
                ),
                select_features_by_indices(
                    full_test_dataset,
                    selected_indices,
                ),
            )

        for config in MODEL_CONFIGS:
            train_dataset, test_dataset = selected_datasets[config["top_k"]]

            evaluation = fit_and_predict(
                config,
                train_dataset,
                test_dataset,
                seed,
            )

            metrics = evaluation["metrics"]

            result_row = {
                "seed": seed,
                "classifier": config["name"],
                "family": config["family"],
                "top_k": config["top_k"],
                "estimated_model_units": config["estimated_model_units"],
                "max_depth": config.get("max_depth", ""),
                "n_estimators": config.get("n_estimators", ""),
                "c": config.get("c", ""),
                "class_weight": config.get("class_weight", ""),
                "train_samples": len(train_dataset),
                "test_samples": len(test_dataset),
            }
            result_row.update(metrics)
            seed_results.append(result_row)

            for class_row in evaluation["per_class"]:
                per_class_seed_rows.append(
                    {
                        "seed": seed,
                        "classifier": config["name"],
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
                            "classifier": config["name"],
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                            "count": matrix[true_index][predicted_index],
                            "normalized_by_true_class": normalized_matrix[
                                true_index
                            ][predicted_index],
                        }
                    )

            print(
                f"{config['name']:<36} "
                f"macro_f1={metrics['macro_f1']:.6f} "
                f"weighted_f1={metrics['weighted_f1']:.6f} "
                f"roc_auc={metrics['roc_auc_ovr_macro']:.6f} "
                f"ap={metrics['average_precision_macro']:.6f}"
            )

        # Preserve partial results after every completed seed.
        # This makes long comparisons resumable for manual analysis even if
        # execution is interrupted before the final summary is produced.
        partial_summary = summarize_scalar_results(seed_results)
        partial_per_class_summary = summarize_per_class_results(
            per_class_seed_rows
        )
        partial_aggregated_confusion = aggregate_confusion_matrices(
            confusion_records
        )

        save_csv(seed_results, paths["seed_results"])
        save_csv(partial_summary, paths["summary_results"])
        save_csv(per_class_seed_rows, paths["per_class_seed_results"])
        save_csv(
            partial_per_class_summary,
            paths["per_class_summary"],
        )
        save_csv(confusion_records, paths["confusion_by_seed"])
        save_csv(
            partial_aggregated_confusion,
            paths["aggregated_confusion"],
        )

    summary_results = summarize_scalar_results(seed_results)
    per_class_summary = summarize_per_class_results(per_class_seed_rows)
    aggregated_confusion = aggregate_confusion_matrices(confusion_records)

    save_csv(seed_results, paths["seed_results"])
    save_csv(summary_results, paths["summary_results"])
    save_csv(per_class_seed_rows, paths["per_class_seed_results"])
    save_csv(per_class_summary, paths["per_class_summary"])
    save_csv(confusion_records, paths["confusion_by_seed"])
    save_csv(aggregated_confusion, paths["aggregated_confusion"])
    save_experiment_notes(
        paths["experiment_notes"],
        feature_dataset,
        summary_results,
    )
    update_experiment_index(summary_results[0])

    best = summary_results[0]

    print()
    print("=" * 78)
    print("Experiment 024 finished")
    print("=" * 78)
    print(f"Best classifier: {best['classifier']}")
    print(f"Macro F1 mean: {best['macro_f1_mean']:.6f}")
    print(f"Macro F1 std: {best['macro_f1_std']:.6f}")
    print(f"Weighted F1 mean: {best['weighted_f1_mean']:.6f}")
    print(f"ROC-AUC OvR macro mean: {best['roc_auc_ovr_macro_mean']:.6f}")
    print(f"Average Precision macro mean: {best['average_precision_macro_mean']:.6f}")
    print("Results saved to:", paths["run_dir"])


def main():
    print()
    print("Professor Suggestion Classifier Comparison")
    print("=" * 78)
    print("Experiment:", EXPERIMENT_ID)
    print("Feature dataset:", FEATURE_DATASET_FILE)
    print("Seeds:", SEEDS)
    print("Models:", len(MODEL_CONFIGS))

    run_experiment()


if __name__ == "__main__":
    main()
