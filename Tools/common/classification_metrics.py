"""Common multiclass classification metrics for training experiments.

The functions in this module evaluate hard predictions and probability scores.
Probability-based metrics use a fixed class order so results from different
classifiers remain directly comparable.
"""

import math

import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize
except ImportError as error:
    raise ImportError(
        "scikit-learn is required for classification metrics. "
        "Install it with: python -m pip install scikit-learn"
    ) from error


def align_probability_columns(probabilities, model_classes, class_order):
    """Reorder predict_proba columns to match class_order."""
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 2:
        raise ValueError("Probability output must be a 2D array.")

    model_classes = list(model_classes)
    class_order = list(class_order)

    if probabilities.shape[1] != len(model_classes):
        raise ValueError(
            "Probability column count does not match the model class count."
        )

    missing_classes = [
        label
        for label in class_order
        if label not in model_classes
    ]

    if missing_classes:
        raise ValueError(
            "The trained model did not expose all expected classes: "
            f"{missing_classes}"
        )

    aligned_columns = [
        probabilities[:, model_classes.index(label)]
        for label in class_order
    ]

    aligned = np.column_stack(aligned_columns)
    row_sums = aligned.sum(axis=1, keepdims=True)

    if np.any(row_sums <= 0.0):
        raise ValueError("At least one probability row has a non-positive sum.")

    return aligned / row_sums


def _safe_probability_metrics(true_labels, probabilities, class_order):
    """Compute multiclass ROC-AUC and Average Precision metrics."""
    true_labels = list(true_labels)
    class_order = list(class_order)
    probabilities = np.asarray(probabilities, dtype=float)

    y_binary = label_binarize(
        true_labels,
        classes=class_order,
    )

    probability_metrics = {}

    try:
        probability_metrics["roc_auc_ovr_macro"] = roc_auc_score(
            y_binary,
            probabilities,
            average="macro",
        )
        probability_metrics["roc_auc_ovr_weighted"] = roc_auc_score(
            y_binary,
            probabilities,
            average="weighted",
        )
        per_class_roc_auc = roc_auc_score(
            y_binary,
            probabilities,
            average=None,
        )
    except ValueError:
        probability_metrics["roc_auc_ovr_macro"] = math.nan
        probability_metrics["roc_auc_ovr_weighted"] = math.nan
        per_class_roc_auc = [math.nan] * len(class_order)

    try:
        probability_metrics["average_precision_macro"] = average_precision_score(
            y_binary,
            probabilities,
            average="macro",
        )
        probability_metrics["average_precision_weighted"] = average_precision_score(
            y_binary,
            probabilities,
            average="weighted",
        )
        per_class_average_precision = average_precision_score(
            y_binary,
            probabilities,
            average=None,
        )
    except ValueError:
        probability_metrics["average_precision_macro"] = math.nan
        probability_metrics["average_precision_weighted"] = math.nan
        per_class_average_precision = [math.nan] * len(class_order)

    for index, label in enumerate(class_order):
        probability_metrics[f"{label}_roc_auc"] = float(
            per_class_roc_auc[index]
        )
        probability_metrics[f"{label}_average_precision"] = float(
            per_class_average_precision[index]
        )

    return probability_metrics


def evaluate_multiclass_predictions(
    true_labels,
    predicted_labels,
    class_order,
    probabilities=None,
):
    """Return scalar metrics, per-class metrics and confusion matrices."""
    true_labels = list(true_labels)
    predicted_labels = list(predicted_labels)
    class_order = list(class_order)

    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have equal length.")

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=class_order,
    )

    normalized_matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=class_order,
        normalize="true",
    )

    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=class_order,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "macro_f1": f1_score(
            true_labels,
            predicted_labels,
            labels=class_order,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            true_labels,
            predicted_labels,
            labels=class_order,
            average="weighted",
            zero_division=0,
        ),
    }

    per_class_rows = []

    for index, label in enumerate(class_order):
        class_metrics = {
            "label": label,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(per_class_f1[index]),
            "support": int(support[index]),
        }
        per_class_rows.append(class_metrics)

        metrics[f"{label}_precision"] = class_metrics["precision"]
        metrics[f"{label}_recall"] = class_metrics["recall"]
        metrics[f"{label}_f1"] = class_metrics["f1"]
        metrics[f"{label}_support"] = class_metrics["support"]

    if probabilities is not None:
        probability_metrics = _safe_probability_metrics(
            true_labels,
            probabilities,
            class_order,
        )
        metrics.update(probability_metrics)

        for row in per_class_rows:
            label = row["label"]
            row["roc_auc"] = metrics[f"{label}_roc_auc"]
            row["average_precision"] = metrics[
                f"{label}_average_precision"
            ]

    return {
        "metrics": metrics,
        "per_class": per_class_rows,
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": normalized_matrix.tolist(),
    }
