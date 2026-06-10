"""
Feature Selection using Fisher Score

This module ranks and selects the most discriminative features extracted
from CSI windows.

After feature extraction, each window may contain many features. For
example:

    31 subcarriers × 6 features = 186 features

Not all of these features are useful for classification. Some may be
redundant, noisy or weakly related to the target classes.

Fisher Score is used to measure how well each feature separates the
classes.

A good feature should have:

    - high separation between class means
    - low variation within each class

In other words, the feature should produce clearly different values for
different classes while remaining stable inside the same class.

Pipeline position:

    feature extraction
    ↓
    Fisher Score ranking
    ↓
    top-K feature selection
    ↓
    decision tree training
"""

from core_math import mean


def group_feature_values_by_label(feature_dataset, feature_index):
    """
    Group the values of a single feature by class label.

    Input format:

        {
            "label": class_name,
            "features": [...]
        }

    Output example:

        {
            "empty": [values...],
            "static_presence": [values...],
            "movement": [values...]
        }

    This grouping is required because Fisher Score compares the behavior
    of each feature across different classes.
    """
    groups = {}

    for item in feature_dataset:
        label = item["label"]
        value = item["features"][feature_index]

        if label not in groups:
            groups[label] = []

        groups[label].append(value)

    return groups


def variance(values):
    """
    Compute the population variance of a signal.

    Formula:

        variance = Σ(x_i - μ)² / N

    where:

        x_i = each value
        μ   = mean of the values
        N   = number of values

    Variance measures how spread out the values are around their mean.

    In Fisher Score, lower within-class variance is desirable because it
    means that samples from the same class behave consistently.
    """
    if not values:
        return 0.0

    avg = mean(values)
    total = 0.0

    for value in values:
        total += (value - avg) ** 2

    return total / len(values)


def fisher_score_for_feature(feature_dataset, feature_index, eps=1e-8):
    """
    Compute the Fisher Score for one feature.

    Fisher Score measures how discriminative a feature is.

    Formula:

                    Σ n_c (μ_c - μ)²
        Fisher = -----------------------
                    Σ n_c σ_c²

    where:

        c     = class
        n_c   = number of samples in class c
        μ_c   = mean of the feature in class c
        μ     = global mean of the feature
        σ_c²  = variance of the feature in class c

    Interpretation:

        High Fisher Score:
            class means are far apart and within-class variance is low

        Low Fisher Score:
            class means are close together or within-class variance is high

    The small epsilon avoids division by zero when the denominator is
    very close to zero.
    """
    groups = group_feature_values_by_label(
        feature_dataset,
        feature_index,
    )

    all_values = []

    for item in feature_dataset:
        all_values.append(item["features"][feature_index])

    global_mean = mean(all_values)

    numerator = 0.0
    denominator = 0.0

    for label in groups:
        values = groups[label]

        class_mean = mean(values)
        class_variance = variance(values)

        n_class = len(values)

        numerator += n_class * ((class_mean - global_mean) ** 2)
        denominator += n_class * class_variance

    return numerator / (denominator + eps)


def rank_features_by_fisher_score(feature_dataset):
    """
    Rank all features according to their Fisher Score.

    Each feature is evaluated independently.

    Output format:

        [
            {
                "feature_index": index,
                "score": fisher_score
            }
        ]

    The ranking is sorted from the most discriminative feature to the
    least discriminative feature.
    """
    if not feature_dataset:
        return []

    num_features = len(feature_dataset[0]["features"])

    ranking = []

    for feature_index in range(num_features):
        score = fisher_score_for_feature(
            feature_dataset,
            feature_index,
        )

        ranking.append(
            {
                "feature_index": feature_index,
                "score": score,
            }
        )

    ranking.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return ranking


def select_top_features(feature_dataset, ranking, top_k=30):
    """
    Select the top-K features from a Fisher Score ranking.

    Selection rule:

        keep the first top_k features from the ranking

    Example:

        top_k = 30

        original feature vector: 186 features
        selected feature vector: 30 features

    This reduces dimensionality and computational cost before training
    the classifier.

    The function returns:

        selected_dataset:
            dataset containing only the selected features

        selected_indices:
            original feature indices kept from the full feature vector

    The selected indices must be saved because the real-time pipeline
    must apply the exact same feature selection order during inference.
    """
    selected_indices = []

    for item in ranking[:top_k]:
        selected_indices.append(item["feature_index"])

    selected_dataset = []

    for item in feature_dataset:
        selected_features = []

        for index in selected_indices:
            selected_features.append(item["features"][index])

        selected_dataset.append(
            {
                "label": item["label"],
                "features": selected_features,
            }
        )

    return selected_dataset, selected_indices