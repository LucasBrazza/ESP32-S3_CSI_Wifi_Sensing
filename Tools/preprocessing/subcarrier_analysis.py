"""
Subcarrier Relevance Analysis

This module analyzes the output of the Fisher Score ranking in terms of
subcarrier relevance.

Feature extraction generates multiple features for each subcarrier:

    mean
    std
    min
    max
    peak_to_peak
    energy

Because each subcarrier generates six features, a feature index can be
mapped back to:

    subcarrier index
    feature type

This module is useful for understanding whether the Fisher Score is
selecting isolated features or repeatedly selecting features from the
same subcarriers.

If several high-ranked features belong to the same subcarrier, this is
an indication that the subcarrier may be highly informative for the
classification problem.

This stage is mainly diagnostic and analytical. It helps interpret the
model and supports decisions about which subcarriers may be useful for
future embedded implementations.
"""

FEATURES_PER_SUBCARRIER = 6

FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "peak_to_peak",
    "energy",
]

"""
Each subcarrier generates the same fixed number of features.

The order defined here must match the order used during feature
extraction.

Feature index mapping:

    feature_index = subcarrier_index × FEATURES_PER_SUBCARRIER
                    + feature_position

Example:

    FEATURES_PER_SUBCARRIER = 6

    feature_index = 14

    subcarrier_index = 14 // 6 = 2
    feature_position = 14 % 6 = 2

    FEATURE_NAMES[2] = "min"

So feature 14 corresponds to:

    subcarrier 2, minimum value
"""


def feature_index_to_subcarrier(feature_index):
    """
    Convert a global feature index into its corresponding subcarrier
    and feature type.

    Formula:

        subcarrier = feature_index // FEATURES_PER_SUBCARRIER

        feature_position = feature_index % FEATURES_PER_SUBCARRIER

    where:

        // = integer division
        %  = remainder operation

    This mapping is necessary because the feature vector is stored as a
    flat list, while conceptually each group of features belongs to one
    subcarrier.
    """
    subcarrier = feature_index // FEATURES_PER_SUBCARRIER
    feature_position = feature_index % FEATURES_PER_SUBCARRIER
    feature_name = FEATURE_NAMES[feature_position]

    return subcarrier, feature_name


def count_subcarriers_from_ranking(ranking, top_n=30):
    """
    Count how many times each subcarrier appears among the top-ranked
    Fisher Score features.

    The ranking input is expected to contain dictionaries with:

        {
            "feature_index": index,
            "score": fisher_score
        }

    For each selected feature, the feature index is converted back into:

        subcarrier
        feature name

    The function returns a dictionary containing:

        count:
            how many selected features belong to this subcarrier

        features:
            which feature types were selected

        scores:
            Fisher scores associated with those features

    This helps identify subcarriers that are repeatedly considered
    relevant by the feature selection stage.
    """
    subcarrier_count = {}

    for item in ranking[:top_n]:
        feature_index = item["feature_index"]

        subcarrier, feature_name = feature_index_to_subcarrier(
            feature_index
        )

        if subcarrier not in subcarrier_count:
            subcarrier_count[subcarrier] = {
                "count": 0,
                "features": [],
                "scores": [],
            }

        subcarrier_count[subcarrier]["count"] += 1
        subcarrier_count[subcarrier]["features"].append(feature_name)
        subcarrier_count[subcarrier]["scores"].append(item["score"])

    return subcarrier_count


def rank_subcarriers_by_occurrence(ranking, top_n=30):
    """
    Rank subcarriers according to their presence in the top Fisher
    Score features.

    Sorting priority:

        1. Number of occurrences
        2. Average Fisher Score

    Average score formula:

        avg_score = Σ score_i / N

    where:

        score_i = Fisher Score of each selected feature
        N       = number of selected features for that subcarrier

    A subcarrier with several high-ranked features is considered more
    consistently relevant than a subcarrier that appears only once.
    """
    subcarrier_count = count_subcarriers_from_ranking(
        ranking,
        top_n=top_n,
    )

    ranked = []

    for subcarrier, data in subcarrier_count.items():
        scores = data["scores"]

        avg_score = sum(scores) / len(scores)

        ranked.append(
            {
                "subcarrier": subcarrier,
                "count": data["count"],
                "features": data["features"],
                "avg_score": avg_score,
                "max_score": max(scores),
            }
        )

    ranked.sort(
        key=lambda item: (
            item["count"],
            item["avg_score"],
        ),
        reverse=True,
    )

    return ranked


def select_subcarriers_from_ranking(ranking, top_n=30, min_count=1):
    """
    Select subcarriers based on their occurrence count in the top-ranked
    Fisher Score features.

    Selection rule:

        keep subcarrier if:

            occurrence_count >= min_count

    Example:

        min_count = 2

        subcarrier appears once  -> discard
        subcarrier appears twice -> keep

    This can be used as an additional interpretability tool to identify
    subcarriers that are repeatedly associated with relevant features.
    """
    ranked_subcarriers = rank_subcarriers_by_occurrence(
        ranking,
        top_n=top_n,
    )

    selected = []

    for item in ranked_subcarriers:
        if item["count"] >= min_count:
            selected.append(item["subcarrier"])

    return selected


def print_ranked_subcarriers(ranked_subcarriers):
    print()
    print("Subportadoras mais relevantes:")

    for item in ranked_subcarriers:
        print(
            "SC:",
            item["subcarrier"],
            "| Ocorrências:",
            item["count"],
            "| Features:",
            item["features"],
            "| Score médio:",
            item["avg_score"],
            "| Score máximo:",
            item["max_score"],
        )


