FEATURES_PER_SUBCARRIER = 6

FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "peak_to_peak",
    "energy",
]


def feature_index_to_subcarrier(feature_index):
    subcarrier = feature_index // FEATURES_PER_SUBCARRIER
    feature_position = feature_index % FEATURES_PER_SUBCARRIER
    feature_name = FEATURE_NAMES[feature_position]

    return subcarrier, feature_name


def count_subcarriers_from_ranking(ranking, top_n=30):
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


