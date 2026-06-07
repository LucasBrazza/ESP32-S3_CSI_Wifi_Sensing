from core_math import mean


def group_feature_values_by_label(feature_dataset, feature_index):
    groups = {}

    for item in feature_dataset:
        label = item["label"]
        value = item["features"][feature_index]

        if label not in groups:
            groups[label] = []

        groups[label].append(value)

    return groups


def variance(values):
    if not values:
        return 0.0

    avg = mean(values)
    total = 0.0

    for value in values:
        total += (value - avg) ** 2

    return total / len(values)


def fisher_score_for_feature(feature_dataset, feature_index, eps=1e-8):
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


if __name__ == "__main__":
    fake_dataset = [
        {"label": "empty", "features": [1.0, 5.0, 0.1]},
        {"label": "empty", "features": [1.1, 5.1, 0.2]},
        {"label": "movement", "features": [5.0, 5.0, 0.3]},
        {"label": "movement", "features": [5.2, 5.1, 0.4]},
    ]

    ranking = rank_features_by_fisher_score(fake_dataset)

    print("Ranking:")

    for item in ranking:
        print(item)

    selected, indices = select_top_features(
        fake_dataset,
        ranking,
        top_k=2,
    )

    print()
    print("Índices selecionados:", indices)
    print("Features selecionadas:", selected)