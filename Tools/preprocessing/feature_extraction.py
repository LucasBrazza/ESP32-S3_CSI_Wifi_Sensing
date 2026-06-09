from core_math import mean, std, get_column


def minimum(values):
    if not values:
        return 0.0

    min_value = values[0]

    for value in values:
        if value < min_value:
            min_value = value

    return min_value


def maximum(values):
    if not values:
        return 0.0

    max_value = values[0]

    for value in values:
        if value > max_value:
            max_value = value

    return max_value


def energy(values):
    total = 0.0

    for value in values:
        total += value * value

    return total


def extract_features_from_window(window):
    """
    Extrai features de uma janela CSI.

    Entrada:
        window[pacote][subportadora]

    Saída:
        features: lista numérica
    """

    features = []

    if not window:
        return features

    num_subcarriers = len(window[0])

    for sc in range(num_subcarriers):
        signal = get_column(window, sc)

        min_value = minimum(signal)
        max_value = maximum(signal)

        features.append(mean(signal))
        features.append(std(signal))
        features.append(min_value)
        features.append(max_value)
        features.append(max_value - min_value)
        features.append(energy(signal))

    return features


def extract_features_from_labeled_window(labeled_window):
    """
    Entrada:
        {
            "label": "empty",
            "data": window
        }

    Saída:
        {
            "label": "empty",
            "features": [...]
        }
    """

    return {
        "label": labeled_window["label"],
        "features": extract_features_from_window(
            labeled_window["data"]
        ),
    }


def extract_feature_dataset(labeled_windows):
    """
    Converte dataset de janelas em dataset de features.
    """

    dataset = []

    for labeled_window in labeled_windows:
        item = extract_features_from_labeled_window(
            labeled_window
        )

        dataset.append(item)

    return dataset

