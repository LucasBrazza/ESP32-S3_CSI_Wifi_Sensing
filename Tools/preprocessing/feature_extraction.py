"""
Feature Extraction

This module converts CSI windows into numerical feature vectors that can
be used by machine learning algorithms.

Input:

    window[packet][subcarrier]

Output:

    features[feature_index]

Instead of feeding the classifier with the raw CSI signal, a set of
statistical and temporal descriptors is extracted from each subcarrier.

Current features:

    mean
    standard deviation
    minimum
    maximum
    peak-to-peak
    energy
    mean absolute difference
    maximum absolute difference
    difference energy
    standard deviation of differences
    linear slope

The first group summarizes the amplitude distribution inside the window.
The second group describes how the signal changes over time, which is
especially relevant for distinguishing static presence from movement.

Pipeline position:

    sliding window
    ↓
    feature extraction
    ↓
    Fisher Score
    ↓
    decision tree

The selected features were intentionally chosen because they are
computationally inexpensive and can be efficiently implemented in
embedded systems such as the ESP32-S3.
"""

from Tools.preprocessing.core_math import mean, std, get_column


def minimum(values):
    """
    Compute the minimum value of a signal.
    """
    if not values:
        return 0.0

    min_value = values[0]

    for value in values:
        if value < min_value:
            min_value = value

    return min_value


def maximum(values):
    """
    Compute the maximum value of a signal.
    """
    if not values:
        return 0.0

    max_value = values[0]

    for value in values:
        if value > max_value:
            max_value = value

    return max_value


def energy(values):
    """
    Compute the signal energy.

    Formula:

        Energy = Σ Xi²
    """
    total = 0.0

    for value in values:
        total += value * value

    return total


def first_order_differences(values):
    """
    Compute first-order temporal differences.

    Formula:

        Di = Xi - X(i-1)

    These differences describe how the signal changes between consecutive
    packets inside the same window.
    """
    differences = []

    if len(values) < 2:
        return differences

    for index in range(1, len(values)):
        differences.append(values[index] - values[index - 1])

    return differences


def mean_abs_diff(values):
    """
    Compute the mean absolute first-order difference.

    This feature measures the average temporal variation of the signal.
    Movement is expected to produce larger variations than static states.
    """
    differences = first_order_differences(values)

    if not differences:
        return 0.0

    absolute_values = []

    for value in differences:
        if value < 0:
            absolute_values.append(-value)
        else:
            absolute_values.append(value)

    return mean(absolute_values)


def max_abs_diff(values):
    """
    Compute the maximum absolute first-order difference.

    This feature captures the strongest abrupt temporal variation inside
    the window.
    """
    differences = first_order_differences(values)

    if not differences:
        return 0.0

    max_value = 0.0

    for value in differences:
        if value < 0:
            abs_value = -value
        else:
            abs_value = value

        if abs_value > max_value:
            max_value = abs_value

    return max_value


def diff_energy(values):
    """
    Compute the energy of first-order differences.

    Formula:

        DiffEnergy = Σ Di²

    This emphasizes windows with stronger temporal dynamics.
    """
    differences = first_order_differences(values)

    total = 0.0

    for value in differences:
        total += value * value

    return total


def std_diff(values):
    """
    Compute the standard deviation of first-order differences.

    This measures how irregular the temporal variation is inside the
    window.
    """
    differences = first_order_differences(values)

    if not differences:
        return 0.0

    return std(differences)


def slope(values):
    """
    Compute the linear slope of the signal inside the window.

    The slope estimates whether the signal has a rising or falling trend
    over time.

    It is computed using simple linear regression with sample index as x.
    """
    n = len(values)

    if n < 2:
        return 0.0

    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2 = 0.0

    for index, value in enumerate(values):
        x = float(index)
        y = value

        sum_x += x
        sum_y += y
        sum_xy += x * y
        sum_x2 += x * x

    denominator = n * sum_x2 - sum_x * sum_x

    if denominator == 0:
        return 0.0

    return (n * sum_xy - sum_x * sum_y) / denominator


def extract_features_from_window(window):
    """
    Extract statistical and temporal features from a CSI window.

    Input:

        window[packet][subcarrier]

    Output:

        features[feature_index]

    For each subcarrier, the following descriptors are extracted:

        mean
        standard deviation
        minimum
        maximum
        peak-to-peak
        energy
        mean absolute difference
        maximum absolute difference
        difference energy
        standard deviation of differences
        linear slope

    Example:

        31 subcarriers
        11 features per subcarrier

        total_features = 31 × 11 = 341
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

        features.append(mean_abs_diff(signal))
        features.append(max_abs_diff(signal))
        features.append(diff_energy(signal))
        features.append(std_diff(signal))
        features.append(slope(signal))

    return features


def extract_features_from_labeled_window(labeled_window):
    """
    Convert a labeled CSI window into a labeled feature vector.

    The metadata is preserved while the raw signal window is replaced by
    its extracted feature vector.
    """

    item = {}

    for key in labeled_window:
        if key != "data":
            item[key] = labeled_window[key]

    item["features"] = extract_features_from_window(
        labeled_window["data"]
    )

    return item


def extract_feature_dataset(labeled_windows):
    """
    Convert a complete window dataset into a feature dataset.
    """

    dataset = []

    for labeled_window in labeled_windows:
        item = extract_features_from_labeled_window(
            labeled_window
        )

        dataset.append(item)

    return dataset