"""
Feature Extraction

This module converts CSI windows into numerical feature vectors that can
be used by machine learning algorithms.

Input:

    window[packet][subcarrier]

Output:

    features[feature_index]

The selected features were intentionally chosen because they are
computationally inexpensive and can be implemented in embedded systems
such as the ESP32-S3.

Current features per subcarrier:

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
    interquartile range
    median absolute deviation
    sliding variance mean
    sliding variance standard deviation
    autocorrelation lag 1
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


def variance(values):
    """
    Compute population variance.

    Formula:

        Var = Σ(Xi - mean)² / N
    """
    if not values:
        return 0.0

    avg = mean(values)
    total = 0.0

    for value in values:
        difference = value - avg
        total += difference * difference

    return total / len(values)


def first_order_differences(values):
    """
    Compute first-order temporal differences.

    Formula:

        Di = Xi - X(i-1)
    """
    differences = []

    if len(values) < 2:
        return differences

    for index in range(1, len(values)):
        differences.append(values[index] - values[index - 1])

    return differences


def absolute_value(value):
    """
    Compute absolute value without relying on external libraries.
    """
    if value < 0:
        return -value

    return value


def mean_abs_diff(values):
    """
    Compute the mean absolute first-order difference.
    """
    differences = first_order_differences(values)

    if not differences:
        return 0.0

    absolute_values = []

    for value in differences:
        absolute_values.append(absolute_value(value))

    return mean(absolute_values)


def max_abs_diff(values):
    """
    Compute the maximum absolute first-order difference.
    """
    differences = first_order_differences(values)

    if not differences:
        return 0.0

    max_value = 0.0

    for value in differences:
        abs_value = absolute_value(value)

        if abs_value > max_value:
            max_value = abs_value

    return max_value


def diff_energy(values):
    """
    Compute the energy of first-order differences.

    Formula:

        DiffEnergy = Σ Di²
    """
    differences = first_order_differences(values)

    total = 0.0

    for value in differences:
        total += value * value

    return total


def std_diff(values):
    """
    Compute the standard deviation of first-order differences.
    """
    differences = first_order_differences(values)

    if not differences:
        return 0.0

    return std(differences)


def slope(values):
    """
    Compute the linear slope of the signal inside the window.

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


def median(values):
    """
    Compute median value.
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)
    middle = n // 2

    if n % 2 == 1:
        return sorted_values[middle]

    return (sorted_values[middle - 1] + sorted_values[middle]) / 2.0


def percentile(values, percent):
    """
    Compute percentile using linear interpolation.

    This is still simple enough for small embedded windows.
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)

    if n == 1:
        return sorted_values[0]

    position = (percent / 100.0) * (n - 1)
    lower_index = int(position)
    upper_index = lower_index + 1

    if upper_index >= n:
        return sorted_values[lower_index]

    fraction = position - lower_index

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]

    return lower_value + fraction * (upper_value - lower_value)


def iqr(values):
    """
    Compute interquartile range.

    Formula:

        IQR = Q3 - Q1
    """
    if not values:
        return 0.0

    q1 = percentile(values, 25.0)
    q3 = percentile(values, 75.0)

    return q3 - q1


def mad(values):
    """
    Compute median absolute deviation.

    Formula:

        MAD = median(|Xi - median(X)|)
    """
    if not values:
        return 0.0

    center = median(values)
    deviations = []

    for value in values:
        deviations.append(absolute_value(value - center))

    return median(deviations)


def sliding_variances(values, window_size=3):
    """
    Compute local variances using a small sliding window.
    """
    local_variances = []

    if not values:
        return local_variances

    if len(values) < window_size:
        local_variances.append(variance(values))
        return local_variances

    start = 0

    while start + window_size <= len(values):
        local_window = []

        for index in range(start, start + window_size):
            local_window.append(values[index])

        local_variances.append(variance(local_window))

        start += 1

    return local_variances


def sliding_variance_mean(values):
    """
    Compute the mean of local sliding variances.
    """
    local_variances = sliding_variances(values, window_size=3)

    if not local_variances:
        return 0.0

    return mean(local_variances)


def sliding_variance_std(values):
    """
    Compute the standard deviation of local sliding variances.
    """
    local_variances = sliding_variances(values, window_size=3)

    if not local_variances:
        return 0.0

    return std(local_variances)


def autocorrelation_lag1(values):
    """
    Compute lag-1 autocorrelation.

    This measures how similar the signal is to itself one sample later.
    Movement may reduce this temporal similarity.
    """
    n = len(values)

    if n < 2:
        return 0.0

    avg = mean(values)

    numerator = 0.0
    denominator = 0.0

    for index in range(n - 1):
        numerator += (
            (values[index] - avg)
            * (values[index + 1] - avg)
        )

    for value in values:
        difference = value - avg
        denominator += difference * difference

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


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
        interquartile range
        median absolute deviation
        sliding variance mean
        sliding variance standard deviation
        autocorrelation lag 1
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

        features.append(iqr(signal))
        features.append(mad(signal))
        features.append(sliding_variance_mean(signal))
        features.append(sliding_variance_std(signal))
        features.append(autocorrelation_lag1(signal))

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