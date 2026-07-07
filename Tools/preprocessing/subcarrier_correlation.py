"""
Subcarrier Correlation Analysis

This module performs redundancy reduction based on the correlation
between CSI subcarriers.

Neighboring subcarriers often exhibit very similar behavior because
they experience nearly the same wireless channel conditions. Keeping
all of them increases computational cost while providing little
additional information.

The goal of this stage is to identify highly correlated subcarriers
and keep only a representative subset.

Pipeline position:

    normalized CSI
    ↓
    correlation analysis
    ↓
    redundant subcarrier removal
    ↓
    sliding windows
    ↓
    feature extraction

This reduction decreases memory usage, processing time and feature
dimensionality before classification.

The correlation threshold is intentionally configurable because
different datasets may require different levels of redundancy
reduction.

Pearson correlation was chosen because the objective is to identify
subcarriers that exhibit similar temporal behavior.

Highly correlated subcarriers tend to carry redundant information,
making them good candidates for dimensionality reduction.

Compared to rank-based methods such as Spearman correlation,
Pearson is computationally simpler and more suitable for future
embedded implementations.
"""

from Tools.preprocessing.core_math import mean, std, get_column


def covariance(values_a, values_b):
    """
    Compute the covariance between two signals.

    Covariance measures how two signals vary together.

    Formula:

        cov(X,Y) = Σ[(Xi - μx)(Yi - μy)] / N

    where:

        μx = mean of signal X
        μy = mean of signal Y

    Positive values indicate that both signals tend to increase and
    decrease together.

    Negative values indicate opposite behavior.

    Covariance is used as an intermediate step for Pearson correlation.
    """
    if not values_a or not values_b:
        return 0.0

    n = min(len(values_a), len(values_b))

    if n == 0:
        return 0.0

    avg_a = mean(values_a[:n])
    avg_b = mean(values_b[:n])

    total = 0.0

    for i in range(n):
        total += (values_a[i] - avg_a) * (values_b[i] - avg_b)

    return total / n


def pearson_correlation(values_a, values_b, eps=1e-8):
    """
    Compute the Pearson correlation coefficient between two signals.

    Formula:

                    cov(X,Y)
        r = -------------------------
              σx × σy

    where:

        cov(X,Y) = covariance
        σx = standard deviation of X
        σy = standard deviation of Y

    Output range:

        r =  1.0  -> perfectly correlated
        r =  0.0  -> no linear correlation
        r = -1.0  -> perfectly inverse correlation

    In this project the absolute correlation value is used because
    both highly positive and highly negative correlations indicate
    redundant information.
    """
    n = min(len(values_a), len(values_b))

    if n == 0:
        return 0.0

    values_a = values_a[:n]
    values_b = values_b[:n]

    cov = covariance(values_a, values_b)
    std_a = std(values_a)
    std_b = std(values_b)

    return cov / ((std_a * std_b) + eps)


def is_informative_signal(values, min_std=1e-6):
    """
    Check whether a subcarrier signal carries useful variation.

    Subcarriers with near-zero standard deviation are considered
    non-informative because they remain almost constant across time.

    These subcarriers should not enter the correlation-based selection,
    otherwise they may be incorrectly kept as non-redundant.
    """
    if not values:
        return False

    return std(values) > min_std


def is_informative_signal(values, min_std=1e-6):
    """
    Check whether a subcarrier signal carries useful variation.

    Subcarriers with near-zero standard deviation are considered
    non-informative because they remain almost constant across time.

    These subcarriers should not enter the correlation-based selection,
    otherwise they may be incorrectly kept as non-redundant.
    """
    if not values:
        return False

    return std(values) > min_std



def filter_matrix_by_subcarriers(matrix, selected_subcarriers):
    """
    Create a reduced CSI matrix containing only the selected
    subcarriers.

    Input:

        matrix[packet][subcarrier]

    Output:

        filtered_matrix[packet][selected_subcarrier]

    This operation applies the redundancy reduction results to the
    dataset before window generation and feature extraction.
    """

    filtered = []

    for row in matrix:
        new_row = []

        for sc in selected_subcarriers:
            if sc < len(row):
                new_row.append(row[sc])

        filtered.append(new_row)

    return filtered


def print_selected_subcarriers(selected_subcarriers):
    """
    Display the subcarriers that survived the redundancy removal stage.

    This function is intended for diagnostics and experimentation,
    helping evaluate the impact of different correlation thresholds
    on the number of retained subcarriers.
    """
    print()
    print("Subportadoras após remoção de redundância:")
    print(selected_subcarriers)
    print("Total:", len(selected_subcarriers))
    

# ================= INFORMATIVE SUBCARRIER FILTER =================

def _local_mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def _local_std(values):
    if not values:
        return 0.0

    avg = _local_mean(values)

    total = 0.0

    for value in values:
        total += (value - avg) ** 2

    return (total / len(values)) ** 0.5


def _local_get_column(matrix, column_index):
    column = []

    for row in matrix:
        column.append(row[column_index])

    return column


def pearson_correlation(signal_a, signal_b, eps=1e-8):
    """
    Compute Pearson correlation between two signals.
    """
    if not signal_a or not signal_b:
        return 0.0

    if len(signal_a) != len(signal_b):
        return 0.0

    mean_a = _local_mean(signal_a)
    mean_b = _local_mean(signal_b)

    numerator = 0.0
    denominator_a = 0.0
    denominator_b = 0.0

    for i in range(len(signal_a)):
        diff_a = signal_a[i] - mean_a
        diff_b = signal_b[i] - mean_b

        numerator += diff_a * diff_b
        denominator_a += diff_a ** 2
        denominator_b += diff_b ** 2

    denominator = (denominator_a ** 0.5) * (denominator_b ** 0.5)

    if denominator < eps:
        return 0.0

    return numerator / denominator


def is_informative_signal(values, min_std=1e-6):
    """
    Check whether a subcarrier signal carries useful variation.

    Subcarriers with near-zero standard deviation are considered
    non-informative because they remain almost constant across time.
    """
    if not values:
        return False

    return _local_std(values) > min_std


def select_non_redundant_subcarriers(
    matrix,
    threshold=0.95,
    min_std=1e-6,
):
    """
    Select non-redundant and informative subcarriers.

    A subcarrier is kept only if:

    1. It has meaningful temporal variation.
    2. It is not highly correlated with a previously selected subcarrier.
    """
    if not matrix:
        return []

    num_subcarriers = len(matrix[0])

    selected_subcarriers = []

    for candidate_sc in range(num_subcarriers):
        candidate_signal = _local_get_column(matrix, candidate_sc)

        if not is_informative_signal(
            candidate_signal,
            min_std=min_std,
        ):
            continue

        is_redundant = False

        for selected_sc in selected_subcarriers:
            selected_signal = _local_get_column(matrix, selected_sc)

            correlation = pearson_correlation(
                candidate_signal,
                selected_signal,
            )

            if abs(correlation) >= threshold:
                is_redundant = True
                break

        if not is_redundant:
            selected_subcarriers.append(candidate_sc)

    return selected_subcarriers
