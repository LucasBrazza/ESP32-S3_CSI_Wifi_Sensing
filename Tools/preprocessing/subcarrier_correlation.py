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


def select_non_redundant_subcarriers(matrix, threshold=0.95):
    """
    Remove redundant subcarriers using Pearson correlation.

    The algorithm evaluates each candidate subcarrier against the
    already selected subcarriers.

    Selection rule:

        if |correlation| >= threshold

            candidate is considered redundant

        else

            candidate is kept

    Example:

        threshold = 0.90

        corr = 0.95 -> remove
        corr = 0.82 -> keep

    The result is a smaller set of subcarriers that preserves most
    of the useful information while reducing computational cost.

    This stage is particularly important for embedded deployment,
    since fewer subcarriers lead to fewer features and less memory
    consumption.
    """

    if not matrix:
        return []

    num_subcarriers = len(matrix[0])

    selected_subcarriers = []

    for candidate_sc in range(num_subcarriers):
        candidate_signal = get_column(matrix, candidate_sc)

        is_redundant = False

        for selected_sc in selected_subcarriers:
            selected_signal = get_column(matrix, selected_sc)

            corr = pearson_correlation(
                candidate_signal,
                selected_signal,
            )

            if abs(corr) >= threshold:
                is_redundant = True
                break

        if not is_redundant:
            selected_subcarriers.append(candidate_sc)

    return selected_subcarriers


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
