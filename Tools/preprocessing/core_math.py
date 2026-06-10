import math


"""
Core mathematical utilities used by the CSI processing pipeline.

This module intentionally avoids heavy numerical libraries such as NumPy.
The goal is to keep the implementation simple and compatible with future
embedded deployments, including MicroPython and C-based implementations.

The functions provided here are the mathematical foundation for signal
cleaning, normalization, feature extraction and classification.
"""


def mean(values):
    """
    Compute the arithmetic mean of a list of values.

    The mean is one of the most fundamental statistics in the pipeline and
    serves as the reference value for normalization and variability analysis.

    Returns 0.0 for empty lists to prevent runtime errors.
    """
    if not values:
        return 0.0

    return sum(values) / len(values)


def median(values):
    """
    Compute the median of a list of values.

    Unlike the mean, the median is highly resistant to isolated outliers.
    This property makes it particularly useful for robust signal cleaning
    techniques such as the Hampel filter.
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2

    if n % 2 == 1:
        return sorted_values[mid]

    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def std(values):
    """
    Compute the population standard deviation.

    Standard deviation measures how much the values vary around their mean.

    A low value indicates a stable signal, while a high value indicates
    greater variability. This metric is heavily used during normalization
    and feature extraction.
    """
    if not values:
        return 0.0

    avg = mean(values)
    total = 0.0

    for value in values:
        total += (value - avg) ** 2

    return math.sqrt(total / len(values))


def is_nan_or_inf(value):
    """
    Check whether a numeric value is invalid.

    In signal processing, invalid values such as NaN or infinity can
    break filters, normalization and feature extraction. This function
    centralizes that check so invalid CSI samples can be removed during
    the cleaning stage.
    """
    return math.isnan(value) or math.isinf(value)


def amplitude(real, imag):
    """
    Compute CSI amplitude from real and imaginary components.

    Raw CSI values are complex numbers represented by real and imaginary
    parts. The amplitude corresponds to the magnitude of the complex value
    and is typically more stable and easier to process than the raw
    complex representation.

    Formula:

        |H| = sqrt(real² + imag²)
    """
    return math.sqrt(real * real + imag * imag)


def zscore(value, avg, deviation, eps=1e-8):
    """
    Apply Z-score normalization.

    This transformation converts the signal into a standardized scale with
    zero mean and unit variance.

    Using a common scale reduces the influence of absolute amplitude
    differences and helps the classifier focus on signal behavior rather
    than magnitude.
    
    Formula:

        z = (value - mean) / std
    """
    return (value - avg) / (deviation + eps)


def get_column(matrix, column_index):
    """
    Extract a single subcarrier time series from a CSI matrix.

    Most preprocessing operations are applied independently to each
    subcarrier. This helper allows a subcarrier signal to be manipulated
    as a one-dimensional sequence.
    """
    column = []

    for row in matrix:
        column.append(row[column_index])

    return column


def set_column(matrix, column_index, values):
    """
    Replace a subcarrier time series inside a CSI matrix.

    After a signal has been filtered or normalized, the updated values
    must be written back into the matrix structure.
    """
    for i in range(len(matrix)):
        matrix[i][column_index] = values[i]


def copy_matrix(matrix):
    """
    Create a copy of a CSI matrix.

    Many preprocessing stages should operate on a copy instead of modifying
    the original data directly. This helps preserve intermediate results and
    simplifies debugging.
    """
    copied = []

    for row in matrix:
        copied.append(row[:])

    return copied