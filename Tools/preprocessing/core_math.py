import math


def mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def median(values):
    if not values:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2

    if n % 2 == 1:
        return sorted_values[mid]

    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def std(values):
    if not values:
        return 0.0

    avg = mean(values)
    total = 0.0

    for value in values:
        total += (value - avg) ** 2

    return math.sqrt(total / len(values))


def is_nan_or_inf(value):
    return math.isnan(value) or math.isinf(value)


def amplitude(real, imag):
    return math.sqrt(real * real + imag * imag)


def zscore(value, avg, deviation, eps=1e-8):
    return (value - avg) / (deviation + eps)


def get_column(matrix, column_index):
    column = []

    for row in matrix:
        column.append(row[column_index])

    return column


def set_column(matrix, column_index, values):
    for i in range(len(matrix)):
        matrix[i][column_index] = values[i]


def copy_matrix(matrix):
    copied = []

    for row in matrix:
        copied.append(row[:])

    return copied