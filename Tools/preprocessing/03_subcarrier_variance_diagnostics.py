"""
Subcarrier Variance Diagnostics

This script diagnoses how much useful variation each subcarrier carries
at different preprocessing stages.

It is used to investigate whether subcarriers become constant during:

    amplitude extraction
    packet cleaning
    Hampel filtering
    moving average smoothing
    Z-score normalization
    correlation-based subcarrier selection

The script does not modify the pipeline. It only generates reports,
tables and figures for analysis.
"""

import csv
from collections import Counter

import matplotlib.pyplot as plt

from Tools.common.dataset_loader import load_dataset
from Tools.common.project_paths import (
    TABLES_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
)

from Tools.preprocessing.csi_pipeline_core import (
    load_bin_file,
    packets_to_amplitude_matrix,
    remove_invalid_packets,
    hampel_filter_matrix,
    moving_average_matrix,
    fit_zscore_parameters,
    apply_zscore_parameters,
)

from Tools.preprocessing.subcarrier_correlation import (
    select_non_redundant_subcarriers,
)


CORRELATION_THRESHOLD = 0.40
MIN_STD = 1e-6

HAMPEL_WINDOW_SIZE = 5
HAMPEL_N_SIGMAS = 3.0
MOVING_AVERAGE_WINDOW_SIZE = 3

STD_THRESHOLDS = [
    0.0,
    1e-12,
    1e-9,
    1e-6,
    1e-4,
    1e-3,
    1e-2,
]


# ================= BASIC MATH =================

def safe_mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def safe_variance(values):
    if not values:
        return 0.0

    avg = safe_mean(values)

    total = 0.0

    for value in values:
        total += (value - avg) ** 2

    return total / len(values)


def safe_std(values):
    return safe_variance(values) ** 0.5


def safe_min(values):
    if not values:
        return 0.0

    return min(values)


def safe_max(values):
    if not values:
        return 0.0

    return max(values)


def get_column(matrix, column_index):
    column = []

    for row in matrix:
        column.append(row[column_index])

    return column


def median(values):
    if not values:
        return 0.0

    ordered = sorted(values)
    n = len(ordered)
    middle = n // 2

    if n % 2 == 1:
        return ordered[middle]

    return (ordered[middle - 1] + ordered[middle]) / 2


# ================= DATA LOADING =================

def load_all_amplitude_rows(dataset):
    """
    Load all binary files and convert them to amplitude rows.

    Output:
        amplitude_rows:
            list of amplitude rows from all files

        file_errors:
            files that could not be loaded
    """
    amplitude_rows = []
    file_errors = []

    for item in dataset:
        file_path = item["path"]

        try:
            packets = load_bin_file(file_path)
            matrix = packets_to_amplitude_matrix(packets)

            for row in matrix:
                amplitude_rows.append(row)

        except Exception as error:
            file_errors.append(
                {
                    "path": file_path,
                    "error": str(error),
                }
            )

    return amplitude_rows, file_errors


def row_length_distribution(matrix):
    counter = Counter()

    for row in matrix:
        counter[len(row)] += 1

    return counter


# ================= STAGE STATISTICS =================

def compute_subcarrier_statistics(matrix, stage_name):
    """
    Compute per-subcarrier statistics for one preprocessing stage.
    """
    statistics = []

    if not matrix:
        return statistics

    num_subcarriers = len(matrix[0])

    for sc in range(num_subcarriers):
        values = get_column(matrix, sc)

        sc_mean = safe_mean(values)
        sc_variance = safe_variance(values)
        sc_std = sc_variance ** 0.5

        statistics.append(
            {
                "stage": stage_name,
                "subcarrier": sc,
                "mean": sc_mean,
                "variance": sc_variance,
                "std": sc_std,
                "min": safe_min(values),
                "max": safe_max(values),
                "informative": sc_std > MIN_STD,
            }
        )

    return statistics


def summarize_stage(matrix, stage_name):
    """
    Summarize the variance behavior of one preprocessing stage.
    """
    if not matrix:
        return {
            "stage": stage_name,
            "rows": 0,
            "subcarriers": 0,
            "zero_std": 0,
            "informative": 0,
            "min_std": 0.0,
            "median_std": 0.0,
            "max_std": 0.0,
        }

    stats = compute_subcarrier_statistics(matrix, stage_name)

    std_values = []

    for item in stats:
        std_values.append(item["std"])

    zero_std = 0
    informative = 0

    for value in std_values:
        if value <= MIN_STD:
            zero_std += 1
        else:
            informative += 1

    return {
        "stage": stage_name,
        "rows": len(matrix),
        "subcarriers": len(matrix[0]),
        "zero_std": zero_std,
        "informative": informative,
        "min_std": safe_min(std_values),
        "median_std": median(std_values),
        "max_std": safe_max(std_values),
    }


def count_by_threshold(stats):
    """
    Count how many subcarriers have standard deviation above each threshold.
    """
    result = {}

    for threshold in STD_THRESHOLDS:
        count = 0

        for item in stats:
            if item["std"] > threshold:
                count += 1

        result[threshold] = count

    return result


# ================= OUTPUTS =================

def save_stage_summary(summary_rows):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = TABLES_DIR / "stage_variance_summary.csv"

    fieldnames = [
        "stage",
        "rows",
        "subcarriers",
        "zero_std",
        "informative",
        "min_std",
        "median_std",
        "max_std",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)

    return output_path


def save_subcarrier_statistics(all_statistics):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = TABLES_DIR / "subcarrier_variance_by_stage.csv"

    fieldnames = [
        "stage",
        "subcarrier",
        "mean",
        "variance",
        "std",
        "min",
        "max",
        "informative",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in all_statistics:
            writer.writerow(row)

    return output_path


def save_std_plot(stage_statistics):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / "subcarrier_std_by_stage.png"

    plt.figure(figsize=(10, 5))

    for stage_name, stats in stage_statistics.items():
        subcarriers = []
        std_values = []

        for item in stats:
            subcarriers.append(item["subcarrier"])
            std_values.append(item["std"])

        plt.plot(
            subcarriers,
            std_values,
            marker="o",
            linewidth=1,
            markersize=2,
            label=stage_name,
        )

    plt.title("Subcarrier Standard Deviation by Preprocessing Stage")
    plt.xlabel("Subcarrier")
    plt.ylabel("Standard Deviation")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_report(
    dataset,
    amplitude_rows,
    clean,
    hampel,
    smoothed,
    normalized,
    selected_subcarriers,
    file_errors,
    stage_summaries,
    stage_statistics,
):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / "subcarrier_variance_diagnostics_report.txt"

    lines = []

    lines.append("SUBCARRIER VARIANCE DIAGNOSTICS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Files found: {len(dataset)}")
    lines.append(f"Amplitude rows before cleaning: {len(amplitude_rows)}")
    lines.append(f"Valid rows after cleaning: {len(clean)}")
    lines.append(f"File loading errors: {len(file_errors)}")
    lines.append("")
    lines.append("Configuration")
    lines.append("-" * 70)
    lines.append(f"Correlation threshold: {CORRELATION_THRESHOLD}")
    lines.append(f"Minimum informative std: {MIN_STD}")
    lines.append(f"Hampel window size: {HAMPEL_WINDOW_SIZE}")
    lines.append(f"Hampel n sigmas: {HAMPEL_N_SIGMAS}")
    lines.append(f"Moving average window size: {MOVING_AVERAGE_WINDOW_SIZE}")
    lines.append("")

    lines.append("Row length distribution before cleaning")
    lines.append("-" * 70)

    length_counter = row_length_distribution(amplitude_rows)

    for length, count in sorted(length_counter.items()):
        lines.append(f"Length {length}: {count}")

    lines.append("")
    lines.append("Stage summary")
    lines.append("-" * 70)

    for summary in stage_summaries:
        lines.append(
            f"{summary['stage']}: "
            f"rows={summary['rows']} | "
            f"subcarriers={summary['subcarriers']} | "
            f"zero_std={summary['zero_std']} | "
            f"informative={summary['informative']} | "
            f"min_std={summary['min_std']} | "
            f"median_std={summary['median_std']} | "
            f"max_std={summary['max_std']}"
        )

    lines.append("")
    lines.append("Standard deviation threshold analysis")
    lines.append("-" * 70)

    for stage_name, stats in stage_statistics.items():
        lines.append("")
        lines.append(stage_name)

        threshold_counts = count_by_threshold(stats)

        for threshold, count in threshold_counts.items():
            lines.append(f"std > {threshold:<8}: {count}")

    lines.append("")
    lines.append("Selected subcarriers after correlation")
    lines.append("-" * 70)
    lines.append(f"Total: {len(selected_subcarriers)}")
    lines.append(str(selected_subcarriers))
    lines.append("")

    if file_errors:
        lines.append("File loading errors")
        lines.append("-" * 70)

        for item in file_errors:
            lines.append(f"{item['path']} | {item['error']}")

        lines.append("")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    return output_path


# ================= MAIN =================

def main():
    dataset = load_dataset()

    print()
    print("Subcarrier Variance Diagnostics")
    print("=" * 70)
    print("Files found:", len(dataset))

    amplitude_rows, file_errors = load_all_amplitude_rows(dataset)

    print("Amplitude rows before cleaning:", len(amplitude_rows))
    print("File loading errors:", len(file_errors))

    clean = remove_invalid_packets(amplitude_rows)

    print("Valid rows after cleaning:", len(clean))

    hampel = hampel_filter_matrix(
        clean,
        window_size=HAMPEL_WINDOW_SIZE,
        n_sigmas=HAMPEL_N_SIGMAS,
    )

    smoothed = moving_average_matrix(
        hampel,
        window_size=MOVING_AVERAGE_WINDOW_SIZE,
    )

    means, stds = fit_zscore_parameters(smoothed)

    normalized = apply_zscore_parameters(
        smoothed,
        means,
        stds,
    )

    try:
        selected_subcarriers = select_non_redundant_subcarriers(
            normalized,
            threshold=CORRELATION_THRESHOLD,
            min_std=MIN_STD,
        )
    except TypeError:
        selected_subcarriers = select_non_redundant_subcarriers(
            normalized,
            threshold=CORRELATION_THRESHOLD,
        )

    stages = {
        "clean_amplitude": clean,
        "hampel": hampel,
        "moving_average": smoothed,
        "zscore": normalized,
    }

    stage_summaries = []
    stage_statistics = {}
    all_statistics = []

    for stage_name, matrix in stages.items():
        summary = summarize_stage(matrix, stage_name)
        statistics = compute_subcarrier_statistics(matrix, stage_name)

        stage_summaries.append(summary)
        stage_statistics[stage_name] = statistics

        for item in statistics:
            all_statistics.append(item)

    summary_path = save_stage_summary(stage_summaries)
    stats_path = save_subcarrier_statistics(all_statistics)
    plot_path = save_std_plot(stage_statistics)

    report_path = save_report(
        dataset=dataset,
        amplitude_rows=amplitude_rows,
        clean=clean,
        hampel=hampel,
        smoothed=smoothed,
        normalized=normalized,
        selected_subcarriers=selected_subcarriers,
        file_errors=file_errors,
        stage_summaries=stage_summaries,
        stage_statistics=stage_statistics,
    )

    print()
    print("Stage summary saved:")
    print(summary_path)

    print()
    print("Subcarrier statistics saved:")
    print(stats_path)

    print()
    print("Figure saved:")
    print(plot_path)

    print()
    print("Report saved:")
    print(report_path)

    print()
    print("Selected subcarriers after correlation:")
    print(selected_subcarriers)


if __name__ == "__main__":
    main()