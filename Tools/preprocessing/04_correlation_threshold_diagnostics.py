"""
Correlation Threshold Diagnostics

This script evaluates how many subcarriers are selected for different
Pearson correlation thresholds.

It is used to investigate whether the current correlation threshold is
too aggressive.

The script does not modify the pipeline. It only generates diagnostic
reports, tables and figures.
"""

import csv

import matplotlib.pyplot as plt

from Tools.common.dataset_loader import load_dataset
from Tools.common.project_paths import (
    TABLES_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
)

from Tools.common.config import (
    MIN_INFORMATIVE_STD,
    HAMPEL_WINDOW_SIZE,
    HAMPEL_N_SIGMAS,
    MOVING_AVERAGE_WINDOW_SIZE,
    FEATURES_PER_SUBCARRIER,
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


# ================= CONFIG =================

THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.95,
    0.97,
    0.99,
]

MIN_STD = MIN_INFORMATIVE_STD


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


def get_column(matrix, column_index):
    column = []

    for row in matrix:
        column.append(row[column_index])

    return column


# ================= DATA PREPARATION =================

def load_all_amplitude_rows(dataset):
    """
    Load all binary files and convert them to amplitude rows.
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


def prepare_normalized_matrix(dataset):
    """
    Reproduce the preprocessing calibration sequence:

        amplitude
        ↓
        cleaning
        ↓
        Hampel
        ↓
        moving average
        ↓
        Z-score

    The resulting normalized matrix is then used to test different
    correlation thresholds.
    """
    amplitude_rows, file_errors = load_all_amplitude_rows(dataset)

    clean = remove_invalid_packets(amplitude_rows)

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

    return {
        "amplitude_rows": amplitude_rows,
        "clean": clean,
        "hampel": hampel,
        "smoothed": smoothed,
        "normalized": normalized,
        "means": means,
        "stds": stds,
        "file_errors": file_errors,
    }


def count_informative_subcarriers(matrix):
    """
    Count subcarriers with meaningful standard deviation.
    """
    if not matrix:
        return 0

    count = 0
    num_subcarriers = len(matrix[0])

    for sc in range(num_subcarriers):
        values = get_column(matrix, sc)

        if safe_std(values) > MIN_STD:
            count += 1

    return count


# ================= THRESHOLD ANALYSIS =================

def run_threshold_analysis(normalized):
    """
    Run the correlation-based selection for each threshold.
    """
    results = []

    for threshold in THRESHOLDS:
        try:
            selected_subcarriers = select_non_redundant_subcarriers(
                normalized,
                threshold=threshold,
                min_std=MIN_STD,
            )
        except TypeError:
            selected_subcarriers = select_non_redundant_subcarriers(
                normalized,
                threshold=threshold,
            )

        selected_count = len(selected_subcarriers)
        feature_count = selected_count * FEATURES_PER_SUBCARRIER

        results.append(
            {
                "threshold": threshold,
                "selected_count": selected_count,
                "feature_count": feature_count,
                "selected_subcarriers": selected_subcarriers,
            }
        )

    return results


# ================= OUTPUTS =================

def save_threshold_summary(results):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = TABLES_DIR / "correlation_threshold_summary.csv"

    fieldnames = [
        "threshold",
        "selected_count",
        "feature_count",
        "selected_subcarriers",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow(
                {
                    "threshold": row["threshold"],
                    "selected_count": row["selected_count"],
                    "feature_count": row["feature_count"],
                    "selected_subcarriers": ";".join(
                        str(sc)
                        for sc in row["selected_subcarriers"]
                    ),
                }
            )

    return output_path


def save_selected_subcarriers_table(results):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = TABLES_DIR / "selected_subcarriers_by_threshold.csv"

    fieldnames = [
        "threshold",
        "subcarrier",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            threshold = row["threshold"]

            for subcarrier in row["selected_subcarriers"]:
                writer.writerow(
                    {
                        "threshold": threshold,
                        "subcarrier": subcarrier,
                    }
                )

    return output_path


def save_threshold_plot(results):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / "correlation_threshold_selected_count.png"

    thresholds = []
    selected_counts = []

    for row in results:
        thresholds.append(row["threshold"])
        selected_counts.append(row["selected_count"])

    plt.figure(figsize=(8, 5))

    plt.plot(
        thresholds,
        selected_counts,
        marker="o",
        linewidth=2,
    )

    plt.title("Selected Subcarriers by Correlation Threshold")
    plt.xlabel("Pearson Correlation Threshold")
    plt.ylabel("Selected Subcarriers")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_report(
    dataset,
    preprocessing_result,
    informative_count,
    results,
):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / "correlation_threshold_diagnostics_report.txt"

    normalized = preprocessing_result["normalized"]
    file_errors = preprocessing_result["file_errors"]

    lines = []

    lines.append("CORRELATION THRESHOLD DIAGNOSTICS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Files found: {len(dataset)}")
    lines.append(
        f"Amplitude rows before cleaning: "
        f"{len(preprocessing_result['amplitude_rows'])}"
    )
    lines.append(f"Valid rows after cleaning: {len(preprocessing_result['clean'])}")
    lines.append(f"Normalized rows: {len(normalized)}")

    if normalized:
        lines.append(f"Original subcarriers: {len(normalized[0])}")
    else:
        lines.append("Original subcarriers: 0")

    lines.append(f"Informative subcarriers before correlation: {informative_count}")
    lines.append(f"File loading errors: {len(file_errors)}")
    lines.append("")
    lines.append("Configuration")
    lines.append("-" * 70)
    lines.append(f"Minimum informative std: {MIN_STD}")
    lines.append(f"Hampel window size: {HAMPEL_WINDOW_SIZE}")
    lines.append(f"Hampel n sigmas: {HAMPEL_N_SIGMAS}")
    lines.append(f"Moving average window size: {MOVING_AVERAGE_WINDOW_SIZE}")
    lines.append(f"Features per subcarrier: {FEATURES_PER_SUBCARRIER}")
    lines.append("")
    lines.append("Threshold results")
    lines.append("-" * 70)

    previous_count = None

    for row in results:
        threshold = row["threshold"]
        selected_count = row["selected_count"]
        feature_count = row["feature_count"]

        if previous_count is None:
            delta = 0
        else:
            delta = selected_count - previous_count

        previous_count = selected_count

        lines.append(
            f"Threshold {threshold:.2f} | "
            f"Selected subcarriers: {selected_count:3d} | "
            f"Features: {feature_count:4d} | "
            f"Delta: {delta:+3d}"
        )

    lines.append("")
    lines.append("Selected subcarriers by threshold")
    lines.append("-" * 70)

    for row in results:
        lines.append("")
        lines.append(f"Threshold {row['threshold']:.2f}")
        lines.append(str(row["selected_subcarriers"]))

    if file_errors:
        lines.append("")
        lines.append("File loading errors")
        lines.append("-" * 70)

        for item in file_errors:
            lines.append(f"{item['path']} | {item['error']}")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    return output_path


# ================= MAIN =================

def main():
    dataset = load_dataset()

    print()
    print("Correlation Threshold Diagnostics")
    print("=" * 70)
    print("Files found:", len(dataset))

    preprocessing_result = prepare_normalized_matrix(dataset)

    normalized = preprocessing_result["normalized"]

    print("Amplitude rows before cleaning:", len(preprocessing_result["amplitude_rows"]))
    print("Valid rows after cleaning:", len(preprocessing_result["clean"]))
    print("Normalized rows:", len(normalized))

    if normalized:
        print("Original subcarriers:", len(normalized[0]))

    informative_count = count_informative_subcarriers(normalized)

    print("Informative subcarriers before correlation:", informative_count)

    results = run_threshold_analysis(normalized)

    print()
    print("Threshold results")
    print("-" * 70)

    for row in results:
        print(
            "Threshold:",
            row["threshold"],
            "| Selected subcarriers:",
            row["selected_count"],
            "| Features:",
            row["feature_count"],
        )

    summary_path = save_threshold_summary(results)
    subcarriers_path = save_selected_subcarriers_table(results)
    figure_path = save_threshold_plot(results)
    report_path = save_report(
        dataset=dataset,
        preprocessing_result=preprocessing_result,
        informative_count=informative_count,
        results=results,
    )

    print()
    print("Summary saved:")
    print(summary_path)

    print()
    print("Selected subcarriers table saved:")
    print(subcarriers_path)

    print()
    print("Figure saved:")
    print(figure_path)

    print()
    print("Report saved:")
    print(report_path)


if __name__ == "__main__":
    main()