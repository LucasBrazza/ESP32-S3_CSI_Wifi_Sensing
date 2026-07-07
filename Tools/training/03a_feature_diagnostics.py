from collections import Counter

import matplotlib.pyplot as plt

from Tools.common.io_utils import load_json, load_pickle

from Tools.common.project_paths import (
    FEATURE_DATASET_FILE,
    FEATURE_RANKING_FILE,
    FIGURES_DIR,
    REPORTS_DIR,
)


FISHER_THRESHOLDS = [
    0,
    0.0001,
    0.001,
    0.005,
    0.01,
    0.05,
]


def transpose_features(dataset):
    """
    Converts

    sample -> feature vector

    into

    feature -> list of values
    """

    num_features = len(dataset[0]["features"])

    feature_matrix = []

    for feature_index in range(num_features):

        values = []

        for sample in dataset:
            values.append(sample["features"][feature_index])

        feature_matrix.append(values)

    return feature_matrix


def variance(values):

    if not values:
        return 0.0

    avg = sum(values) / len(values)

    total = 0.0

    for value in values:
        total += (value - avg) ** 2

    return total / len(values)


def save_histogram(scores):

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.hist(scores, bins=30)

    plt.title("Fisher Score Distribution")

    plt.xlabel("Fisher Score")

    plt.ylabel("Number of Features")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "fisher_score_histogram.png",
        dpi=300,
    )

    plt.close()


def main():

    feature_dataset = load_pickle(FEATURE_DATASET_FILE)

    ranking = load_json(FEATURE_RANKING_FILE)

    print()
    print("Feature Diagnostics")
    print("=" * 60)

    print("Samples :", len(feature_dataset))
    print("Features:", len(feature_dataset[0]["features"]))

    print()

    feature_matrix = transpose_features(feature_dataset)

    zero_variance = 0

    variances = []

    for values in feature_matrix:

        v = variance(values)

        variances.append(v)

        if abs(v) < 1e-12:
            zero_variance += 1

    print("Features with zero variance :", zero_variance)

    print()

    scores = []

    for item in ranking:
        scores.append(item["score"])

    save_histogram(scores)

    print("Fisher statistics")
    print("-" * 60)

    print("Maximum :", max(scores))
    print("Minimum :", min(scores))
    print("Average :", sum(scores) / len(scores))

    scores_sorted = sorted(scores)

    median = scores_sorted[len(scores_sorted) // 2]

    print("Median  :", median)

    print()

    print("Feature count by threshold")

    report_lines = []

    report_lines.append("FEATURE DIAGNOSTICS")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append(f"Samples : {len(feature_dataset)}")
    report_lines.append(f"Features: {len(feature_dataset[0]['features'])}")
    report_lines.append("")
    report_lines.append(f"Zero variance features: {zero_variance}")
    report_lines.append("")
    report_lines.append("Threshold analysis")

    for threshold in FISHER_THRESHOLDS:

        count = sum(
            1
            for score in scores
            if score > threshold
        )

        print(f"> {threshold:<8} : {count}")

        report_lines.append(
            f"Score > {threshold:<8} : {count}"
        )

    report_lines.append("")
    report_lines.append("Top 20 Fisher Scores")
    report_lines.append("")

    for i, item in enumerate(ranking[:20], start=1):

        report_lines.append(
            f"{i:02d} | Feature {item['feature_index']:3d} | Score {item['score']:.8f}"
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        REPORTS_DIR / "feature_diagnostics_report.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write("\n".join(report_lines))

    print()
    print("Histogram saved:")
    print(FIGURES_DIR / "fisher_score_histogram.png")

    print()
    print("Report saved:")
    print(REPORTS_DIR / "feature_diagnostics_report.txt")


if __name__ == "__main__":
    main()