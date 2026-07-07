import csv
from collections import Counter

import matplotlib.pyplot as plt

from Tools.common.io_utils import load_json
from Tools.common.project_paths import (
    FEATURE_RANKING_FILE,
    FEATURE_SELECTION_PARAMETERS_FILE,
    PREPROCESSING_PARAMETERS_FILE,
    TABLES_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
)


FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "peak_to_peak",
    "energy",
]


def ensure_output_dirs():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def map_feature_index(feature_index, selected_subcarriers):
    features_per_subcarrier = len(FEATURE_NAMES)

    reduced_subcarrier_index = feature_index // features_per_subcarrier
    feature_type_index = feature_index % features_per_subcarrier

    original_subcarrier = selected_subcarriers[reduced_subcarrier_index]
    feature_type = FEATURE_NAMES[feature_type_index]

    return reduced_subcarrier_index, original_subcarrier, feature_type


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_bar_chart(path, labels, values, title, xlabel, ylabel):
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    ensure_output_dirs()

    ranking = load_json(FEATURE_RANKING_FILE)
    selection_params = load_json(FEATURE_SELECTION_PARAMETERS_FILE)
    preprocessing_params = load_json(PREPROCESSING_PARAMETERS_FILE)

    selected_indices = selection_params["selected_indices"]
    selected_subcarriers = preprocessing_params["selected_subcarriers"]

    ranking_by_index = {
        item["feature_index"]: item
        for item in ranking
    }

    top_features = []
    subcarrier_counter = Counter()
    feature_type_counter = Counter()

    for rank_position, feature_index in enumerate(selected_indices, start=1):
        reduced_sc, original_sc, feature_type = map_feature_index(
            feature_index,
            selected_subcarriers,
        )

        score = ranking_by_index[feature_index]["score"]

        top_features.append({
            "rank": rank_position,
            "feature_index": feature_index,
            "reduced_subcarrier_index": reduced_sc,
            "original_subcarrier": original_sc,
            "feature_type": feature_type,
            "fisher_score": score,
        })

        subcarrier_counter[original_sc] += 1
        feature_type_counter[feature_type] += 1

    subcarrier_frequency = [
        {
            "original_subcarrier": subcarrier,
            "count": count,
        }
        for subcarrier, count in subcarrier_counter.most_common()
    ]

    feature_type_frequency = [
        {
            "feature_type": feature_type,
            "count": count,
        }
        for feature_type, count in feature_type_counter.most_common()
    ]

    save_csv(
        TABLES_DIR / "top_features.csv",
        top_features,
        [
            "rank",
            "feature_index",
            "reduced_subcarrier_index",
            "original_subcarrier",
            "feature_type",
            "fisher_score",
        ],
    )

    save_csv(
        TABLES_DIR / "subcarrier_frequency.csv",
        subcarrier_frequency,
        [
            "original_subcarrier",
            "count",
        ],
    )

    save_csv(
        TABLES_DIR / "feature_type_frequency.csv",
        feature_type_frequency,
        [
            "feature_type",
            "count",
        ],
    )

    save_bar_chart(
        FIGURES_DIR / "subcarrier_frequency.png",
        [str(row["original_subcarrier"]) for row in subcarrier_frequency],
        [row["count"] for row in subcarrier_frequency],
        "Selected Feature Frequency by Subcarrier",
        "Subcarrier",
        "Frequency",
    )

    save_bar_chart(
        FIGURES_DIR / "feature_type_frequency.png",
        [row["feature_type"] for row in feature_type_frequency],
        [row["count"] for row in feature_type_frequency],
        "Selected Feature Type Frequency",
        "Feature Type",
        "Frequency",
    )

    report_path = REPORTS_DIR / "feature_analysis_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("FEATURE ANALYSIS REPORT\n")
        file.write("=" * 70 + "\n\n")

        file.write(f"Selection method: {selection_params['method']}\n")
        file.write(f"Top-K features: {selection_params['top_k_features']}\n")
        file.write(f"Original features: {selection_params['original_num_features']}\n")
        file.write(f"Selected features: {selection_params['selected_num_features']}\n\n")

        file.write("Top selected features:\n")

        for row in top_features:
            file.write(
                f"Rank {row['rank']:02d} | "
                f"Feature {row['feature_index']} | "
                f"SC {row['original_subcarrier']} | "
                f"{row['feature_type']} | "
                f"Score {row['fisher_score']}\n"
            )

        file.write("\nSubcarrier frequency:\n")

        for row in subcarrier_frequency:
            file.write(
                f"SC {row['original_subcarrier']}: {row['count']}\n"
            )

        file.write("\nFeature type frequency:\n")

        for row in feature_type_frequency:
            file.write(
                f"{row['feature_type']}: {row['count']}\n"
            )

    print()
    print("Análise de features concluída.")
    print("Tabelas:", TABLES_DIR)
    print("Figuras:", FIGURES_DIR)
    print("Relatório:", report_path)


if __name__ == "__main__":
    main()