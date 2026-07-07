import json
import pickle

from Tools.common.dataset_loader import load_dataset, print_dataset_summary
from Tools.common.project_paths import (
    FEATURE_DATASET_FILE,
    PREPROCESSING_PARAMETERS_FILE,
)

from Tools.preprocessing.csi_pipeline_core import (
    fit_preprocessing_pipeline,
    transform_bin_file,
)

from Tools.preprocessing.subcarrier_correlation import (
    select_non_redundant_subcarriers,
    filter_matrix_by_subcarriers,
    print_selected_subcarriers,
)

from Tools.preprocessing.sliding_window import create_sliding_windows
from Tools.preprocessing.feature_extraction import extract_features_from_window


WINDOW_SIZE = 5
STEP_SIZE = 2
CORRELATION_THRESHOLD = 0.40


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def save_pickle(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump(data, file)


def main():
    dataset = load_dataset()

    print_dataset_summary(dataset)

    file_paths = [item["path"] for item in dataset]

    print()
    print("Ajustando parâmetros de pré-processamento...")

    fit_result = fit_preprocessing_pipeline(file_paths)

    means = fit_result["means"]
    stds = fit_result["stds"]
    normalized_all = fit_result["normalized"]

    selected_subcarriers = select_non_redundant_subcarriers(
        normalized_all,
        threshold=CORRELATION_THRESHOLD,
    )

    print_selected_subcarriers(selected_subcarriers)

    feature_dataset = []

    print()
    print("Gerando features...")

    for item in dataset:
        normalized = transform_bin_file(
            item["path"],
            means,
            stds,
        )

        reduced = filter_matrix_by_subcarriers(
            normalized,
            selected_subcarriers,
        )

        windows = create_sliding_windows(
            reduced,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE,
        )

        for window_index, window in enumerate(windows):
            features = extract_features_from_window(window)

            feature_dataset.append({
                "label": item["label"],
                "quadrant": item["quadrant"],
                "file_name": item["file_name"],
                "window_index": window_index,
                "features": features,
            })

    preprocessing_parameters = {
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "means": means,
        "stds": stds,
        "selected_subcarriers": selected_subcarriers,
        "num_selected_subcarriers": len(selected_subcarriers),
        "features_per_sample": (
            len(feature_dataset[0]["features"])
            if feature_dataset
            else 0
        ),
        "num_samples": len(feature_dataset),
    }

    save_pickle(FEATURE_DATASET_FILE, feature_dataset)
    save_json(PREPROCESSING_PARAMETERS_FILE, preprocessing_parameters)

    print()
    print("Features salvas em:")
    print(FEATURE_DATASET_FILE)

    print()
    print("Parâmetros de pré-processamento salvos em:")
    print(PREPROCESSING_PARAMETERS_FILE)

    print()
    print("Resumo final:")
    print("Amostras:", len(feature_dataset))

    if feature_dataset:
        print("Features por amostra:", len(feature_dataset[0]["features"]))


if __name__ == "__main__":
    main()