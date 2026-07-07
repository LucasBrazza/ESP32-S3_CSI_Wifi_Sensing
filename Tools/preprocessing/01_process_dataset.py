from Tools.common.dataset_loader import load_dataset, print_dataset_summary

from Tools.preprocessing.csi_pipeline_core import (
    build_labeled_window_dataset,
    extract_feature_dataset,
    print_class_count,
)


from Tools.common.config import (
    WINDOW_SIZE,
    STEP_SIZE,
    CORRELATION_THRESHOLD,
)


def build_file_label_pairs(dataset):
    file_label_pairs = []

    for item in dataset:
        file_label_pairs.append(
            (
                item["path"],
                item["label"],
            )
        )

    return file_label_pairs


if __name__ == "__main__":
    dataset = load_dataset()

    print_dataset_summary(dataset)

    file_label_pairs = build_file_label_pairs(dataset)

    print()
    print("Arquivos enviados ao pipeline:", len(file_label_pairs))

    result = build_labeled_window_dataset(
        file_label_pairs,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        correlation_threshold=CORRELATION_THRESHOLD,
    )

    print()
    print("Pré-processamento concluído.")
    print("Total de janelas:", result["num_windows"])
    print("Window size:", result["window_size"])
    print("Step size:", result["step_size"])
    print("Correlation threshold:", CORRELATION_THRESHOLD)
    print("Subportadoras selecionadas:", len(result["selected_subcarriers"]))

    print_class_count(result["dataset"])

    feature_dataset = extract_feature_dataset(
        result["dataset"]
    )

    print()
    print("Dataset de features criado.")
    print("Amostras:", len(feature_dataset))

    if feature_dataset:
        print(
            "Features por amostra:",
            len(feature_dataset[0]["features"]),
        )
        print(
            "Primeira classe:",
            feature_dataset[0]["label"],
        )