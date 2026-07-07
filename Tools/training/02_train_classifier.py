import json
import pickle

from Tools.common.config import (
    MODEL,
    MAX_TREE_DEPTH,
    MIN_SAMPLES_SPLIT,
)

from Tools.common.project_paths import (
    SELECTED_FEATURE_DATASET_FILE,
    CLASSIFIER_FILE,
    CLASSIFIER_PARAMETERS_FILE,
)

from Tools.classification.decision_tree import build_tree


MAX_DEPTH = MAX_TREE_DEPTH


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump(data, file)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def count_by_label(dataset):
    counts = {}

    for item in dataset:
        label = item["label"]
        counts[label] = counts.get(label, 0) + 1

    return counts


def main():
    dataset = load_pickle(SELECTED_FEATURE_DATASET_FILE)

    if not dataset:
        raise ValueError("Dataset selecionado vazio.")

    print("Treinando classificador:", MODEL)
    print("Amostras:", len(dataset))
    print("Features por amostra:", len(dataset[0]["features"]))

    print()
    print("Amostras por classe:")

    for label, count in sorted(count_by_label(dataset).items()):
        print(label, ":", count)

    if MODEL != "decision_tree":
        raise ValueError("Modelo ainda não implementado.")

    classifier = build_tree(
        dataset,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
    )

    parameters = {
        "model": MODEL,
        "max_depth": MAX_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "num_samples": len(dataset),
        "num_features": len(dataset[0]["features"]),
        "class_count": count_by_label(dataset),
    }

    save_pickle(CLASSIFIER_FILE, classifier)
    save_json(CLASSIFIER_PARAMETERS_FILE, parameters)

    print()
    print("Classificador treinado.")
    print("Modelo salvo em:")
    print(CLASSIFIER_FILE)

    print()
    print("Parâmetros salvos em:")
    print(CLASSIFIER_PARAMETERS_FILE)


if __name__ == "__main__":
    main()