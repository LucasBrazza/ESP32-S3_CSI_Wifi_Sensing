import json
import pickle

from Tools.common.config import (
    TOP_K_FEATURES,
)

from Tools.common.project_paths import (
    FEATURE_DATASET_FILE,
    SELECTED_FEATURE_DATASET_FILE,
    FEATURE_RANKING_FILE,
    FEATURE_SELECTION_PARAMETERS_FILE,
)

from Tools.preprocessing.feature_selection import (
    rank_features_by_fisher_score,
    select_top_features,
)


METHOD = "fisher"


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
    print("Carregando dataset de features...")
    feature_dataset = load_pickle(FEATURE_DATASET_FILE)

    if not feature_dataset:
        raise ValueError("Dataset de features vazio.")

    print("Amostras:", len(feature_dataset))
    print("Features por amostra:", len(feature_dataset[0]["features"]))

    print()
    print("Amostras por classe:")

    for label, count in sorted(count_by_label(feature_dataset).items()):
        print(label, ":", count)

    print()
    print("Selecionando features usando:", METHOD)

    if METHOD != "fisher":
        raise ValueError("Método de seleção ainda não implementado.")

    ranking = rank_features_by_fisher_score(feature_dataset)

    selected_dataset, selected_indices = select_top_features(
        feature_dataset,
        ranking,
        top_k=TOP_K_FEATURES,
    )

    parameters = {
        "method": METHOD,
        "top_k_features": TOP_K_FEATURES,
        "selected_indices": selected_indices,
        "original_num_features": len(feature_dataset[0]["features"]),
        "selected_num_features": len(selected_indices),
        "num_samples": len(selected_dataset),
    }

    save_pickle(SELECTED_FEATURE_DATASET_FILE, selected_dataset)
    save_json(FEATURE_RANKING_FILE, ranking)
    save_json(FEATURE_SELECTION_PARAMETERS_FILE, parameters)

    print()
    print("Seleção de features concluída.")
    print("Features originais:", parameters["original_num_features"])
    print("Features selecionadas:", parameters["selected_num_features"])
    print("Top índices:", selected_indices[:10])

    print()
    print("Dataset reduzido salvo em:")
    print(SELECTED_FEATURE_DATASET_FILE)

    print()
    print("Ranking salvo em:")
    print(FEATURE_RANKING_FILE)

    print()
    print("Parâmetros salvos em:")
    print(FEATURE_SELECTION_PARAMETERS_FILE)


if __name__ == "__main__":
    main()