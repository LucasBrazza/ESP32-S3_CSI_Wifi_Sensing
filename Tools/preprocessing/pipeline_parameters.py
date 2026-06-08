import json


def save_pipeline_parameters(
    file_path,
    means,
    stds,
    selected_subcarriers,
    selected_indices,
    window_size,
    step_size,
    correlation_threshold,
    top_k_features,
    decision_tree=None,
):
    data = {
        "means": means,
        "stds": stds,
        "selected_subcarriers": selected_subcarriers,
        "selected_indices": selected_indices,
        "window_size": window_size,
        "step_size": step_size,
        "correlation_threshold": correlation_threshold,
        "top_k_features": top_k_features,
        "decision_tree": decision_tree,
    }

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_pipeline_parameters(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)