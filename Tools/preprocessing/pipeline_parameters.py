import json


def save_pipeline_parameters(
    file_path,
    means,
    stds,
    selected_indices,
    window_size,
    step_size,
):
    data = {
        "means": means,
        "stds": stds,
        "selected_indices": selected_indices,
        "window_size": window_size,
        "step_size": step_size,
    }

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_pipeline_parameters(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data