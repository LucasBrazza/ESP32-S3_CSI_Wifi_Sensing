"""
Pipeline Parameter Persistence

This module is responsible for saving and loading all calibration and
training outputs required for real-time inference.

During training, several parameters are learned or selected:

    - Z-score means
    - Z-score standard deviations
    - selected subcarriers
    - selected feature indices
    - window size
    - step size
    - correlation threshold
    - number of selected features
    - trained decision tree

These parameters are stored in a JSON file so that the real-time
pipeline can reproduce exactly the same preprocessing and classification
steps without retraining the model.

Pipeline position:

    training
    ↓
    save parameters
    ↓
    pipeline_parameters.json
    ↓
    load parameters
    ↓
    real-time inference
    
The JSON format was chosen because it is human-readable, easy to
inspect during development, and can be converted to embedded-friendly
data structures in future deployment stages.
"""

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
    """
    Save all calibration and model parameters to a JSON file.

    Stored information:

        means
            Z-score calibration means

        stds
            Z-score calibration standard deviations

        selected_subcarriers
            subcarriers retained after redundancy removal

        selected_indices
            feature indices retained after Fisher Score selection

        window_size
            number of packets per window

        step_size
            window displacement

        correlation_threshold
            Pearson correlation threshold used during subcarrier
            selection

        top_k_features
            number of selected features

        decision_tree
            trained classification model

    The resulting JSON file becomes the configuration source for the
    real-time pipeline.
    """
    
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
    """
    Load previously saved pipeline parameters.

    This function reconstructs the training configuration used to
    generate the model.

    The loaded parameters are reused during real-time inference to
    guarantee that:

        - the same normalization is applied
        - the same subcarriers are used
        - the same features are selected
        - the same classifier is executed

    This ensures consistency between training and deployment.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)