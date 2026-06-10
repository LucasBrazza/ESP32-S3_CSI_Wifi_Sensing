"""
Real-Time Inference Simulation

This module simulates the real-time CSI classification pipeline using
previously recorded binary files.

Instead of training a new model, this script loads the calibration and
classification parameters saved in:

    pipeline_parameters.json

The goal is to verify whether the trained pipeline can classify CSI
windows using only the saved parameters, which is the same condition
expected during real-time deployment.

Simulated real-time flow:

    raw_bin
    ↓
    amplitude extraction
    ↓
    packet validation
    ↓
    fixed-size buffer
    ↓
    Hampel filtering
    ↓
    moving average smoothing
    ↓
    Z-score normalization using saved parameters
    ↓
    selected subcarrier filtering
    ↓
    feature extraction
    ↓
    selected feature filtering
    ↓
    decision tree prediction
    ↓
    detected class

This file is an intermediate step between offline training and true
serial real-time inference from the ESP32.
"""

from pathlib import Path
import sys

# ================= PATHS =================

TOOLS_DIR = Path(__file__).resolve().parents[1]
CLASSIFICATION_DIR = TOOLS_DIR / "classification"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

if str(CLASSIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFICATION_DIR))


from csi.csi_binary_io import read_packets

from pipeline_parameters import load_pipeline_parameters

from csi_pipeline_core import (
    packets_to_amplitude_matrix,
    remove_invalid_packets,
    hampel_filter_matrix,
    moving_average_matrix,
    apply_zscore_parameters,
)

from subcarrier_correlation import filter_matrix_by_subcarriers

from feature_extraction import extract_features_from_window

from decision_tree import predict_one


# ================= CONFIG =================

"""
Simulation configuration.

PARAMETERS_FILE points to the saved training output containing:

    - Z-score means
    - Z-score standard deviations
    - selected subcarriers
    - selected feature indices
    - trained decision tree
    - window and step sizes

TEST_BIN_FILE defines which recorded CSI binary file will be processed
as a simulated real-time stream.
"""

PARAMETERS_FILE = "preprocessing/pipeline_parameters.json"

# TEST_BIN_FILE = "datasets/raw_bin/empty_20260607_160100.bin"
TEST_BIN_FILE = "datasets/raw_bin/static_presence_20260607_160319.bin"
# TEST_BIN_FILE = "datasets/raw_bin/movement_20260607_160442.bin"

# ================= HELPERS =================

def select_features(features, selected_indices):
    """
    Select only the feature positions chosen during training.

    Feature extraction generates a full feature vector, but the model
    was trained using only the best-ranked features selected by Fisher
    Score.

    This function reproduces the same feature selection step during
    inference.

    Output:

        selected_features[index]
    """
    selected = []

    for index in selected_indices:
        if index < len(features):
            selected.append(features[index])

    return selected


def classify_window(window, parameters):
    """
    Classify a single CSI window using saved pipeline parameters.

    Input:

        window[packet][subcarrier]

    The function applies the same processing sequence used during
    training, but without recalculating calibration values.

    Processing sequence:

        1. Remove invalid packets
        2. Apply Hampel filter
        3. Apply moving average
        4. Normalize using saved mean and standard deviation
        5. Keep only selected subcarriers
        6. Extract features
        7. Keep only selected features
        8. Predict class using the saved decision tree

    Z-score formula:

        z = (x - μ) / σ

    where:

        μ and σ are loaded from the training parameters.

    If the cleaned window has fewer packets than required, the function
    returns None because the classifier expects a complete fixed-size
    window.
    """ 

    means = parameters["means"]
    stds = parameters["stds"]
    selected_subcarriers = parameters["selected_subcarriers"]
    selected_indices = parameters["selected_indices"]
    tree = parameters["decision_tree"]

    clean = remove_invalid_packets(window)

    print(
        "Debug janela | window:",
        len(window),
        "| clean:",
        len(clean),
    )

    if len(clean) < parameters["window_size"]:
        print(
            "Janela descartada: pacotes válidos insuficientes"
        )
        return None

    hampel = hampel_filter_matrix(
        clean,
        window_size=5,
        n_sigmas=3.0,
    )

    smoothed = moving_average_matrix(
        hampel,
        window_size=3,
    )

    normalized = apply_zscore_parameters(
        smoothed,
        means,
        stds,
    )

    reduced = filter_matrix_by_subcarriers(
        normalized,
        selected_subcarriers,
    )

    features = extract_features_from_window(
        reduced
    )

    selected_features = select_features(
        features,
        selected_indices,
    )

    print(
        "Features:",
        len(features),
        "| Selecionadas:",
        len(selected_features),
    )

    if len(selected_features) != len(selected_indices):
        print(
            "Janela descartada: quantidade de features incorreta"
        )
        return None

    predicted = predict_one(
        tree,
        selected_features,
    )

    return predicted


def simulate_realtime_from_bin(file_path, parameters):
    """
    Simulate real-time classification from a recorded binary CSI file.

    The file is first decoded into CSI packets and converted into
    amplitude rows.

    Invalid packets are removed before entering the buffer so that the
    simulated behavior remains consistent with the training pipeline.

    A fixed-size buffer is then updated packet by packet.

    Window generation rule:

        classify when:

            (valid_packet_index - window_size) % step_size == 0

    This reproduces sliding-window behavior in a streaming scenario.

    Example:

        window_size = 20
        step_size = 5

        classifications occur at valid packets:

            20, 25, 30, 35, ...

    The function returns the predicted class for each processed window.
    """
    packets = read_packets(file_path)

    amplitudes = packets_to_amplitude_matrix(packets)

    valid_amplitudes = remove_invalid_packets(
        amplitudes
    )

    print()
    print("Pacotes brutos:", len(packets))
    print("Pacotes com amplitude:", len(amplitudes))
    print("Pacotes válidos:", len(valid_amplitudes))

    window_size = parameters["window_size"]
    step_size = parameters["step_size"]

    predictions = []

    buffer = []

    for valid_packet_index, row in enumerate(
        valid_amplitudes,
        start=1,
    ):
        buffer.append(row)

        if len(buffer) < window_size:
            continue

        if len(buffer) > window_size:
            buffer = buffer[-window_size:]

        if (valid_packet_index - window_size) % step_size != 0:
            continue

        predicted = classify_window(
            buffer,
            parameters,
        )

        if predicted is None:
            print(
                "Pacote válido:",
                valid_packet_index,
                "| Sem classificação",
            )
            continue

        predictions.append(predicted)

        print(
            "Pacote válido:",
            valid_packet_index,
            "| Classe detectada:",
            predicted,
        )

    return predictions


def summarize_predictions(predictions):
    """
    Summarize the predicted classes generated during simulation.

    The function counts how many windows were assigned to each class and
    reports the majority class.

    The majority class is computed as:

        majority = class with the highest prediction count

    This is useful for file-level interpretation, since each binary file
    produces multiple window-level predictions.
    """
    counts = {}

    for prediction in predictions:
        if prediction not in counts:
            counts[prediction] = 0

        counts[prediction] += 1

    print()
    print("Resumo das predições:")

    for label, count in counts.items():
        print(label, ":", count)

    if predictions:
        majority = max(counts, key=counts.get)

        print()
        print("Classe predominante:", majority)


