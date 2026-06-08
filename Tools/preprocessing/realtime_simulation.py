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

PARAMETERS_FILE = "preprocessing/pipeline_parameters.json"

# TEST_BIN_FILE = "datasets/raw_bin/empty_20260607_160100.bin"
TEST_BIN_FILE = "datasets/raw_bin/static_presence_20260607_160319.bin"
# TEST_BIN_FILE = "datasets/raw_bin/movement_20260607_160442.bin"

# ================= HELPERS =================

def select_features(features, selected_indices):
    selected = []

    for index in selected_indices:
        if index < len(features):
            selected.append(features[index])

    return selected


def classify_window(window, parameters):
    """
    Classifica uma janela CSI já em formato:
        window[pacote][subportadora]
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


# ================= MAIN =================

if __name__ == "__main__":
    parameters = load_pipeline_parameters(
        PARAMETERS_FILE
    )

    print("Parâmetros carregados:")
    print("Window size:", parameters["window_size"])
    print("Step size:", parameters["step_size"])
    print("Subportadoras:", len(parameters["selected_subcarriers"]))
    print("Features:", len(parameters["selected_indices"]))

    print()
    print("Arquivo testado:")
    print(TEST_BIN_FILE)

    predictions = simulate_realtime_from_bin(
        TEST_BIN_FILE,
        parameters,
    )

    summarize_predictions(predictions)