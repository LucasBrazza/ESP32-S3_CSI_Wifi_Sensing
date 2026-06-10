"""
CSI Training Pipeline

This module implements the complete offline training and calibration
pipeline used by the project.

Its purpose is to transform raw CSI packets collected from the ESP32
into a trained classification model and a set of calibration parameters
that can later be reused during real-time inference.

Pipeline overview:

    raw_bin
    ↓
    amplitude extraction
    ↓
    packet validation and cleaning
    ↓
    Hampel filtering
    ↓
    moving average smoothing
    ↓
    Z-score normalization
    ↓
    subcarrier redundancy removal
    ↓
    sliding windows
    ↓
    feature extraction
    ↓
    Fisher Score ranking
    ↓
    feature selection
    ↓
    decision tree training
    ↓
    pipeline_parameters.json

Training phase outputs:

    - Mean and standard deviation of each subcarrier
    - Selected non-redundant subcarriers
    - Selected feature indices
    - Decision tree model
    - Pipeline configuration parameters

These outputs are saved and later reused by the real-time pipeline,
ensuring that the same preprocessing and classification steps are
applied during deployment.

The implementation intentionally avoids NumPy and other heavy scientific
libraries whenever possible, keeping the code easier to port to
MicroPython or embedded environments in future stages of the project.
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


from core_math import (
    mean,
    median,
    std,
    is_nan_or_inf,
    amplitude,
    zscore,
    get_column,
    set_column,
    copy_matrix,
)

from sliding_window import create_labeled_windows
from feature_extraction import extract_feature_dataset
from feature_selection import (
    rank_features_by_fisher_score,
    select_top_features,
)
from pipeline_parameters import save_pipeline_parameters
from subcarrier_correlation import (
    select_non_redundant_subcarriers,
    filter_matrix_by_subcarriers,
    print_selected_subcarriers,
)
from subcarrier_analysis import (
    rank_subcarriers_by_occurrence,
    select_subcarriers_from_ranking,
    print_ranked_subcarriers,
)
from decision_tree import (
    build_tree,
    print_tree,
    leave_one_out_cross_validation,
    accuracy,
    confusion_matrix,
    print_confusion_matrix,
)

from csi.csi_binary_io import read_packets


# ================= CONFIG =================

WINDOW_SIZE = 20
STEP_SIZE = 5
TOP_K_FEATURES = 30
CORRELATION_THRESHOLD = 0.40

FEATURES_PER_SUBCARRIER = 6

FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "peak_to_peak",
    "energy",
]


# ================= LOAD =================

"""
Data loading utilities.

These functions are responsible for reading CSI binary files and
converting the raw complex CSI values into amplitude matrices.

Output format used throughout the pipeline:

    matrix[packet][subcarrier]

Each row represents a CSI packet and each column represents a
subcarrier amplitude over time.
"""

def load_bin_file(file_path):
    file_path = Path(file_path)
    packets = read_packets(file_path)

    if not packets:
        raise ValueError(f"Nenhum pacote encontrado em: {file_path}")

    return packets


def packets_to_amplitude_matrix(packets):
    matrix = []

    for packet in packets:
        real_values = packet.get("real", [])
        imag_values = packet.get("imag", [])

        if not real_values or not imag_values:
            continue

        if len(real_values) != len(imag_values):
            continue

        row = []

        for real, imag in zip(real_values, imag_values):
            row.append(amplitude(real, imag))

        matrix.append(row)

    return matrix


# ================= LIMPEZA =================

"""
Signal validation and cleaning.

The purpose of this stage is to remove corrupted or incomplete packets
before any filtering or normalization is applied.

Packets are discarded when:

    - They contain NaN values
    - They contain infinite values
    - They contain only zeros
    - Their subcarrier count differs from the expected size

This guarantees a consistent matrix structure for the remaining stages
of the pipeline.
"""

def is_valid_amplitude_row(row):
    if not row:
        return False

    total = 0.0

    for value in row:
        if value is None:
            return False

        if is_nan_or_inf(value):
            return False

        total += abs(value)

    return total != 0


def remove_invalid_packets(matrix):
    cleaned = []

    if not matrix:
        return cleaned

    expected_len = 0

    for row in matrix:
        if row:
            expected_len = len(row)
            break

    if expected_len == 0:
        return cleaned

    for row in matrix:
        if not is_valid_amplitude_row(row):
            continue

        if len(row) != expected_len:
            continue

        cleaned.append(row)

    return cleaned


# ================= HAMPEL =================

"""
Outlier removal using the Hampel filter.

The Hampel filter replaces isolated abnormal samples using the local
median of a sliding neighborhood.

For each sample:

    deviation = |x - median|

A sample is replaced when:

    |x - median| > n_sigmas × MAD

where:

    MAD = median absolute deviation

and

    scaled_MAD = 1.4826 × MAD

This filter is particularly useful for CSI data because it removes
isolated spikes without significantly affecting the overall signal
shape.
"""

def hampel_filter_1d(signal, window_size=5, n_sigmas=3.0):
    filtered = signal[:]

    scale_factor = 1.4826
    n = len(signal)

    for i in range(n):
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)

        window = signal[start:end]

        med = median(window)

        deviations = []

        for value in window:
            deviations.append(abs(value - med))

        mad = scale_factor * median(deviations)

        if mad == 0:
            continue

        if abs(signal[i] - med) > n_sigmas * mad:
            filtered[i] = med

    return filtered


def hampel_filter_matrix(matrix, window_size=5, n_sigmas=3.0):
    if not matrix:
        return []

    filtered = copy_matrix(matrix)
    num_subcarriers = len(filtered[0])

    for sc in range(num_subcarriers):
        column = get_column(filtered, sc)

        filtered_column = hampel_filter_1d(
            column,
            window_size=window_size,
            n_sigmas=n_sigmas,
        )

        set_column(filtered, sc, filtered_column)

    return filtered


# ================= MÉDIA MÓVEL =================

"""
Signal smoothing using a moving average.

Each sample is replaced by the average value of its local neighborhood.

Formula:

                Σ x(i)
    mean = ---------------
            number_of_samples

This stage reduces high-frequency fluctuations while preserving the
general behavior of the CSI signal.

The moving average is applied after the Hampel filter because most
outliers have already been removed.
"""

def moving_average_1d(signal, window_size=3):
    if window_size <= 1:
        return signal[:]

    filtered = []
    n = len(signal)
    half_window = window_size // 2

    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)

        window = signal[start:end]

        filtered.append(mean(window))

    return filtered


def moving_average_matrix(matrix, window_size=3):
    if not matrix:
        return []

    smoothed = copy_matrix(matrix)
    num_subcarriers = len(smoothed[0])

    for sc in range(num_subcarriers):
        column = get_column(smoothed, sc)

        smoothed_column = moving_average_1d(
            column,
            window_size=window_size,
        )

        set_column(smoothed, sc, smoothed_column)

    return smoothed


# ================= Z-SCORE =================

"""
Subcarrier normalization using Z-score.

Each subcarrier is normalized independently.

Formula:

        x - μ
    z = -------
           σ

where:

    μ = mean
    σ = standard deviation

This transformation places all subcarriers on a comparable scale,
reducing the influence of absolute amplitude differences.
"""

def fit_zscore_parameters(matrix):
    if not matrix:
        return [], []

    num_subcarriers = len(matrix[0])

    means = []
    stds = []

    for sc in range(num_subcarriers):
        column = get_column(matrix, sc)

        means.append(mean(column))
        stds.append(std(column))

    return means, stds


def apply_zscore_parameters(matrix, means, stds):
    if not matrix:
        return []

    normalized = copy_matrix(matrix)

    for i in range(len(matrix)):
        for sc in range(len(matrix[i])):
            normalized[i][sc] = zscore(
                matrix[i][sc],
                means[sc],
                stds[sc],
            )

    return normalized


# ================= PIPELINE DE CALIBRAÇÃO =================

"""
Training-time calibration.

All calibration files are combined into a single dataset in order to
estimate global normalization parameters.

Outputs:

    means[subcarrier]
    stds[subcarrier]

These values are later reused during real-time inference so that the
same normalization reference is applied to unseen data.
"""

def fit_preprocessing_pipeline(file_paths):
    all_rows = []

    for file_path in file_paths:
        packets = load_bin_file(file_path)
        amplitudes = packets_to_amplitude_matrix(packets)
        clean = remove_invalid_packets(amplitudes)

        for row in clean:
            all_rows.append(row)

    hampel = hampel_filter_matrix(
        all_rows,
        window_size=5,
        n_sigmas=3.0,
    )

    smoothed = moving_average_matrix(
        hampel,
        window_size=3,
    )

    means, stds = fit_zscore_parameters(smoothed)

    normalized = apply_zscore_parameters(
        smoothed,
        means,
        stds,
    )

    return {
        "means": means,
        "stds": stds,
        "normalized": normalized,
        "num_packets": len(normalized),
        "num_subcarriers": len(normalized[0]) if normalized else 0,
    }


# ================= PIPELINE DE TRANSFORMAÇÃO =================

"""
Data transformation using previously calibrated parameters.

Unlike the calibration stage, no new normalization parameters are
computed here.

The incoming data is transformed using the means and standard
deviations obtained during training.
"""

def transform_amplitude_matrix(matrix, means, stds):
    clean = remove_invalid_packets(matrix)

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

    return normalized


def transform_bin_file(file_path, means, stds):
    packets = load_bin_file(file_path)
    amplitudes = packets_to_amplitude_matrix(packets)

    return transform_amplitude_matrix(
        amplitudes,
        means,
        stds,
    )


# ================= DIAGNÓSTICO =================

"""
Dataset inspection utilities.

These functions are not part of the preprocessing pipeline itself.

Their purpose is to provide a quick overview of collected CSI files
before training begins.

The diagnostics help verify:

    - Number of packets successfully decoded
    - Number of packets converted to amplitude
    - Number of packets remaining after cleaning
    - Number of detected subcarriers

This information is useful for identifying corrupted files,
unexpected packet losses or inconsistencies between datasets.

Example:

    Raw packets:            54
    Amplitude packets:      54
    Valid packets:          53
    Subcarriers detected:   192

The diagnostics stage does not modify the data and is intended
only for inspection and debugging purposes.
"""

def inspect_bin_file(file_path):
    packets = load_bin_file(file_path)
    amplitudes = packets_to_amplitude_matrix(packets)
    clean = remove_invalid_packets(amplitudes)

    print()
    print("Arquivo:", file_path)
    print("Pacotes lidos:", len(packets))
    print("Pacotes com amplitude:", len(amplitudes))
    print("Pacotes válidos:", len(clean))

    if clean:
        print("Subportadoras:", len(clean[0]))


# ================= DATASET COM JANELAS =================

"""
Sliding-window dataset generation.

After preprocessing, the continuous CSI stream is divided into
overlapping windows.

Example:

    Window size = 20
    Step size   = 5

    packets 1-20
    packets 6-25
    packets 11-30
    ...

Each window becomes one training sample for the classifier.
"""

def build_labeled_window_dataset(
    file_label_pairs,
    window_size=20,
    step_size=5,
    correlation_threshold=0.80,
):
    file_paths = []

    for file_path, label in file_label_pairs:
        file_paths.append(file_path)

    fit_result = fit_preprocessing_pipeline(file_paths)

    means = fit_result["means"]
    stds = fit_result["stds"]
    normalized_all = fit_result["normalized"]

    selected_subcarriers = select_non_redundant_subcarriers(
        normalized_all,
        threshold=correlation_threshold,
    )

    print_selected_subcarriers(selected_subcarriers)

    dataset = []

    for file_path, label in file_label_pairs:
        normalized = transform_bin_file(
            file_path,
            means,
            stds,
        )

        reduced = filter_matrix_by_subcarriers(
            normalized,
            selected_subcarriers,
        )

        labeled_windows = create_labeled_windows(
            reduced,
            label,
            window_size=window_size,
            step_size=step_size,
        )

        for window in labeled_windows:
            dataset.append(window)

    return {
        "dataset": dataset,
        "means": means,
        "stds": stds,
        "selected_subcarriers": selected_subcarriers,
        "window_size": window_size,
        "step_size": step_size,
        "num_windows": len(dataset),
    }


# ================= ANÁLISE DE FEATURES =================

def describe_feature_index(feature_index):
    subcarrier = feature_index // FEATURES_PER_SUBCARRIER
    feature_position = feature_index % FEATURES_PER_SUBCARRIER
    feature_name = FEATURE_NAMES[feature_position]

    return subcarrier, feature_name


def print_class_count(window_dataset):
    class_count = {}

    for item in window_dataset:
        label = item["label"]

        if label not in class_count:
            class_count[label] = 0

        class_count[label] += 1

    print()
    print("Janelas por classe:")

    for label, count in class_count.items():
        print(label, ":", count)


def print_top_fisher_features(ranking, top_n=20):
    print()
    print(f"Top {top_n} features por Fisher Score:")

    for item in ranking[:top_n]:
        feature_index = item["feature_index"]
        score = item["score"]

        subcarrier, feature_name = describe_feature_index(feature_index)

        print(
            "Feature:",
            feature_index,
            "| Subportadora:",
            subcarrier,
            "| Tipo:",
            feature_name,
            "| Score:",
            score,
        )


# ================= MAIN =================

if __name__ == "__main__":
    file_label_pairs = [
        (
            "datasets/raw_bin/empty_20260607_160100.bin",
            "empty",
        ),
        (
            "datasets/raw_bin/static_presence_20260607_160319.bin",
            "static_presence",
        ),
        (
            "datasets/raw_bin/movement_20260607_160442.bin",
            "movement",
        ),
    ]

    print("Inspecionando arquivos:")

    for file_path, label in file_label_pairs:
        inspect_bin_file(file_path)

    result = build_labeled_window_dataset(
        file_label_pairs,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        correlation_threshold=CORRELATION_THRESHOLD,
    )

    print()
    print("Dataset com janelas criado.")
    print("Total de janelas:", result["num_windows"])
    print("Window size:", result["window_size"])
    print("Step size:", result["step_size"])
    print("Correlation threshold:", CORRELATION_THRESHOLD)
    print("Médias:", len(result["means"]))
    print("Desvios:", len(result["stds"]))
    print(
        "Subportadoras após correlação:",
        len(result["selected_subcarriers"])
    )

    print_class_count(result["dataset"])

    if result["dataset"]:
        first = result["dataset"][0]

        print()
        print("Primeira janela:")
        print("Label:", first["label"])
        print("Pacotes na janela:", len(first["data"]))
        print("Subportadoras:", len(first["data"][0]))

    # ================= FEATURES =================

    feature_dataset = extract_feature_dataset(
        result["dataset"]
    )

    print()
    print("Dataset de features criado.")
    print("Amostras:", len(feature_dataset))

    if feature_dataset:
        print(
            "Features por amostra:",
            len(feature_dataset[0]["features"])
        )
        print(
            "Primeira classe:",
            feature_dataset[0]["label"]
        )

    # ================= FISHER SCORE =================

    ranking = rank_features_by_fisher_score(
        feature_dataset
    )

    selected_dataset, selected_indices = select_top_features(
        feature_dataset,
        ranking,
        top_k=TOP_K_FEATURES,
    )

    print()
    print("Fisher Score concluído.")
    print(
        "Features originais:",
        len(feature_dataset[0]["features"])
    )
    print(
        "Features selecionadas:",
        len(selected_indices)
    )
    print(
        "Top índices:",
        selected_indices[:10]
    )

    print_top_fisher_features(
        ranking,
        top_n=20,
    )

    # ================= SAVE PARAMETERS =================


    print()
    print("Parâmetros do pipeline salvos em:")
    print("preprocessing/pipeline_parameters.json")

    # ================= DATASET REDUZIDO =================

    print()
    print("Dataset reduzido criado.")
    print("Amostras:", len(selected_dataset))

    if selected_dataset:
        print(
            "Features por amostra reduzida:",
            len(selected_dataset[0]["features"])
        )
        print(
            "Primeira classe:",
            selected_dataset[0]["label"]
        )

    # ================= ANÁLISE DAS SUBPORTADORAS =================

    ranked_subcarriers = rank_subcarriers_by_occurrence(
        ranking,
        top_n=TOP_K_FEATURES,
    )

    selected_subcarriers_fisher = select_subcarriers_from_ranking(
        ranking,
        top_n=TOP_K_FEATURES,
        min_count=1,
    )

    print_ranked_subcarriers(
        ranked_subcarriers
    )

    print()
    print("Subportadoras selecionadas pelo Fisher:")
    print(selected_subcarriers_fisher)
    print("Total:", len(selected_subcarriers_fisher))

    # ================= CLASSIFICAÇÃO =================

    print()
    print("Treinando árvore de decisão...")

    tree = build_tree(
        selected_dataset,
        max_depth=4,
        min_samples_split=2,
    )

    save_pipeline_parameters(
        "preprocessing/pipeline_parameters.json",
        means=result["means"],
        stds=result["stds"],
        selected_subcarriers=result["selected_subcarriers"],
        selected_indices=selected_indices,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        correlation_threshold=CORRELATION_THRESHOLD,
        top_k_features=TOP_K_FEATURES,
        decision_tree=tree,
    )
    
    
    print()
    print("Árvore treinada:")
    print_tree(tree)

    predictions = leave_one_out_cross_validation(
        selected_dataset,
        max_depth=4,
        min_samples_split=2,
    )

    acc = accuracy(predictions)

    print()
    print("Validação Leave-One-Out concluída.")
    print("Acurácia:", acc)

    labels, matrix = confusion_matrix(predictions)

    print_confusion_matrix(
        labels,
        matrix,
    )

    # ================= RESUMO FINAL =================

    print()
    print("Resumo final:")
    print(
        "Subportadoras após correlação:",
        len(result["selected_subcarriers"])
    )
    print(
        "Subportadoras relevantes pelo Fisher:",
        len(selected_subcarriers_fisher)
    )
    print(
        "Features finais:",
        len(selected_indices)
    )
    print(
        "Acurácia LOOCV:",
        acc,
    )