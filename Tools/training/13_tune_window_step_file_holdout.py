"""
Window and Step Tuning with File-Based Holdout

Experiment 014

This script evaluates different sliding-window configurations for the CSI
pipeline.

For each window/step pair, it:

    1. Regenerates the feature dataset using the selected window size and
       step size.
    2. Runs top-k tuning with Fisher Score computed only on the training
       set.
    3. Saves each individual run under a dedicated experiment folder.
    4. Creates a summary comparing all window/step configurations.

This experiment is intended to verify whether the current temporal
resolution is adequate for distinguishing:

    empty
    static_presence
    movement

The main target is improving movement recognition while keeping the
pipeline compatible with real-time embedded execution.
"""

import csv
import subprocess
import sys
from pathlib import Path

from Tools.common.config import (
    MODEL,
    MAX_TREE_DEPTH,
    MIN_SAMPLES_SPLIT,
)

from Tools.common.project_paths import (
    RESULTS_DIR,
    FEATURE_DATASET_FILE,
)

from Tools.common.io_utils import load_pickle


EXPERIMENT_ID = "014"
EXPERIMENT_FOLDER_NAME = "014_window_step_tuning_summary"

TOP_K_VALUES = [
    30,
    50,
    70,
    100,
    126,
    150,
    180,
    200,
    231,
]

WINDOW_STEP_TESTS = [
    {
        "window_size": 5,
        "step_size": 1,
    },
    {
        "window_size": 5,
        "step_size": 2,
    },
    {
        "window_size": 6,
        "step_size": 2,
    },
    {
        "window_size": 8,
        "step_size": 2,
    },
    {
        "window_size": 8,
        "step_size": 4,
    },
    {
        "window_size": 10,
        "step_size": 2,
    },
    {
        "window_size": 10,
        "step_size": 5,
    },
    {
        "window_size": 12,
        "step_size": 3,
    },
    {
        "window_size": 15,
        "step_size": 5,
    },
]


MIN_VALID_SAMPLES = 200
MIN_VALID_CLASS_F1 = 0.20


def run_command(command):
    print()
    print("Running command:")
    print(" ".join(command))
    print("-" * 70)

    subprocess.run(
        command,
        check=True,
    )


def build_experiment_id(window_size, step_size):
    return f"014_w{window_size}_s{step_size}"


def build_experiment_folder(window_size, step_size):
    return f"014_window{window_size}_step{step_size}_temporal_features"


def get_run_dir(experiment_folder):
    return RESULTS_DIR / "runs" / experiment_folder


def get_summary_paths():
    run_dir = RESULTS_DIR / "runs" / EXPERIMENT_FOLDER_NAME
    tables_dir = run_dir / "tables"
    reports_dir = run_dir / "reports"

    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "tables_dir": tables_dir,
        "reports_dir": reports_dir,
        "summary_csv": tables_dir / "window_step_tuning_results.csv",
        "experiment_notes": run_dir / "experiment_notes.txt",
    }


def read_best_topk_result(experiment_folder):
    run_dir = get_run_dir(experiment_folder)
    csv_path = run_dir / "tables" / "topk_tuning_results.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            "Top-K tuning result file was not found: "
            f"{csv_path}"
        )

    rows = []

    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(
            "Top-K tuning result file is empty: "
            f"{csv_path}"
        )

    ordered_rows = sorted(
        rows,
        key=lambda row: (
            float(row["macro_f1"]),
            float(row["accuracy"]),
            -int(row["top_k"]),
            -int(row["tree_nodes"]),
        ),
        reverse=True,
    )

    return ordered_rows[0]


def get_current_feature_dataset_summary():
    feature_dataset = load_pickle(FEATURE_DATASET_FILE)

    if not feature_dataset:
        return {
            "samples": 0,
            "features_per_sample": 0,
        }

    return {
        "samples": len(feature_dataset),
        "features_per_sample": len(feature_dataset[0]["features"]),
    }


def save_summary_csv(results, output_path):
    fieldnames = [
        "experiment_id",
        "experiment_folder",
        "window_size",
        "step_size",
        "samples",
        "features_per_sample",
        "top_k",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "empty_f1",
        "static_presence_f1",
        "movement_f1",
        "tree_nodes",
        "tree_decision_nodes",
        "tree_leaf_nodes",
        "tree_actual_max_depth",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result)


def save_experiment_notes(results, best_result, output_path):
    lines = []

    lines.append("Experiment 014 - Window and step tuning")
    lines.append("")
    lines.append("Description:")
    lines.append(
        "This experiment evaluates different sliding-window sizes and step "
        "sizes using the current temporal CSI feature set."
    )
    lines.append("")
    lines.append("Method:")
    lines.append(
        "For each window/step pair, the feature dataset is regenerated and "
        "Top-K tuning is executed using file-based holdout validation. Fisher "
        "Score ranking is computed only on the training data."
    )
    lines.append("")
    lines.append("Classifier:")
    lines.append(f"- Model: {MODEL}")
    lines.append(f"- Max tree depth: {MAX_TREE_DEPTH}")
    lines.append(f"- Min samples split: {MIN_SAMPLES_SPLIT}")
    lines.append("")
    lines.append("Top-K values tested for each window/step pair:")
    lines.append(str(TOP_K_VALUES))
    lines.append("")
    lines.append("Window/step configurations tested:")

    for test in WINDOW_STEP_TESTS:
        lines.append(
            f"- window_size={test['window_size']}, "
            f"step_size={test['step_size']}"
        )

    lines.append("")
    lines.append("Best result:")
    lines.append(f"- experiment_id: {best_result['experiment_id']}")
    lines.append(f"- folder: {best_result['experiment_folder']}")
    lines.append(f"- window_size: {best_result['window_size']}")
    lines.append(f"- step_size: {best_result['step_size']}")
    lines.append(f"- samples: {best_result['samples']}")
    lines.append(f"- features per sample: {best_result['features_per_sample']}")
    lines.append(f"- top_k: {best_result['top_k']}")
    lines.append(f"- accuracy: {float(best_result['accuracy']):.6f}")
    lines.append(f"- macro F1-score: {float(best_result['macro_f1']):.6f}")
    lines.append(f"- weighted F1-score: {float(best_result['weighted_f1']):.6f}")
    lines.append(f"- empty F1-score: {float(best_result['empty_f1']):.6f}")
    lines.append(
        f"- static_presence F1-score: "
        f"{float(best_result['static_presence_f1']):.6f}"
    )
    lines.append(f"- movement F1-score: {float(best_result['movement_f1']):.6f}")
    lines.append(f"- tree nodes: {best_result['tree_nodes']}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "This experiment should be used to decide whether the current window "
        "configuration captures enough temporal information for movement "
        "recognition. If larger windows improve movement F1-score, the previous "
        "window was likely too short. If performance does not improve, the next "
        "step should be adding more motion-specific features or revising "
        "preprocessing."
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def is_valid_window_result(result):
    if int(result["samples"]) < MIN_VALID_SAMPLES:
        return False

    if float(result["empty_f1"]) <= MIN_VALID_CLASS_F1:
        return False

    if float(result["static_presence_f1"]) <= MIN_VALID_CLASS_F1:
        return False

    if float(result["movement_f1"]) <= MIN_VALID_CLASS_F1:
        return False

    return True


def select_best_window_result(results):
    valid_results = []

    for result in results:
        if is_valid_window_result(result):
            valid_results.append(result)

    if not valid_results:
        valid_results = results

    ordered_results = sorted(
        valid_results,
        key=lambda result: (
            float(result["macro_f1"]),
            float(result["accuracy"]),
            float(result["movement_f1"]),
            -int(result["top_k"]),
            -int(result["tree_nodes"]),
        ),
        reverse=True,
    )

    return ordered_results[0]


def update_experiment_index(best_result):
    index_path = RESULTS_DIR / "experiment_index.csv"

    fieldnames = [
        "experiment_id",
        "folder",
        "description",
        "model",
        "validation_method",
        "top_k",
        "max_depth",
        "min_samples_split",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "main_observation",
    ]

    existing_rows = []

    if index_path.exists():
        with open(index_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row.get("experiment_id") != EXPERIMENT_ID:
                    existing_rows.append(row)

    new_row = {
        "experiment_id": EXPERIMENT_ID,
        "folder": EXPERIMENT_FOLDER_NAME,
        "description": "Window and step tuning with temporal CSI features",
        "model": MODEL,
        "validation_method": "File-based stratified holdout 80/20",
        "top_k": best_result["top_k"],
        "max_depth": MAX_TREE_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "accuracy": f"{float(best_result['accuracy']):.6f}",
        "macro_f1": f"{float(best_result['macro_f1']):.6f}",
        "weighted_f1": f"{float(best_result['weighted_f1']):.6f}",
        "main_observation": (
            f"Best window_size={best_result['window_size']}, "
            f"step_size={best_result['step_size']}, "
            f"movement_f1={float(best_result['movement_f1']):.6f}"
        ),
    }

    existing_rows.append(new_row)

    with open(index_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in existing_rows:
            writer.writerow(row)


def run_feature_extraction(window_size, step_size):
    command = [
        sys.executable,
        "-m",
        "Tools.preprocessing.02_extract_features",
        "--window-size",
        str(window_size),
        "--step-size",
        str(step_size),
    ]

    run_command(command)


def run_topk_tuning(experiment_id, experiment_folder):
    top_k_values = ",".join(str(value) for value in TOP_K_VALUES)

    command = [
        sys.executable,
        "-m",
        "Tools.training.07_tune_topk_file_holdout",
        "--experiment-id",
        experiment_id,
        "--experiment-folder",
        experiment_folder,
        "--top-k-values",
        top_k_values,
    ]

    run_command(command)


def run_experiment():
    results = []

    total_tests = len(WINDOW_STEP_TESTS)

    for test_index, test in enumerate(WINDOW_STEP_TESTS, start=1):
        window_size = test["window_size"]
        step_size = test["step_size"]

        experiment_id = build_experiment_id(
            window_size,
            step_size,
        )

        experiment_folder = build_experiment_folder(
            window_size,
            step_size,
        )

        print()
        print("=" * 70)
        print(
            f"Window/step test {test_index}/{total_tests}: "
            f"window_size={window_size}, step_size={step_size}"
        )
        print("=" * 70)

        run_feature_extraction(
            window_size,
            step_size,
        )

        feature_summary = get_current_feature_dataset_summary()

        run_topk_tuning(
            experiment_id,
            experiment_folder,
        )

        best_topk_result = read_best_topk_result(
            experiment_folder,
        )

        result = {
            "experiment_id": experiment_id,
            "experiment_folder": experiment_folder,
            "window_size": window_size,
            "step_size": step_size,
            "samples": feature_summary["samples"],
            "features_per_sample": feature_summary["features_per_sample"],
            "top_k": best_topk_result["top_k"],
            "accuracy": best_topk_result["accuracy"],
            "macro_f1": best_topk_result["macro_f1"],
            "weighted_f1": best_topk_result["weighted_f1"],
            "empty_f1": best_topk_result["empty_f1"],
            "static_presence_f1": best_topk_result["static_presence_f1"],
            "movement_f1": best_topk_result["movement_f1"],
            "tree_nodes": best_topk_result["tree_nodes"],
            "tree_decision_nodes": best_topk_result["tree_decision_nodes"],
            "tree_leaf_nodes": best_topk_result["tree_leaf_nodes"],
            "tree_actual_max_depth": best_topk_result["tree_actual_max_depth"],
        }

        results.append(result)

    best_result = select_best_window_result(results)

    paths = get_summary_paths()

    save_summary_csv(
        results,
        paths["summary_csv"],
    )

    save_experiment_notes(
        results,
        best_result,
        paths["experiment_notes"],
    )

    update_experiment_index(best_result)

    print()
    print("=" * 70)
    print("Window/step tuning finished.")
    print("=" * 70)
    print("Best result:")
    print(f"experiment_id: {best_result['experiment_id']}")
    print(f"window_size: {best_result['window_size']}")
    print(f"step_size: {best_result['step_size']}")
    print(f"samples: {best_result['samples']}")
    print(f"features_per_sample: {best_result['features_per_sample']}")
    print(f"top_k: {best_result['top_k']}")
    print(f"accuracy: {float(best_result['accuracy']):.6f}")
    print(f"macro_f1: {float(best_result['macro_f1']):.6f}")
    print(f"weighted_f1: {float(best_result['weighted_f1']):.6f}")
    print(f"movement_f1: {float(best_result['movement_f1']):.6f}")
    print()
    print("Summary saved to:")
    print(paths["run_dir"])

    print()
    print("Restoring feature dataset using the best window/step configuration...")
    run_feature_extraction(
        int(best_result["window_size"]),
        int(best_result["step_size"]),
    )


def main():
    print()
    print("Window and Step Tuning with File-Based Holdout")
    print("=" * 70)
    print("Experiment:", EXPERIMENT_ID)
    print("Model:", MODEL)
    print("Max tree depth:", MAX_TREE_DEPTH)
    print("Min samples split:", MIN_SAMPLES_SPLIT)
    print("Window/step tests:", WINDOW_STEP_TESTS)
    print("Top-K values:", TOP_K_VALUES)

    run_experiment()


if __name__ == "__main__":
    main()