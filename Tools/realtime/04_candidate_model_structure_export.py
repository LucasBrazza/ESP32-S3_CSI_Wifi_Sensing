"""
Candidate Model Structure Export

Experiment 023

This experiment inspects the current candidate model for embedded use:

    GradientBoostingClassifier
    n_estimators = 20
    max_depth = 3
    top_k = 231

Important:

    In multiclass classification, scikit-learn GradientBoostingClassifier
    usually creates one regression tree per class per boosting stage.

    Therefore, with 20 estimators and 3 classes, the actual model may contain:

        20 * 3 = 60 trees

This script measures:

    - total trees
    - total nodes
    - internal decision nodes
    - leaf nodes
    - max depth
    - estimated comparisons per window
    - features effectively used by the trained model
"""

import csv
import json
import importlib
from collections import Counter


base = importlib.import_module(
    "Tools.realtime.02_scenario_stream_state_machine"
)


try:
    from Tools.common.project_paths import PREPROCESSING_PARAMETERS_FILE
except ImportError:
    PREPROCESSING_PARAMETERS_FILE = None


EXPERIMENT_ID = "023"
EXPERIMENT_FOLDER_NAME = "023_candidate_model_structure_export"

FULL_DATASET_EXPORT_SEED = 42

FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "peak_to_peak",
    "energy",
    "mean_abs_diff",
    "max_abs_diff",
    "diff_energy",
    "std_diff",
    "slope",
]

FEATURES_PER_SUBCARRIER = len(FEATURE_NAMES)
EXPECTED_SUBCARRIERS = base.EXPECTED_FEATURES_PER_SAMPLE // FEATURES_PER_SUBCARRIER


# ================= BASIC HELPERS =================

def safe_division(numerator, denominator):
    if denominator == 0:
        return 0.0

    return numerator / denominator


def mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def std(values):
    if not values:
        return 0.0

    average = mean(values)
    total = 0.0

    for value in values:
        difference = value - average
        total += difference * difference

    return (total / len(values)) ** 0.5


def save_csv(rows, output_path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def flatten_first_value(value):
    current = value

    while True:
        try:
            current = current[0]
        except (TypeError, IndexError):
            return float(current)


# ================= FEATURE DESCRIPTION =================

def load_json_if_exists(path):
    if path is None:
        return None

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def find_numeric_lists(data, key_path=""):
    found = []

    if isinstance(data, dict):
        for key, value in data.items():
            next_path = key

            if key_path:
                next_path = f"{key_path}.{key}"

            found.extend(
                find_numeric_lists(value, next_path)
            )

    elif isinstance(data, list):
        is_numeric_list = True

        for value in data:
            if not isinstance(value, (int, float)):
                is_numeric_list = False

        if is_numeric_list:
            found.append(
                {
                    "key_path": key_path,
                    "values": data,
                }
            )

        for index, value in enumerate(data):
            next_path = f"{key_path}[{index}]"
            found.extend(
                find_numeric_lists(value, next_path)
            )

    return found


def load_selected_subcarriers():
    parameters = load_json_if_exists(PREPROCESSING_PARAMETERS_FILE)

    if parameters is None:
        return []

    numeric_lists = find_numeric_lists(parameters)

    candidates = []

    for item in numeric_lists:
        values = item["values"]

        if len(values) == EXPECTED_SUBCARRIERS:
            candidates.append(item)

    for item in candidates:
        key_path = item["key_path"].lower()

        if "subcarrier" in key_path and (
            "selected" in key_path
            or "remaining" in key_path
            or "kept" in key_path
        ):
            return item["values"]

    for item in candidates:
        key_path = item["key_path"].lower()

        if "subcarrier" in key_path:
            return item["values"]

    if candidates:
        return candidates[0]["values"]

    return []


def describe_original_feature(original_feature_index, selected_subcarriers):
    subcarrier_position = original_feature_index // FEATURES_PER_SUBCARRIER
    feature_type_index = original_feature_index % FEATURES_PER_SUBCARRIER

    feature_name = ""

    if 0 <= feature_type_index < len(FEATURE_NAMES):
        feature_name = FEATURE_NAMES[feature_type_index]

    subcarrier_id = ""

    if 0 <= subcarrier_position < len(selected_subcarriers):
        subcarrier_id = selected_subcarriers[subcarrier_position]

    return {
        "original_feature_index": original_feature_index,
        "subcarrier_position": subcarrier_position,
        "subcarrier_id": subcarrier_id,
        "feature_type_index": feature_type_index,
        "feature_name": feature_name,
    }


# ================= MODEL STRUCTURE =================

def get_all_trees(model):
    trees = []

    estimators = model.estimators_

    for stage_index in range(estimators.shape[0]):
        for class_index in range(estimators.shape[1]):
            class_label = str(model.classes_[class_index])

            trees.append(
                {
                    "stage_index": stage_index,
                    "class_index": class_index,
                    "class_label": class_label,
                    "tree_model": estimators[stage_index, class_index],
                }
            )

    return trees


def count_tree_path_decisions(tree_model, row):
    tree = tree_model.tree_

    node_id = 0
    comparisons = 0

    while tree.children_left[node_id] != tree.children_right[node_id]:
        feature_index = tree.feature[node_id]
        threshold = tree.threshold[node_id]

        if row[feature_index] <= threshold:
            node_id = tree.children_left[node_id]
        else:
            node_id = tree.children_right[node_id]

        comparisons += 1

    return comparisons


def analyze_model_structure(
    model,
    selected_indices,
    x_values,
    seed,
    training_mode,
    train_windows,
    test_windows,
    test_groups,
):
    all_trees = get_all_trees(model)

    total_nodes = 0
    internal_nodes = 0
    leaf_nodes = 0
    max_depth = 0

    used_model_feature_counter = Counter()
    used_original_feature_counter = Counter()

    for item in all_trees:
        tree_model = item["tree_model"]
        tree = tree_model.tree_

        total_nodes += tree.node_count
        max_depth = max(max_depth, tree.max_depth)

        for node_id in range(tree.node_count):
            is_leaf = tree.children_left[node_id] == tree.children_right[node_id]

            if is_leaf:
                leaf_nodes += 1
            else:
                internal_nodes += 1

                model_feature_position = int(tree.feature[node_id])
                original_feature_index = int(selected_indices[model_feature_position])

                used_model_feature_counter[model_feature_position] += 1
                used_original_feature_counter[original_feature_index] += 1

    comparison_counts = []

    for row in x_values:
        comparisons = 0

        for item in all_trees:
            comparisons += count_tree_path_decisions(
                item["tree_model"],
                row,
            )

        comparison_counts.append(comparisons)

    summary = {
        "seed": seed,
        "training_mode": training_mode,
        "n_estimators": model.n_estimators,
        "n_classes": len(model.classes_),
        "total_trees": len(all_trees),
        "total_nodes": total_nodes,
        "internal_decision_nodes": internal_nodes,
        "leaf_nodes": leaf_nodes,
        "max_depth": max_depth,
        "selected_features": len(selected_indices),
        "used_model_feature_positions": len(used_model_feature_counter),
        "used_original_features": len(used_original_feature_counter),
        "train_windows": train_windows,
        "test_windows": test_windows,
        "test_groups": test_groups,
        "comparisons_mean_per_window": mean(comparison_counts),
        "comparisons_std_per_window": std(comparison_counts),
        "comparisons_max_per_window": max(comparison_counts) if comparison_counts else 0,
        "comparisons_min_per_window": min(comparison_counts) if comparison_counts else 0,
    }

    return summary, used_original_feature_counter


def export_tree_nodes(model, selected_indices, selected_subcarriers):
    rows = []

    all_trees = get_all_trees(model)

    for item in all_trees:
        stage_index = item["stage_index"]
        class_index = item["class_index"]
        class_label = item["class_label"]
        tree_model = item["tree_model"]
        tree = tree_model.tree_

        for node_id in range(tree.node_count):
            is_leaf = tree.children_left[node_id] == tree.children_right[node_id]

            model_feature_position = ""
            original_feature_index = ""
            subcarrier_position = ""
            subcarrier_id = ""
            feature_type_index = ""
            feature_name = ""
            threshold = ""

            if not is_leaf:
                model_feature_position = int(tree.feature[node_id])
                original_feature_index = int(selected_indices[model_feature_position])
                description = describe_original_feature(
                    original_feature_index,
                    selected_subcarriers,
                )

                subcarrier_position = description["subcarrier_position"]
                subcarrier_id = description["subcarrier_id"]
                feature_type_index = description["feature_type_index"]
                feature_name = description["feature_name"]
                threshold = float(tree.threshold[node_id])

            rows.append(
                {
                    "stage_index": stage_index,
                    "class_index": class_index,
                    "class_label": class_label,
                    "node_id": node_id,
                    "is_leaf": is_leaf,
                    "left_child": int(tree.children_left[node_id]),
                    "right_child": int(tree.children_right[node_id]),
                    "model_feature_position": model_feature_position,
                    "original_feature_index": original_feature_index,
                    "subcarrier_position": subcarrier_position,
                    "subcarrier_id": subcarrier_id,
                    "feature_type_index": feature_type_index,
                    "feature_name": feature_name,
                    "threshold": threshold,
                    "node_value": flatten_first_value(tree.value[node_id]),
                }
            )

    return rows


def build_used_feature_rows(counter, selected_subcarriers):
    total_splits = sum(counter.values())

    rows = []

    for original_feature_index, count in counter.most_common():
        description = describe_original_feature(
            int(original_feature_index),
            selected_subcarriers,
        )

        rows.append(
            {
                "original_feature_index": int(original_feature_index),
                "subcarrier_position": description["subcarrier_position"],
                "subcarrier_id": description["subcarrier_id"],
                "feature_type_index": description["feature_type_index"],
                "feature_name": description["feature_name"],
                "split_count": int(count),
                "split_frequency": safe_division(count, total_splits),
            }
        )

    return rows


def summarize_holdout_structure(summary_rows):
    numeric_keys = [
        "total_trees",
        "total_nodes",
        "internal_decision_nodes",
        "leaf_nodes",
        "max_depth",
        "used_original_features",
        "comparisons_mean_per_window",
        "comparisons_max_per_window",
    ]

    output = {}

    for key in numeric_keys:
        values = []

        for row in summary_rows:
            values.append(float(row[key]))

        output[f"{key}_mean"] = mean(values)
        output[f"{key}_std"] = std(values)
        output[f"{key}_min"] = min(values)
        output[f"{key}_max"] = max(values)

    output["seeds"] = len(summary_rows)

    return output


# ================= OUTPUT =================

def get_experiment_paths():
    run_dir = base.RESULTS_DIR / "runs" / EXPERIMENT_FOLDER_NAME
    tables_dir = run_dir / "tables"
    reports_dir = run_dir / "reports"

    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "tables_dir": tables_dir,
        "reports_dir": reports_dir,

        "structure_by_seed": tables_dir / "model_structure_by_seed.csv",
        "structure_summary": tables_dir / "model_structure_summary.csv",
        "used_features_holdout": tables_dir / "used_features_holdout_aggregate.csv",
        "used_features_full": tables_dir / "used_features_full_dataset_seed42.csv",
        "tree_nodes_full": tables_dir / "tree_nodes_full_dataset_seed42.csv",

        "embedded_snapshot": reports_dir / "embedded_model_snapshot.txt",
        "experiment_notes": run_dir / "experiment_notes.txt",
    }


def update_experiment_index(full_summary):
    index_path = base.RESULTS_DIR / "experiment_index.csv"

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
        "description": "Candidate model structure export for embedded feasibility",
        "model": "gradient_boosting_depth3_20",
        "validation_method": "Model structure analysis, not performance validation",
        "top_k": base.MODEL_TOP_K,
        "max_depth": base.MODEL_MAX_DEPTH,
        "min_samples_split": "",
        "accuracy": "",
        "macro_f1": "",
        "weighted_f1": "",
        "main_observation": (
            f"Multiclass model contains {full_summary['total_trees']} trees, "
            f"{full_summary['total_nodes']} nodes and about "
            f"{full_summary['comparisons_mean_per_window']:.2f} comparisons per window."
        ),
    }

    existing_rows.append(new_row)

    with open(index_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in existing_rows:
            writer.writerow(row)


def save_embedded_snapshot(output_path, full_summary, used_features_rows):
    lines = []

    lines.append("Embedded model snapshot - Experiment 023")
    lines.append("")
    lines.append("Candidate model:")
    lines.append("- GradientBoostingClassifier")
    lines.append(f"- n_estimators: {base.MODEL_N_ESTIMATORS}")
    lines.append(f"- max_depth: {base.MODEL_MAX_DEPTH}")
    lines.append(f"- top_k: {base.MODEL_TOP_K}")
    lines.append("")
    lines.append("Important structure observation:")
    lines.append(
        "- Because this is a multiclass GradientBoostingClassifier, "
        "the model uses one regression tree per class per boosting stage."
    )
    lines.append(
        f"- Total trees: {full_summary['total_trees']} "
        f"({base.MODEL_N_ESTIMATORS} stages x {full_summary['n_classes']} classes)"
    )
    lines.append("")
    lines.append("Full dataset seed 42 structure:")
    lines.append(f"- Total nodes: {full_summary['total_nodes']}")
    lines.append(f"- Internal decision nodes: {full_summary['internal_decision_nodes']}")
    lines.append(f"- Leaf nodes: {full_summary['leaf_nodes']}")
    lines.append(f"- Max depth: {full_summary['max_depth']}")
    lines.append(f"- Used original features: {full_summary['used_original_features']}")
    lines.append(f"- Mean comparisons per window: {full_summary['comparisons_mean_per_window']:.6f}")
    lines.append(f"- Max comparisons per window: {full_summary['comparisons_max_per_window']}")
    lines.append("")
    lines.append("Top used features:")
    for row in used_features_rows[:15]:
        subcarrier = row["subcarrier_id"]

        if subcarrier == "":
            subcarrier = f"position {row['subcarrier_position']}"

        lines.append(
            f"- feature_index={row['original_feature_index']}, "
            f"subcarrier={subcarrier}, "
            f"feature={row['feature_name']}, "
            f"splits={row['split_count']}"
        )

    lines.append("")
    lines.append("Embedded interpretation:")
    lines.append(
        "- The model is still structurally simple compared with neural networks, "
        "but it is larger than a single decision tree."
    )
    lines.append(
        "- For embedded implementation, it should be converted to fixed arrays "
        "or hardcoded tree traversal logic."
    )
    lines.append(
        "- The final export should be repeated after Dataset v2, because the "
        "current model is still trained on a limited dataset."
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def save_experiment_notes(output_path, holdout_summary, full_summary):
    lines = []

    lines.append("Experiment 023 - Candidate model structure export")
    lines.append("")
    lines.append("Objective:")
    lines.append(
        "Inspect the current candidate model structure to estimate embedded "
        "feasibility before implementing the realtime version."
    )
    lines.append("")
    lines.append("Candidate model:")
    lines.append(f"- Gradient Boosting, n_estimators={base.MODEL_N_ESTIMATORS}")
    lines.append(f"- max_depth={base.MODEL_MAX_DEPTH}")
    lines.append(f"- top_k={base.MODEL_TOP_K}")
    lines.append("")
    lines.append("Key observation:")
    lines.append(
        "Because the problem has three classes, scikit-learn's multiclass "
        "GradientBoostingClassifier uses one regression tree per class per "
        "boosting stage."
    )
    lines.append(
        f"Therefore, the candidate model contains {full_summary['total_trees']} "
        f"trees, not only {base.MODEL_N_ESTIMATORS} trees."
    )
    lines.append("")
    lines.append("Full dataset seed 42 structure:")
    lines.append(f"- Total trees: {full_summary['total_trees']}")
    lines.append(f"- Total nodes: {full_summary['total_nodes']}")
    lines.append(f"- Internal decision nodes: {full_summary['internal_decision_nodes']}")
    lines.append(f"- Leaf nodes: {full_summary['leaf_nodes']}")
    lines.append(f"- Max depth: {full_summary['max_depth']}")
    lines.append(f"- Used original features: {full_summary['used_original_features']}")
    lines.append(f"- Mean comparisons per window: {full_summary['comparisons_mean_per_window']:.6f}")
    lines.append(f"- Max comparisons per window: {full_summary['comparisons_max_per_window']}")
    lines.append("")
    lines.append("Repeated holdout structure summary:")
    lines.append(f"- Seeds: {holdout_summary['seeds']}")
    lines.append(f"- Total trees mean: {holdout_summary['total_trees_mean']:.6f}")
    lines.append(f"- Total nodes mean: {holdout_summary['total_nodes_mean']:.6f}")
    lines.append(f"- Internal decision nodes mean: {holdout_summary['internal_decision_nodes_mean']:.6f}")
    lines.append(f"- Used original features mean: {holdout_summary['used_original_features_mean']:.6f}")
    lines.append(f"- Mean comparisons per window: {holdout_summary['comparisons_mean_per_window_mean']:.6f}")
    lines.append("")
    lines.append("Conclusion:")
    lines.append(
        "The model is feasible to analyze and export, but it is not as small "
        "as the n_estimators value alone suggests. The next implementation "
        "step should convert the trained structure into an embedded-friendly "
        "representation using arrays or generated code."
    )

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


# ================= MAIN =================

def run_experiment():
    feature_dataset = base.load_pickle(base.FEATURE_DATASET_FILE)

    if not feature_dataset:
        raise ValueError("Feature dataset is empty.")

    features_per_sample = len(feature_dataset[0]["features"])

    if features_per_sample != base.EXPECTED_FEATURES_PER_SAMPLE:
        raise ValueError(
            "This experiment must be run with the official 014 feature set. "
            f"Expected {base.EXPECTED_FEATURES_PER_SAMPLE} features per sample, "
            f"but found {features_per_sample}."
        )

    selected_subcarriers = load_selected_subcarriers()

    print()
    print("Candidate Model Structure Export")
    print("=" * 70)
    print("Experiment:", EXPERIMENT_ID)
    print("Feature dataset:", base.FEATURE_DATASET_FILE)
    print("Samples/windows:", len(feature_dataset))
    print("Features per sample:", features_per_sample)
    print("Detected selected subcarriers:", selected_subcarriers)
    print()

    structure_rows = []
    aggregate_used_features = Counter()

    for seed in base.SEEDS:
        print(f"Analyzing holdout seed {seed}...")

        train_dataset, test_dataset, train_groups, test_groups = base.file_stratified_holdout_split(
            feature_dataset,
            base.TEST_SIZE,
            seed,
        )

        selected_train_dataset, selected_indices = base.select_top_k_from_training(
            train_dataset,
            base.MODEL_TOP_K,
        )

        selected_test_dataset = base.select_features_by_indices(
            test_dataset,
            selected_indices,
        )

        x_test, _ = base.dataset_to_xy(selected_test_dataset)

        model = base.train_gradient_boosting(
            selected_train_dataset,
            seed,
        )

        summary, used_counter = analyze_model_structure(
            model=model,
            selected_indices=selected_indices,
            x_values=x_test,
            seed=seed,
            training_mode="file_holdout",
            train_windows=len(selected_train_dataset),
            test_windows=len(selected_test_dataset),
            test_groups=len(test_groups),
        )

        structure_rows.append(summary)
        aggregate_used_features.update(used_counter)

    print()
    print("Training representative full-dataset model...")

    selected_full_dataset, full_selected_indices = base.select_top_k_from_training(
        feature_dataset,
        base.MODEL_TOP_K,
    )

    x_full, _ = base.dataset_to_xy(selected_full_dataset)

    full_model = base.train_gradient_boosting(
        selected_full_dataset,
        FULL_DATASET_EXPORT_SEED,
    )

    full_summary, full_used_counter = analyze_model_structure(
        model=full_model,
        selected_indices=full_selected_indices,
        x_values=x_full,
        seed=FULL_DATASET_EXPORT_SEED,
        training_mode="full_dataset",
        train_windows=len(selected_full_dataset),
        test_windows=0,
        test_groups=0,
    )

    tree_node_rows = export_tree_nodes(
        full_model,
        full_selected_indices,
        selected_subcarriers,
    )

    used_features_holdout_rows = build_used_feature_rows(
        aggregate_used_features,
        selected_subcarriers,
    )

    used_features_full_rows = build_used_feature_rows(
        full_used_counter,
        selected_subcarriers,
    )

    holdout_summary = summarize_holdout_structure(
        structure_rows,
    )

    structure_summary_rows = [
        holdout_summary,
    ]

    paths = get_experiment_paths()

    save_csv(
        structure_rows,
        paths["structure_by_seed"],
    )

    save_csv(
        structure_summary_rows,
        paths["structure_summary"],
    )

    save_csv(
        used_features_holdout_rows,
        paths["used_features_holdout"],
    )

    save_csv(
        used_features_full_rows,
        paths["used_features_full"],
    )

    save_csv(
        tree_node_rows,
        paths["tree_nodes_full"],
    )

    save_embedded_snapshot(
        paths["embedded_snapshot"],
        full_summary,
        used_features_full_rows,
    )

    save_experiment_notes(
        paths["experiment_notes"],
        holdout_summary,
        full_summary,
    )

    update_experiment_index(
        full_summary,
    )

    print()
    print("=" * 70)
    print("Candidate model structure export finished.")
    print("=" * 70)
    print("Full dataset representative model:")
    print(f"total_trees: {full_summary['total_trees']}")
    print(f"total_nodes: {full_summary['total_nodes']}")
    print(f"internal_decision_nodes: {full_summary['internal_decision_nodes']}")
    print(f"leaf_nodes: {full_summary['leaf_nodes']}")
    print(f"max_depth: {full_summary['max_depth']}")
    print(f"used_original_features: {full_summary['used_original_features']}")
    print(f"comparisons_mean_per_window: {full_summary['comparisons_mean_per_window']:.6f}")
    print(f"comparisons_max_per_window: {full_summary['comparisons_max_per_window']}")
    print()
    print("Results saved to:")
    print(paths["run_dir"])


def main():
    run_experiment()


if __name__ == "__main__":
    main()