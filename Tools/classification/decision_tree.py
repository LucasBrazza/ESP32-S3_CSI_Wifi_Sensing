"""
Simple Decision Tree Classifier

This module implements a lightweight decision tree classifier using
Gini impurity as the split criterion.

The implementation is intentionally written without scikit-learn or
other machine learning libraries. This keeps the model transparent,
easy to inspect, and easier to port to embedded environments such as
MicroPython or C.

Input format:

    dataset = [
        {
            "label": class_name,
            "features": [...]
        }
    ]

The decision tree learns rules of the form:

    if feature[index] <= threshold:
        go left
    else:
        go right

The final model is stored as nested dictionaries, which makes it easy
to save as JSON and reuse during real-time inference.

Pipeline position:

    selected features
    ↓
    decision tree training
    ↓
    prediction
    ↓
    evaluation
"""

from pathlib import Path
import sys


# ================= PATHS =================

TOOLS_DIR = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = TOOLS_DIR / "preprocessing"

if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))


# ================= BASIC UTILS =================
"""
Basic dataset utilities.

These functions count class labels and identify simple stopping
conditions used during tree construction.
"""


def count_labels(dataset):
    """
    Count how many samples belong to each class.

    Output example:

        {
            "empty": 8,
            "static_presence": 7,
            "movement": 7
        }

    This is used to compute impurity and to determine the majority
    class in a node.
    """
    counts = {}

    for item in dataset:
        label = item["label"]

        if label not in counts:
            counts[label] = 0

        counts[label] += 1

    return counts


def majority_label(dataset):
    """
    Return the most frequent class label in a dataset.

    If a node cannot be split further, the tree predicts the majority
    class among the samples that reached that node.
    """
    counts = count_labels(dataset)

    best_label = None
    best_count = -1

    for label, count in counts.items():
        if count > best_count:
            best_label = label
            best_count = count

    return best_label


def all_same_label(dataset):
    """
    Check whether all samples in a dataset belong to the same class.

    If all labels are equal, the node is already pure and becomes a
    leaf node.
    """
    if not dataset:
        return True

    first_label = dataset[0]["label"]

    for item in dataset:
        if item["label"] != first_label:
            return False

    return True


def gini_impurity(dataset):
    """
    Compute the Gini impurity of a dataset.

    Formula:

        Gini = 1 - Σ p_i²

    where:

        p_i = proportion of samples belonging to class i

    Interpretation:

        Gini = 0
            all samples belong to the same class

        higher Gini
            samples are mixed across different classes

    The decision tree tries to find splits that reduce Gini impurity.
    """
    if not dataset:
        return 0.0

    counts = count_labels(dataset)
    total = len(dataset)

    impurity = 1.0

    for label in counts:
        probability = counts[label] / total
        impurity -= probability * probability

    return impurity


def split_dataset(dataset, feature_index, threshold):
    """
    Split a dataset into two groups using one feature and one threshold.

    Rule:

        if feature[feature_index] <= threshold:
            sample goes to the left branch
        else:
            sample goes to the right branch

    This is the basic decision rule used by every internal node of the
    tree.
    """
    left = []
    right = []

    for item in dataset:
        value = item["features"][feature_index]

        if value <= threshold:
            left.append(item)
        else:
            right.append(item)

    return left, right


def weighted_gini(left, right):
    """
    Compute the weighted Gini impurity after a split.

    Formula:

        weighted_gini =
            (N_left / N_total)  × Gini_left
            +
            (N_right / N_total) × Gini_right

    where:

        N_left  = number of samples in the left branch
        N_right = number of samples in the right branch

    The best split is the one that produces the lowest weighted Gini.
    """
    total = len(left) + len(right)

    if total == 0:
        return 0.0

    left_weight = len(left) / total
    right_weight = len(right) / total

    return (
        left_weight * gini_impurity(left)
        + right_weight * gini_impurity(right)
    )


# ================= BEST SPLIT =================

"""
Split search utilities.

These functions test possible thresholds for each feature and select
the split that produces the lowest weighted Gini impurity.
"""

def get_candidate_thresholds(dataset, feature_index):
    """
    Generate candidate thresholds for a feature.

    The values of the selected feature are sorted, and thresholds are
    placed halfway between consecutive unique values.

    Formula:

        threshold_i = (value_i + value_(i+1)) / 2

    This avoids testing thresholds that would produce identical splits.
    """
    values = []

    for item in dataset:
        values.append(item["features"][feature_index])

    values = sorted(set(values))

    thresholds = []

    for i in range(len(values) - 1):
        threshold = (values[i] + values[i + 1]) / 2.0
        thresholds.append(threshold)

    return thresholds


def find_best_split(dataset):
    """
    Search for the best decision rule for the current node.

    The function tests:

        every feature
        every candidate threshold

    and selects the pair that minimizes weighted Gini impurity.

    Output:

        {
            "feature_index": best_feature,
            "threshold": best_threshold,
            "gini": best_score,
            "left": left_subset,
            "right": right_subset
        }

    If no valid split is found, the function returns None.
    """
    if not dataset:
        return None

    num_features = len(dataset[0]["features"])

    best_feature = None
    best_threshold = None
    best_score = None
    best_left = None
    best_right = None

    for feature_index in range(num_features):
        thresholds = get_candidate_thresholds(
            dataset,
            feature_index,
        )

        for threshold in thresholds:
            left, right = split_dataset(
                dataset,
                feature_index,
                threshold,
            )

            if not left or not right:
                continue

            score = weighted_gini(left, right)

            if best_score is None or score < best_score:
                best_feature = feature_index
                best_threshold = threshold
                best_score = score
                best_left = left
                best_right = right

    if best_feature is None:
        return None

    return {
        "feature_index": best_feature,
        "threshold": best_threshold,
        "gini": best_score,
        "left": best_left,
        "right": best_right,
    }


# ================= TREE BUILD =================

"""
Recursive tree construction.

The tree is built by repeatedly selecting the best split until a
stopping condition is reached.
"""

def build_tree(dataset, max_depth=4, min_samples_split=2, depth=0):
    """
    Build a decision tree recursively.

    At each node, the algorithm tries to find the feature and threshold
    that best separate the classes.

    A node becomes a leaf when:

        - the dataset is empty
        - all samples have the same label
        - the maximum depth is reached
        - the number of samples is below min_samples_split
        - no valid split can be found

    The returned tree is a nested dictionary structure that can be saved
    directly in JSON format.
    """

    node = {
        "depth": depth,
        "samples": len(dataset),
        "majority_label": majority_label(dataset),
    }

    if not dataset:
        node["type"] = "leaf"
        node["label"] = None
        return node

    if all_same_label(dataset):
        node["type"] = "leaf"
        node["label"] = dataset[0]["label"]
        return node

    if depth >= max_depth:
        node["type"] = "leaf"
        node["label"] = majority_label(dataset)
        return node

    if len(dataset) < min_samples_split:
        node["type"] = "leaf"
        node["label"] = majority_label(dataset)
        return node

    split = find_best_split(dataset)

    if split is None:
        node["type"] = "leaf"
        node["label"] = majority_label(dataset)
        return node

    node["type"] = "decision"
    node["feature_index"] = split["feature_index"]
    node["threshold"] = split["threshold"]
    node["gini"] = split["gini"]

    node["left"] = build_tree(
        split["left"],
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        depth=depth + 1,
    )

    node["right"] = build_tree(
        split["right"],
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        depth=depth + 1,
    )

    return node


# ================= PREDICTION =================

"""
Prediction utilities.

A trained tree is traversed from the root node to a leaf node using the
same threshold rules learned during training.
"""


def predict_one(tree, features):
    """
    Predict the class of a single feature vector.

    The function starts at the root node and follows the decision rules:

        if feature[index] <= threshold:
            go left
        else:
            go right

    When a leaf node is reached, its label is returned as the predicted
    class.
    """
    node = tree

    while node["type"] != "leaf":
        feature_index = node["feature_index"]
        threshold = node["threshold"]

        if features[feature_index] <= threshold:
            node = node["left"]
        else:
            node = node["right"]

    return node["label"]


def predict_dataset(tree, dataset):
    """
    Predict the class of every sample in a dataset.

    Output format:

        {
            "true": true_label,
            "predicted": predicted_label
        }

    This format is used by the evaluation functions.
    """
    predictions = []

    for item in dataset:
        predicted = predict_one(
            tree,
            item["features"],
        )

        predictions.append(
            {
                "true": item["label"],
                "predicted": predicted,
            }
        )

    return predictions


# ================= EVALUATION =================

"""
Evaluation utilities.

These functions measure how well the decision tree predictions match
the expected labels.
"""


def accuracy(predictions):
    """
    Compute classification accuracy.

    Formula:

        accuracy = correct_predictions / total_predictions

    Accuracy measures the fraction of samples correctly classified.
    """
    if not predictions:
        return 0.0

    correct = 0

    for item in predictions:
        if item["true"] == item["predicted"]:
            correct += 1

    return correct / len(predictions)


def confusion_matrix(predictions):
    """
    Build a confusion matrix from prediction results.

    The matrix counts how many times each true class was classified as
    each predicted class.

    Rows represent true labels.
    Columns represent predicted labels.

    This helps identify which classes are being confused by the model.
    """
    labels = []

    for item in predictions:
        true_label = item["true"]
        predicted_label = item["predicted"]

        if true_label not in labels:
            labels.append(true_label)

        if predicted_label not in labels:
            labels.append(predicted_label)

    matrix = {}

    for true_label in labels:
        matrix[true_label] = {}

        for predicted_label in labels:
            matrix[true_label][predicted_label] = 0

    for item in predictions:
        matrix[item["true"]][item["predicted"]] += 1

    return labels, matrix


def print_confusion_matrix(labels, matrix):
    """
    Print the confusion matrix in a readable table format.

    This function is intended for diagnostics and experiment analysis.
    """
    print()
    print("Matriz de confusão:")

    print("true\\pred", end="")

    for label in labels:
        print("\t", label, end="")

    print()

    for true_label in labels:
        print(true_label, end="")

        for predicted_label in labels:
            print("\t", matrix[true_label][predicted_label], end="")

        print()


# ================= LOOCV =================

"""
Leave-One-Out Cross Validation.

LOOCV is useful when the dataset is small. Each sample is tested once
while all remaining samples are used for training.
"""

def leave_one_out_cross_validation(
    dataset,
    max_depth=4,
    min_samples_split=2,
):
    """
    Evaluate the model using Leave-One-Out Cross Validation.

    For each sample:

        1. Remove one sample from the dataset
        2. Train the tree using all remaining samples
        3. Predict the removed sample
        4. Store the result

    Formula for number of training runs:

        runs = N

    where:

        N = number of samples in the dataset

    This validation strategy is useful for small datasets because every
    sample is used for testing exactly once.
    """
    
    predictions = []

    for test_index in range(len(dataset)):
        train_dataset = []

        for i in range(len(dataset)):
            if i != test_index:
                train_dataset.append(dataset[i])

        test_item = dataset[test_index]

        tree = build_tree(
            train_dataset,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )

        predicted = predict_one(
            tree,
            test_item["features"],
        )

        predictions.append(
            {
                "true": test_item["label"],
                "predicted": predicted,
            }
        )

    return predictions


# ================= PRINT TREE =================

"""
Tree visualization utility.

This section prints the learned decision rules in a human-readable
format.
"""

def print_tree(tree, indent=""):
    """
    Print the decision tree structure.

    Example output:

        if feature[0] <= 0.20:
            if feature[1] <= 0.08:
                Leaf -> static_presence
            else:
                Leaf -> movement
        else:
            Leaf -> empty

    This is useful for interpreting the model and checking whether the
    learned decision rules are simple enough for embedded deployment.
    """
    if tree["type"] == "leaf":
        print(
            indent
            + "Leaf -> "
            + str(tree["label"])
            + " | samples="
            + str(tree["samples"])
        )
        return

    print(
        indent
        + "if feature["
        + str(tree["feature_index"])
        + "] <= "
        + str(tree["threshold"])
        + ":"
    )

    print_tree(
        tree["left"],
        indent + "    ",
    )

    print(indent + "else:")

    print_tree(
        tree["right"],
        indent + "    ",
    )


# ================= TEST =================

if __name__ == "__main__":
    dataset = [
        {"label": "empty", "features": [0.1, 0.2]},
        {"label": "empty", "features": [0.2, 0.1]},
        {"label": "movement", "features": [1.0, 0.9]},
        {"label": "movement", "features": [1.1, 1.0]},
    ]

    tree = build_tree(
        dataset,
        max_depth=2,
    )

    print("Árvore treinada:")
    print_tree(tree)

    predictions = predict_dataset(
        tree,
        dataset,
    )

    print()
    print("Acurácia treino:", accuracy(predictions))

    labels, matrix = confusion_matrix(predictions)
    print_confusion_matrix(labels, matrix)