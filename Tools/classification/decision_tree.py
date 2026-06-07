from pathlib import Path
import sys


# ================= PATHS =================

TOOLS_DIR = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = TOOLS_DIR / "preprocessing"

if str(PREPROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESSING_DIR))


# ================= BASIC UTILS =================

def count_labels(dataset):
    counts = {}

    for item in dataset:
        label = item["label"]

        if label not in counts:
            counts[label] = 0

        counts[label] += 1

    return counts


def majority_label(dataset):
    counts = count_labels(dataset)

    best_label = None
    best_count = -1

    for label, count in counts.items():
        if count > best_count:
            best_label = label
            best_count = count

    return best_label


def all_same_label(dataset):
    if not dataset:
        return True

    first_label = dataset[0]["label"]

    for item in dataset:
        if item["label"] != first_label:
            return False

    return True


def gini_impurity(dataset):
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

def get_candidate_thresholds(dataset, feature_index):
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

def build_tree(dataset, max_depth=4, min_samples_split=2, depth=0):
    """
    Cria uma árvore de decisão simples usando Gini.

    dataset:
        [
            {
                "label": "empty",
                "features": [...]
            }
        ]
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

def predict_one(tree, features):
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

def accuracy(predictions):
    if not predictions:
        return 0.0

    correct = 0

    for item in predictions:
        if item["true"] == item["predicted"]:
            correct += 1

    return correct / len(predictions)


def confusion_matrix(predictions):
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

def leave_one_out_cross_validation(
    dataset,
    max_depth=4,
    min_samples_split=2,
):
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

def print_tree(tree, indent=""):
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