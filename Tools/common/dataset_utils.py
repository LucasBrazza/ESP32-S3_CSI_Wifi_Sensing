def count_by_label(dataset):
    counts = {}

    for item in dataset:
        label = item["label"]
        counts[label] = counts.get(label, 0) + 1

    return counts


def count_by_quadrant(dataset):
    counts = {}

    for item in dataset:
        quadrant = item.get("quadrant", "unknown")
        counts[quadrant] = counts.get(quadrant, 0) + 1

    return counts


def count_by_quadrant_and_label(dataset):
    counts = {}

    for item in dataset:
        quadrant = item.get("quadrant", "unknown")
        label = item["label"]

        key = (quadrant, label)
        counts[key] = counts.get(key, 0) + 1

    return counts


def print_class_distribution(dataset):
    print()
    print("Amostras por classe:")

    for label, count in sorted(count_by_label(dataset).items()):
        print(label, ":", count)


def print_quadrant_distribution(dataset):
    print()
    print("Amostras por quadrante:")

    for quadrant, count in sorted(count_by_quadrant(dataset).items()):
        print(quadrant, ":", count)


def print_quadrant_label_distribution(dataset):
    print()
    print("Amostras por quadrante/classe:")

    for (quadrant, label), count in sorted(
        count_by_quadrant_and_label(dataset).items()
    ):
        print(quadrant, "|", label, ":", count)