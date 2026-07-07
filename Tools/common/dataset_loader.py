from pathlib import Path

from Tools.common.project_paths import RAW_BIN_DIR, setup_import_paths


setup_import_paths()


VALID_LABELS = {
    "empty": "empty",
    "static": "static_presence",
    "static_presence": "static_presence",
    "presence": "static_presence",
    "movement": "movement",
}


def normalize_label(label):
    label = label.lower().strip()
    return VALID_LABELS.get(label)


def find_quadrant(path):
    for part in path.parts:
        part_lower = part.lower()

        if part_lower.startswith("quad"):
            return part_lower

    return None


def infer_label_from_path(file_path):
    parent_label = normalize_label(file_path.parent.name)

    if parent_label is not None:
        return parent_label

    file_name = file_path.name.lower()

    for raw_label, normalized_label in VALID_LABELS.items():
        if file_name.startswith(raw_label):
            return normalized_label

    return None


def load_dataset(
    base_path=None,
    labels=None,
    quadrants=None,
):
    base = Path(base_path) if base_path is not None else RAW_BIN_DIR

    if not base.exists():
        raise FileNotFoundError(f"Pasta do dataset não encontrada: {base}")

    if labels is not None:
        labels = set(labels)

    if quadrants is not None:
        quadrants = set(q.lower() for q in quadrants)

    dataset = []

    for file_path in sorted(base.rglob("*.bin")):
        quadrant = find_quadrant(file_path)
        label = infer_label_from_path(file_path)

        if quadrant is None:
            continue

        if label is None:
            continue

        if labels is not None and label not in labels:
            continue

        if quadrants is not None and quadrant not in quadrants:
            continue

        dataset.append({
            "path": str(file_path),
            "label": label,
            "quadrant": quadrant,
            "file_name": file_path.name,
            "size_bytes": file_path.stat().st_size,
        })

    return dataset


def count_by_label(dataset):
    counts = {}

    for item in dataset:
        label = item["label"]
        counts[label] = counts.get(label, 0) + 1

    return counts


def count_by_quadrant_and_label(dataset):
    counts = {}

    for item in dataset:
        key = (item["quadrant"], item["label"])
        counts[key] = counts.get(key, 0) + 1

    return counts


def print_dataset_summary(dataset):
    print()
    print("Resumo do dataset")
    print("Total de arquivos:", len(dataset))

    print()
    print("Arquivos por classe:")

    for label, count in sorted(count_by_label(dataset).items()):
        print(label, ":", count)

    print()
    print("Arquivos por quadrante/classe:")

    for (quadrant, label), count in sorted(
        count_by_quadrant_and_label(dataset).items()
    ):
        print(quadrant, "|", label, ":", count)


if __name__ == "__main__":
    dataset = load_dataset()
    print_dataset_summary(dataset)