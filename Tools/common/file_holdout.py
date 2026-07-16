"""Utilities for file-based dataset splitting and feature selection.

These helpers keep all windows from the same acquisition file in the same
partition. This avoids information leakage between training and test sets.
"""

import random
from collections import Counter


def get_sample_group_id(sample):
    source_file = sample.get("source_file", "")

    if source_file:
        return str(source_file)

    label = sample.get("label", "unknown_label")
    quadrant = sample.get("quadrant", "unknown_quadrant")
    file_name = sample.get("file_name", "unknown_file")

    return f"{label}|{quadrant}|{file_name}"


def group_samples_by_file(dataset):
    groups = {}

    for sample in dataset:
        group_id = get_sample_group_id(sample)
        groups.setdefault(group_id, []).append(sample)

    grouped_items = []

    for group_id, samples in groups.items():
        labels = sorted({sample["label"] for sample in samples})

        if len(labels) != 1:
            raise ValueError(
                "A file group contains more than one label. "
                f"group_id={group_id}, labels={labels}"
            )

        representative = samples[0]

        grouped_items.append(
            {
                "group_id": group_id,
                "label": labels[0],
                "quadrant": representative.get("quadrant", "unknown"),
                "file_name": representative.get("file_name", ""),
                "source_file": representative.get("source_file", ""),
                "samples": samples,
                "sample_count": len(samples),
            }
        )

    return grouped_items


def file_stratified_holdout_split(dataset, test_size=0.20, seed=42):
    """Split acquisition-file groups while preserving every class in train/test."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    random_generator = random.Random(seed)
    groups = group_samples_by_file(dataset)
    groups_by_label = {}

    for group in groups:
        groups_by_label.setdefault(group["label"], []).append(group)

    train_groups = []
    test_groups = []

    for label, label_groups_original in groups_by_label.items():
        label_groups = label_groups_original[:]
        random_generator.shuffle(label_groups)

        if len(label_groups) < 2:
            raise ValueError(
                "File-based holdout requires at least two acquisition files "
                f"for class '{label}'."
            )

        test_group_count = int(round(len(label_groups) * test_size))
        test_group_count = max(1, test_group_count)
        test_group_count = min(test_group_count, len(label_groups) - 1)

        test_groups.extend(label_groups[:test_group_count])
        train_groups.extend(label_groups[test_group_count:])

    random_generator.shuffle(train_groups)
    random_generator.shuffle(test_groups)

    train_dataset = [
        sample
        for group in train_groups
        for sample in group["samples"]
    ]
    test_dataset = [
        sample
        for group in test_groups
        for sample in group["samples"]
    ]

    random_generator.shuffle(train_dataset)
    random_generator.shuffle(test_dataset)

    return train_dataset, test_dataset, train_groups, test_groups


def select_features_by_indices(feature_dataset, selected_indices):
    selected_dataset = []

    for sample in feature_dataset:
        selected_sample = {
            key: value
            for key, value in sample.items()
            if key != "features"
        }
        selected_sample["features"] = [
            sample["features"][index]
            for index in selected_indices
        ]
        selected_dataset.append(selected_sample)

    return selected_dataset


def dataset_to_xy(dataset):
    x_values = [sample["features"] for sample in dataset]
    y_values = [sample["label"] for sample in dataset]
    return x_values, y_values


def count_by_label(dataset):
    return Counter(sample["label"] for sample in dataset)


def count_groups_by_label(groups):
    return Counter(group["label"] for group in groups)
