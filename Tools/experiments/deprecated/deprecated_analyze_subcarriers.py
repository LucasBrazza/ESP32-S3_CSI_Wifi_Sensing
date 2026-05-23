import sys
import csv
import math
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent.parent

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from Tools.csi.csi_binary_io import read_packets


# ================= PATHS =================

RAW_BIN_DIR = TOOLS_DIR / "datasets" / "bin" / "raw"
OUTPUT_DIR = TOOLS_DIR / "datasets" / "csv" / "subcarrier_analysis"

OUTPUT_RANKING_FILE = OUTPUT_DIR / "subcarrier_ranking.csv"
OUTPUT_BLOCK_FILE = OUTPUT_DIR / "block_summary.csv"


# ================= CONFIG =================

EPSILON = 0.000001

EXPECTED_LABELS = [
    "empty",
    "static_presence",
    "movement",
]


# ================= BASIC MATH =================

def mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def variance(values):
    if not values:
        return 0.0

    avg = mean(values)

    return sum(
        (value - avg) ** 2
        for value in values
    ) / len(values)


def std(values):
    return math.sqrt(variance(values))


def amplitude(real, imag):
    return math.sqrt(
        real * real + imag * imag
    )


# ================= BLOCK STRATEGY =================

def get_block_ranges(num_subcarriers):
    """
    Splits CSI vector into 4 blocks.

    Example:
    192 subcarriers:
        block 0: 0..47
        block 1: 48..95
        block 2: 96..143
        block 3: 144..191
    """

    block_size = num_subcarriers // 4

    blocks = []

    for block_id in range(4):
        start = block_id * block_size

        if block_id == 3:
            end = num_subcarriers
        else:
            end = start + block_size

        blocks.append(
            {
                "block_id": block_id,
                "start": start,
                "end": end,
            }
        )

    return blocks


def get_block_id(subcarrier, num_subcarriers):
    block_size = num_subcarriers // 4

    if block_size <= 0:
        return 0

    block_id = subcarrier // block_size

    if block_id > 3:
        block_id = 3

    return block_id


# ================= DATA LOADING =================

def detect_label_from_filename(file_path):
    name = file_path.name.lower()

    if "static_presence" in name:
        return "static_presence"

    if "movement" in name:
        return "movement"

    if "empty" in name:
        return "empty"

    return "unknown"


def load_amplitudes_by_label():
    """
    Returns:
    {
        label: {
            subcarrier_index: [amplitude_values]
        }
    }
    """

    data = {}

    for label in EXPECTED_LABELS:
        data[label] = {}

    bin_files = list(RAW_BIN_DIR.glob("*.bin"))

    if not bin_files:
        print(f"No .bin files found in: {RAW_BIN_DIR}")
        return data, 0

    max_subcarriers = 0

    for file_path in bin_files:
        packets = read_packets(file_path)

        if not packets:
            continue

        label = packets[0].get(
            "label",
            detect_label_from_filename(file_path),
        )

        if label not in data:
            data[label] = {}

        print(
            f"Reading {file_path.name} | "
            f"label={label} | packets={len(packets)}"
        )

        for packet in packets:
            imag = packet["imag"]
            real = packet["real"]

            n = min(
                len(imag),
                len(real),
            )

            if n > max_subcarriers:
                max_subcarriers = n

            for subcarrier in range(n):
                amp = amplitude(
                    real[subcarrier],
                    imag[subcarrier],
                )

                if subcarrier not in data[label]:
                    data[label][subcarrier] = []

                data[label][subcarrier].append(amp)

    return data, max_subcarriers


# ================= ANALYSIS =================

def compute_subcarrier_rows(data, max_subcarriers):
    rows = []

    for subcarrier in range(max_subcarriers):
        empty_values = data.get(
            "empty",
            {},
        ).get(
            subcarrier,
            [],
        )

        static_values = data.get(
            "static_presence",
            {},
        ).get(
            subcarrier,
            [],
        )

        movement_values = data.get(
            "movement",
            {},
        ).get(
            subcarrier,
            [],
        )

        mean_empty = mean(empty_values)
        std_empty = std(empty_values)

        mean_static = mean(static_values)
        std_static = std(static_values)

        mean_movement = mean(movement_values)
        std_movement = std(movement_values)

        movement_score = std_movement / (
            std_empty + EPSILON
        )

        static_score = abs(
            mean_static - mean_empty
        ) / (
            std_empty + EPSILON
        )

        final_score = movement_score + static_score

        block_id = get_block_id(
            subcarrier,
            max_subcarriers,
        )

        rows.append(
            {
                "subcarrier": subcarrier,
                "block_id": block_id,
                "mean_empty": mean_empty,
                "std_empty": std_empty,
                "mean_static": mean_static,
                "std_static": std_static,
                "mean_movement": mean_movement,
                "std_movement": std_movement,
                "movement_score": movement_score,
                "static_score": static_score,
                "final_score": final_score,
                "empty_samples": len(empty_values),
                "static_samples": len(static_values),
                "movement_samples": len(movement_values),
            }
        )

    rows.sort(
        key=lambda item: item["final_score"],
        reverse=True,
    )

    return rows


def compute_block_summary(rows):
    block_data = {}

    for block_id in range(4):
        block_data[block_id] = []

    for row in rows:
        block_id = int(row["block_id"])

        if block_id not in block_data:
            block_data[block_id] = []

        block_data[block_id].append(row)

    summary_rows = []

    for block_id in sorted(block_data.keys()):
        block_rows = block_data[block_id]

        final_scores = [
            row["final_score"]
            for row in block_rows
        ]

        movement_scores = [
            row["movement_score"]
            for row in block_rows
        ]

        static_scores = [
            row["static_score"]
            for row in block_rows
        ]

        summary_rows.append(
            {
                "block_id": block_id,
                "num_subcarriers": len(block_rows),
                "mean_final_score": mean(final_scores),
                "max_final_score": max(final_scores) if final_scores else 0.0,
                "mean_movement_score": mean(movement_scores),
                "mean_static_score": mean(static_scores),
            }
        )

    summary_rows.sort(
        key=lambda item: item["mean_final_score"],
        reverse=True,
    )

    return summary_rows


# ================= OUTPUT =================

def save_csv(file_path, rows):
    if not rows:
        print(f"No rows to save: {file_path}")
        return

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {file_path}")


def print_top_rows(rows, limit=20):
    print()
    print("Top useful subcarriers:")
    print("-----------------------")

    for row in rows[:limit]:
        print(
            f"SC={row['subcarrier']:03d} | "
            f"Block={row['block_id']} | "
            f"Final={row['final_score']:.3f} | "
            f"Move={row['movement_score']:.3f} | "
            f"Static={row['static_score']:.3f}"
        )


def print_block_summary(summary_rows):
    print()
    print("Block summary:")
    print("--------------")

    for row in summary_rows:
        print(
            f"Block={row['block_id']} | "
            f"N={row['num_subcarriers']} | "
            f"Mean final={row['mean_final_score']:.3f} | "
            f"Max final={row['max_final_score']:.3f}"
        )


# ================= MAIN =================

def main():
    data, max_subcarriers = load_amplitudes_by_label()

    if max_subcarriers == 0:
        print("No CSI data loaded.")
        return

    print()
    print(f"Detected max subcarriers: {max_subcarriers}")

    blocks = get_block_ranges(max_subcarriers)

    print()
    print("Using 4 blocks:")
    for block in blocks:
        print(
            f"Block {block['block_id']}: "
            f"{block['start']}..{block['end'] - 1}"
        )

    rows = compute_subcarrier_rows(
        data,
        max_subcarriers,
    )

    summary_rows = compute_block_summary(rows)

    save_csv(
        OUTPUT_RANKING_FILE,
        rows,
    )

    save_csv(
        OUTPUT_BLOCK_FILE,
        summary_rows,
    )

    print_top_rows(rows)
    print_block_summary(summary_rows)


if __name__ == "__main__":
    main()