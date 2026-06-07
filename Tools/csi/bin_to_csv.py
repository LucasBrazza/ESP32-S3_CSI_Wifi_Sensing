import sys
import csv
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from csi.csi_binary_io import read_packets


def convert_bin_to_csv(input_file: Path, output_file: Path):
    packets = read_packets(input_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "label",
        "pc_timestamp",
        "packet_index",
        "subcarrier",
        "imag",
        "real",
        "rssi",
        "rate",
        "channel",
        "csi_len",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for packet_index, packet in enumerate(packets):
            imag = packet["imag"]
            real = packet["real"]

            n = min(len(imag), len(real))

            for subcarrier in range(n):
                writer.writerow(
                    {
                        "label": packet["label"],
                        "pc_timestamp": packet["pc_timestamp"],
                        "packet_index": packet_index,
                        "subcarrier": subcarrier,
                        "imag": imag[subcarrier],
                        "real": real[subcarrier],
                        "rssi": packet["rssi"],
                        "rate": packet["rate"],
                        "channel": packet["channel"],
                        "csi_len": packet["csi_len"],
                    }
                )

    print(f"Converted: {input_file}")
    print(f"Saved CSV: {output_file}")
    print(f"Packets: {len(packets)}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python bin_to_csv.py <input_bin_file> [output_csv_file]")
        return

    input_file = Path(sys.argv[1])

    if len(sys.argv) >= 3:
        output_file = Path(sys.argv[2])
    else:
        output_file = input_file.with_suffix(".csv")

    convert_bin_to_csv(input_file, output_file)


if __name__ == "__main__":
    main()