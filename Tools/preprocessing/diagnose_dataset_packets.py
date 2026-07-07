from Tools.common.dataset_loader import load_dataset

from Tools.preprocessing.csi_pipeline_core import (
    load_bin_file,
    packets_to_amplitude_matrix,
    remove_invalid_packets,
)


def main():
    dataset = load_dataset()

    summary = {}

    for item in dataset:
        path = item["path"]
        label = item["label"]
        quadrant = item["quadrant"]

        packets = load_bin_file(path)
        amplitudes = packets_to_amplitude_matrix(packets)
        clean = remove_invalid_packets(amplitudes)

        key = (quadrant, label)

        if key not in summary:
            summary[key] = {
                "files": 0,
                "raw_packets": 0,
                "amp_packets": 0,
                "valid_packets": 0,
                "files_with_windows": 0,
            }

        summary[key]["files"] += 1
        summary[key]["raw_packets"] += len(packets)
        summary[key]["amp_packets"] += len(amplitudes)
        summary[key]["valid_packets"] += len(clean)

        if len(clean) >= 20:
            summary[key]["files_with_windows"] += 1

    print()
    print("Diagnóstico por quadrante/classe")

    for (quadrant, label), data in sorted(summary.items()):
        print()
        print(quadrant, "|", label)
        print("Arquivos:", data["files"])
        print("Pacotes brutos:", data["raw_packets"])
        print("Pacotes amplitude:", data["amp_packets"])
        print("Pacotes válidos:", data["valid_packets"])
        print("Arquivos com >=20 pacotes válidos:", data["files_with_windows"])


if __name__ == "__main__":
    main()