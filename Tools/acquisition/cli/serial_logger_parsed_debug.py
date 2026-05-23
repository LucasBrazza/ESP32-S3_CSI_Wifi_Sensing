import argparse
import csv
import math
import serial
from datetime import datetime


def parse_csi_line(raw_line):
    parts = raw_line.split(",")

    if len(parts) < 7:
        print("REJEITADA: poucos campos")
        print(raw_line[:200])
        return None

    if parts[0] != "CSI":
        print("REJEITADA: não começa com CSI")
        print(raw_line[:200])
        return None

    try:
        esp_timestamp_us = int(parts[1])
        rssi = int(parts[2])
        rate = int(parts[3])
        channel = int(parts[4])
        csi_len = int(parts[5])
        csi_data = [int(x) for x in parts[6:]]
    except ValueError as e:
        print("REJEITADA: erro ao converter número")
        print("Erro:", e)
        print(raw_line[:200])
        return None

    if len(csi_data) != csi_len:
        print("REJEITADA: tamanho CSI incompatível")
        print(f"Esperado pelo len: {csi_len}")
        print(f"Recebido no buffer: {len(csi_data)}")
        print(raw_line[:200])
        return None

    if len(csi_data) % 2 != 0:
        print("AVISO: CSI com tamanho ímpar, removendo último valor")
        csi_data = csi_data[:-1]

    subcarriers = []

    for i in range(0, len(csi_data), 2):
        imag = csi_data[i]
        real = csi_data[i + 1]

        amplitude = math.sqrt(real ** 2 + imag ** 2)
        phase = math.atan2(imag, real)

        subcarriers.append({
            "subcarrier": i // 2,
            "imag": imag,
            "real": real,
            "amplitude": amplitude,
            "phase": phase,
        })

    return {
        "esp_timestamp_us": esp_timestamp_us,
        "rssi": rssi,
        "rate": rate,
        "channel": channel,
        "len": csi_len,
        "subcarriers": subcarriers,
    }


def main():
    parser = argparse.ArgumentParser(description="Parsed Serial CSI data logger - DEBUG")

    parser.add_argument("--port", required=True, help="Serial port, example: COM4")
    parser.add_argument("--baud", default=115200, type=int, help="Serial baud rate")
    parser.add_argument("--label", required=True, help="Data label: empty, static_presence, activity")
    parser.add_argument("--output", required=True, help="Output CSV file")

    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)

    print("Parsed CSI logger DEBUG started")
    print(f"Port: {args.port}")
    print(f"Baud: {args.baud}")
    print(f"Label: {args.label}")
    print(f"Output: {args.output}")
    print("Press Ctrl+C to stop\n")

    with open(args.output, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if file.tell() == 0:
            writer.writerow([
                "label",
                "pc_timestamp",
                "esp_timestamp_us",
                "rssi",
                "rate",
                "channel",
                "len",
                "subcarrier",
                "imag",
                "real",
                "amplitude",
                "phase",
            ])

        try:
            while True:
                raw_line = ser.readline().decode("utf-8", errors="ignore").strip()

                if not raw_line:
                    continue

                print("RAW:", raw_line[:200])

                if not raw_line.startswith("CSI,"):
                    continue

                pc_timestamp = datetime.now().isoformat(timespec="milliseconds")
                parsed = parse_csi_line(raw_line)

                if parsed is None:
                    print("LINHA CSI REJEITADA\n")
                    continue

                for sc in parsed["subcarriers"]:
                    writer.writerow([
                        args.label,
                        pc_timestamp,
                        parsed["esp_timestamp_us"],
                        parsed["rssi"],
                        parsed["rate"],
                        parsed["channel"],
                        parsed["len"],
                        sc["subcarrier"],
                        sc["imag"],
                        sc["real"],
                        f"{sc['amplitude']:.6f}",
                        f"{sc['phase']:.6f}",
                    ])

                file.flush()

                print(
                    "SALVO:",
                    args.label,
                    pc_timestamp,
                    "RSSI:",
                    parsed["rssi"],
                    "Subcarriers:",
                    len(parsed["subcarriers"])
                )

        except KeyboardInterrupt:
            print("\nLogger stopped by user.")

        finally:
            ser.close()


if __name__ == "__main__":
    print("Arquivo sendo executado")
    main()