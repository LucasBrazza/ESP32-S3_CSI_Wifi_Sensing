import argparse
import csv
import math
import serial
from datetime import datetime
import time
import os


def parse_csi_line(raw_line):
    parts = raw_line.split(",")

    if len(parts) < 7:
        return None

    if parts[0] != "CSI":
        return None

    try:
        esp_timestamp_us = int(parts[1])
        rssi = int(parts[2])
        rate = int(parts[3])
        channel = int(parts[4])
        csi_len = int(parts[5])
        csi_data = [int(x) for x in parts[6:]]
    except ValueError:
        return None

    if len(csi_data) != csi_len:
        return None

    if len(csi_data) % 2 != 0:
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
    parser = argparse.ArgumentParser(description="Parsed Serial CSI data logger")

    parser.add_argument("--port", required=True, help="Serial port, example: COM4")
    parser.add_argument("--baud", default=115200, type=int, help="Serial baud rate")
    parser.add_argument("--label", required=True, help="Data label: empty, static_presence, activity")
    parser.add_argument("--output", default=None, help="Output CSV file")
    parser.add_argument("--duration", type=int, default=60, help="Collection duration in seconds")
    

    args = parser.parse_args()

    if args.output is None:
        os.makedirs("datasets", exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        args.output = os.path.join(
            "datasets",
            f"{args.label}_{timestamp}_{args.duration}s.csv"
        )

    ser = serial.Serial(args.port, args.baud, timeout=1)
    
    start_time = time.time()

    print("Parsed CSI logger started")
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
                
                if time.time() - start_time >= args.duration:
                    print("\nCollection finished by timer.")
                    break
                
                raw_line = ser.readline().decode("utf-8", errors="ignore").strip()

                if not raw_line:
                    continue

                if not raw_line.startswith("CSI,"):
                    continue

                pc_timestamp = datetime.now().isoformat(timespec="milliseconds")
                parsed = parse_csi_line(raw_line)

                if parsed is None:
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
    main()