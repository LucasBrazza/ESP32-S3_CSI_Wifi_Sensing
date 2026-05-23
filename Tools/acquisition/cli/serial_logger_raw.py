import argparse
import csv
import serial
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Serial CSI data logger")

    parser.add_argument("--port", required=True, help="Serial port, example: COM4")
    parser.add_argument("--baud", default=115200, type=int, help="Serial baud rate")
    parser.add_argument("--label", required=True, help="Data label: empty, static_presence, activity")
    parser.add_argument("--output", required=True, help="Output CSV file")

    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)

    print("Serial logger started")
    print(f"Port: {args.port}")
    print(f"Baud: {args.baud}")
    print(f"Label: {args.label}")
    print(f"Output: {args.output}")
    print("Press Ctrl+C to stop\n")

    with open(args.output, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        try:
            while True:
                raw_line = ser.readline().decode("utf-8", errors="ignore").strip()

                if not raw_line:
                    continue

                if raw_line.startswith("CSI,"):
                    pc_timestamp = datetime.now().isoformat(timespec="milliseconds")

                    writer.writerow([
                        args.label,
                        pc_timestamp,
                        raw_line
                    ])

                    file.flush()
                    print(args.label, pc_timestamp, raw_line)

        except KeyboardInterrupt:
            print("\nLogger stopped by user.")

        finally:
            ser.close()


if __name__ == "__main__":
    main()