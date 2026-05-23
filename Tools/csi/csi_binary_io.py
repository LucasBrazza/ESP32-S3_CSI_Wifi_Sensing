import struct
from pathlib import Path


MAGIC = b"CSIBIN1"
VERSION = 1

HEADER_FORMAT = "<7sI"
PACKET_HEADER_FORMAT = "<H d i i i i I"
# label_len, timestamp, rssi, rate, channel, csi_len, num_subcarriers


def write_packets(file_path: Path, packets: list[dict]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as file:
        file.write(struct.pack(HEADER_FORMAT, MAGIC, VERSION))

        for packet in packets:
            label_bytes = packet["label"].encode("utf-8")

            imag = packet["imag"]
            real = packet["real"]

            n = min(len(imag), len(real))

            file.write(
                struct.pack(
                    PACKET_HEADER_FORMAT,
                    len(label_bytes),
                    float(packet["pc_timestamp"]),
                    int(packet["rssi"]),
                    int(packet["rate"]),
                    int(packet["channel"]),
                    int(packet["csi_len"]),
                    int(n),
                )
            )

            file.write(label_bytes)

            for value in imag[:n]:
                file.write(struct.pack("<h", int(value)))

            for value in real[:n]:
                file.write(struct.pack("<h", int(value)))


def read_packets(file_path: Path) -> list[dict]:
    packets = []

    with open(file_path, "rb") as file:
        header_size = struct.calcsize(HEADER_FORMAT)
        header_data = file.read(header_size)

        if len(header_data) != header_size:
            raise ValueError("Invalid or empty CSI binary file.")

        magic, version = struct.unpack(HEADER_FORMAT, header_data)

        if magic != MAGIC:
            raise ValueError("Invalid CSI binary file.")

        if version != VERSION:
            raise ValueError(f"Unsupported CSI binary version: {version}")

        packet_header_size = struct.calcsize(PACKET_HEADER_FORMAT)

        while True:
            header_data = file.read(packet_header_size)

            if not header_data:
                break

            if len(header_data) != packet_header_size:
                raise ValueError("Corrupted packet header.")

            (
                label_len,
                pc_timestamp,
                rssi,
                rate,
                channel,
                csi_len,
                n,
            ) = struct.unpack(PACKET_HEADER_FORMAT, header_data)

            label_data = file.read(label_len)

            if len(label_data) != label_len:
                raise ValueError("Corrupted label data.")

            label = label_data.decode("utf-8")

            imag = []
            real = []

            for _ in range(n):
                data = file.read(2)

                if len(data) != 2:
                    raise ValueError("Corrupted imaginary data.")

                imag.append(struct.unpack("<h", data)[0])

            for _ in range(n):
                data = file.read(2)

                if len(data) != 2:
                    raise ValueError("Corrupted real data.")

                real.append(struct.unpack("<h", data)[0])

            packets.append(
                {
                    "label": label,
                    "pc_timestamp": pc_timestamp,
                    "rssi": rssi,
                    "rate": rate,
                    "channel": channel,
                    "csi_len": csi_len,
                    "imag": imag,
                    "real": real,
                }
            )

    return packets