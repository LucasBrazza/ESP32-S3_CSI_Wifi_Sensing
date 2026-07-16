from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO


MAGIC = b"CSIBIN1"
CURRENT_VERSION = 2
SUPPORTED_VERSIONS = {1, 2}

HEADER_FORMAT = "<7sI"

V1_PACKET_HEADER_FORMAT = "<HdiiiiI"
# label_len, pc_timestamp, rssi, rate, channel, csi_len, num_subcarriers

V2_PACKET_HEADER_FORMAT = "<HddQIIiiiIII"
# label_len, pc_timestamp, capture_timestamp, esp_timestamp_us,
# sequence, packet_index, rssi, rate, channel, csi_len,
# num_subcarriers, flags


def write_packets(file_path: Path, packets: list[dict]) -> None:
    """Writes packets using dataset binary format version 2.

    Version 2 preserves the ESP32 timestamp and sequence number while keeping
    ``read_packets`` compatible with all existing version 1 datasets.
    """

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as file:
        file.write(struct.pack(HEADER_FORMAT, MAGIC, CURRENT_VERSION))

        for fallback_index, packet in enumerate(packets, start=1):
            _write_v2_packet(file, packet, fallback_index)


def read_packets(file_path: Path) -> list[dict]:
    packets: list[dict] = []

    with open(file_path, "rb") as file:
        header_size = struct.calcsize(HEADER_FORMAT)
        header_data = _read_exact(file, header_size)

        magic, version = struct.unpack(HEADER_FORMAT, header_data)

        if magic != MAGIC:
            raise ValueError("Invalid CSI binary file.")

        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported CSI binary version: {version}")

        if version == 1:
            packets = _read_v1_packets(file)
        else:
            packets = _read_v2_packets(file)

    return packets


def _write_v2_packet(
    file: BinaryIO,
    packet: dict,
    fallback_index: int,
) -> None:
    label_bytes = str(packet.get("label", "")).encode("utf-8")

    imag = packet.get("imag", [])
    real = packet.get("real", [])
    n = min(len(imag), len(real))

    pc_timestamp = float(packet.get("pc_timestamp", 0.0) or 0.0)
    capture_timestamp = float(
        packet.get("capture_timestamp", pc_timestamp) or pc_timestamp
    )

    file.write(
        struct.pack(
            V2_PACKET_HEADER_FORMAT,
            len(label_bytes),
            pc_timestamp,
            capture_timestamp,
            int(packet.get("esp_timestamp_us", 0) or 0),
            int(packet.get("sequence", 0) or 0),
            int(packet.get("packet_index", fallback_index) or fallback_index),
            int(packet.get("rssi", 0) or 0),
            int(packet.get("rate", 0) or 0),
            int(packet.get("channel", 0) or 0),
            int(packet.get("csi_len", 0) or 0),
            int(n),
            int(packet.get("flags", 0) or 0),
        )
    )

    file.write(label_bytes)

    if n > 0:
        file.write(struct.pack(f"<{n}h", *(int(value) for value in imag[:n])))
        file.write(struct.pack(f"<{n}h", *(int(value) for value in real[:n])))


def _read_v1_packets(file: BinaryIO) -> list[dict]:
    packets: list[dict] = []
    packet_header_size = struct.calcsize(V1_PACKET_HEADER_FORMAT)
    packet_index = 0

    while True:
        header_data = file.read(packet_header_size)

        if not header_data:
            break

        if len(header_data) != packet_header_size:
            raise ValueError("Corrupted version 1 packet header.")

        (
            label_len,
            pc_timestamp,
            rssi,
            rate,
            channel,
            csi_len,
            n,
        ) = struct.unpack(V1_PACKET_HEADER_FORMAT, header_data)

        packet_index += 1
        label = _read_exact(file, label_len).decode("utf-8")
        imag = _read_int16_array(file, n, "imaginary")
        real = _read_int16_array(file, n, "real")

        packets.append(
            {
                "label": label,
                "pc_timestamp": pc_timestamp,
                "capture_timestamp": pc_timestamp,
                "esp_timestamp_us": 0,
                "sequence": 0,
                "packet_index": packet_index,
                "rssi": rssi,
                "rate": rate,
                "channel": channel,
                "csi_len": csi_len,
                "flags": 0,
                "imag": imag,
                "real": real,
                "file_version": 1,
            }
        )

    return packets


def _read_v2_packets(file: BinaryIO) -> list[dict]:
    packets: list[dict] = []
    packet_header_size = struct.calcsize(V2_PACKET_HEADER_FORMAT)

    while True:
        header_data = file.read(packet_header_size)

        if not header_data:
            break

        if len(header_data) != packet_header_size:
            raise ValueError("Corrupted version 2 packet header.")

        (
            label_len,
            pc_timestamp,
            capture_timestamp,
            esp_timestamp_us,
            sequence,
            packet_index,
            rssi,
            rate,
            channel,
            csi_len,
            n,
            flags,
        ) = struct.unpack(V2_PACKET_HEADER_FORMAT, header_data)

        label = _read_exact(file, label_len).decode("utf-8")
        imag = _read_int16_array(file, n, "imaginary")
        real = _read_int16_array(file, n, "real")

        packets.append(
            {
                "label": label,
                "pc_timestamp": pc_timestamp,
                "capture_timestamp": capture_timestamp,
                "esp_timestamp_us": esp_timestamp_us,
                "sequence": sequence,
                "packet_index": packet_index,
                "rssi": rssi,
                "rate": rate,
                "channel": channel,
                "csi_len": csi_len,
                "flags": flags,
                "imag": imag,
                "real": real,
                "file_version": 2,
            }
        )

    return packets


def _read_exact(file: BinaryIO, size: int) -> bytes:
    data = file.read(size)

    if len(data) != size:
        raise ValueError("Unexpected end of CSI binary file.")

    return data


def _read_int16_array(
    file: BinaryIO,
    count: int,
    field_name: str,
) -> list[int]:
    if count < 0:
        raise ValueError(f"Invalid {field_name} sample count: {count}")

    if count == 0:
        return []

    byte_count = count * 2
    data = _read_exact(file, byte_count)
    return list(struct.unpack(f"<{count}h", data))
