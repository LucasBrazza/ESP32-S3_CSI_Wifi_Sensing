from __future__ import annotations

import struct
from typing import Any

import numpy as np


CSI_MAGIC = b"CSI2"
CSI_PROTOCOL_VERSION = 1

CSI_FRAME_TYPE_SAMPLE = 1
CSI_FRAME_TYPE_STATS = 2

CSI_COMMON_HEADER_SIZE = 8
CSI_SAMPLE_METADATA_SIZE = 18
CSI_STATS_FRAME_SIZE = 46
CSI_MAX_FRAME_SIZE = 2048


class CSIFrameParser:
    """Incrementally parses the binary CSI stream emitted by the ESP32-S3.

    The parser is resilient to text logs or corrupted bytes mixed into the
    serial stream. It searches for the ``CSI2`` magic, validates frame length
    and CRC-16/CCITT-FALSE, and then returns sample or statistics events.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.last_sequence: int | None = None

        self.sample_frames = 0
        self.stats_frames = 0
        self.crc_errors = 0
        self.invalid_frames = 0
        self.unsupported_frames = 0
        self.bytes_discarded = 0
        self.sequence_gaps = 0

    def reset(self) -> None:
        self._buffer.clear()
        self.last_sequence = None

        self.sample_frames = 0
        self.stats_frames = 0
        self.crc_errors = 0
        self.invalid_frames = 0
        self.unsupported_frames = 0
        self.bytes_discarded = 0
        self.sequence_gaps = 0

    def feed(self, data: bytes | bytearray | memoryview) -> list[dict[str, Any]]:
        """Feeds serial bytes and returns every complete decoded event."""

        if data:
            self._buffer.extend(data)

        events: list[dict[str, Any]] = []

        while True:
            if len(self._buffer) < len(CSI_MAGIC):
                break

            magic_index = self._buffer.find(CSI_MAGIC)

            if magic_index < 0:
                # Keep the final three bytes because they may be the beginning
                # of a magic word split across two serial reads.
                keep = len(CSI_MAGIC) - 1
                discard_count = max(0, len(self._buffer) - keep)

                if discard_count:
                    del self._buffer[:discard_count]
                    self.bytes_discarded += discard_count

                break

            if magic_index > 0:
                del self._buffer[:magic_index]
                self.bytes_discarded += magic_index

            if len(self._buffer) < CSI_COMMON_HEADER_SIZE:
                break

            version = self._buffer[4]
            frame_type = self._buffer[5]
            frame_size = struct.unpack_from("<H", self._buffer, 6)[0]

            if (
                version != CSI_PROTOCOL_VERSION
                or frame_size < CSI_COMMON_HEADER_SIZE + 2
                or frame_size > CSI_MAX_FRAME_SIZE
            ):
                # Discard one byte and search again. This recovers from a false
                # magic sequence in logs or corrupted payloads.
                del self._buffer[0]
                self.invalid_frames += 1
                continue

            if len(self._buffer) < frame_size:
                break

            frame = bytes(self._buffer[:frame_size])

            expected_crc = struct.unpack_from("<H", frame, frame_size - 2)[0]
            calculated_crc = calculate_crc16_ccitt(frame[4:-2])

            if expected_crc != calculated_crc:
                del self._buffer[0]
                self.crc_errors += 1
                continue

            del self._buffer[:frame_size]

            if frame_type == CSI_FRAME_TYPE_SAMPLE:
                event = self._parse_sample_frame(frame)

                if event is not None:
                    events.append(event)
                    self.sample_frames += 1

            elif frame_type == CSI_FRAME_TYPE_STATS:
                event = self._parse_stats_frame(frame)

                if event is not None:
                    events.append(event)
                    self.stats_frames += 1

            else:
                self.unsupported_frames += 1

        return events

    def diagnostics(self) -> dict[str, int]:
        return {
            "sample_frames": self.sample_frames,
            "stats_frames": self.stats_frames,
            "crc_errors": self.crc_errors,
            "invalid_frames": self.invalid_frames,
            "unsupported_frames": self.unsupported_frames,
            "bytes_discarded": self.bytes_discarded,
            "sequence_gaps": self.sequence_gaps,
            "buffered_bytes": len(self._buffer),
        }

    def _parse_sample_frame(self, frame: bytes) -> dict[str, Any] | None:
        minimum_size = (
            CSI_COMMON_HEADER_SIZE
            + CSI_SAMPLE_METADATA_SIZE
            + 2
        )

        if len(frame) < minimum_size:
            self.invalid_frames += 1
            return None

        offset = CSI_COMMON_HEADER_SIZE

        sequence = struct.unpack_from("<I", frame, offset)[0]
        offset += 4

        timestamp_us = struct.unpack_from("<Q", frame, offset)[0]
        offset += 8

        rssi = struct.unpack_from("<b", frame, offset)[0]
        offset += 1

        rate = frame[offset]
        offset += 1

        channel = frame[offset]
        offset += 1

        flags = frame[offset]
        offset += 1

        csi_len = struct.unpack_from("<H", frame, offset)[0]
        offset += 2

        expected_size = offset + csi_len + 2

        if expected_size != len(frame):
            self.invalid_frames += 1
            return None

        raw_csi = np.frombuffer(
            frame,
            dtype=np.int8,
            count=csi_len,
            offset=offset,
        ).astype(np.float32)

        if raw_csi.size % 2 != 0:
            raw_csi = raw_csi[:-1]

        imag = raw_csi[0::2].copy()
        real = raw_csi[1::2].copy()

        if self.last_sequence is not None:
            expected_sequence = (self.last_sequence + 1) & 0xFFFFFFFF

            if sequence != expected_sequence:
                gap = (sequence - expected_sequence) & 0xFFFFFFFF

                # A very large unsigned gap normally means the ESP32 restarted.
                if gap < 0x80000000:
                    self.sequence_gaps += gap

        self.last_sequence = sequence

        return {
            "type": "sample",
            "metadata": {
                "protocol_version": frame[4],
                "sequence": sequence,
                "timestamp_us": timestamp_us,
                "rssi": rssi,
                "rate": rate,
                "channel": channel,
                "flags": flags,
                "csi_len": csi_len,
            },
            "imag": imag,
            "real": real,
        }

    def _parse_stats_frame(self, frame: bytes) -> dict[str, Any] | None:
        if len(frame) != CSI_STATS_FRAME_SIZE:
            self.invalid_frames += 1
            return None

        offset = CSI_COMMON_HEADER_SIZE

        timestamp_us = struct.unpack_from("<Q", frame, offset)[0]
        offset += 8

        names = (
            "csi_received",
            "csi_queued",
            "csi_serialized",
            "queue_drops",
            "invalid",
            "oversize",
        )

        values: dict[str, int] = {}

        for name in names:
            values[name] = struct.unpack_from("<I", frame, offset)[0]
            offset += 4

        values["queue_pending"] = struct.unpack_from("<H", frame, offset)[0]
        offset += 2

        values["reserved"] = struct.unpack_from("<H", frame, offset)[0]

        return {
            "type": "stats",
            "metadata": {
                "protocol_version": frame[4],
                "timestamp_us": timestamp_us,
            },
            "stats": values,
        }


def calculate_crc16_ccitt(data: bytes | bytearray | memoryview) -> int:
    """CRC-16/CCITT-FALSE used by the ESP32 binary protocol."""

    crc = 0xFFFF

    for value in data:
        crc ^= int(value) << 8

        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF

    return crc
