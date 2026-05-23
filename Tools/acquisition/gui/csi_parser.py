import re
import numpy as np


def parse_csi_line(line: str):
    """
    Parses raw CSI lines from ESP32 serial output.

    Expected format:
    CSI,<timestamp_us>,<rssi>,<rate>,<channel>,<len>,imag0,real0,imag1,real1,...

    This function only parses raw CSI data.
    It does not compute amplitude, phase, features, or statistics.
    """

    if "CSI" not in line:
        return None

    values = re.findall(r"-?\d+", line)

    if len(values) < 12:
        return None

    nums = list(map(int, values))

    metadata = {
        "timestamp_us": nums[0],
        "rssi": nums[1],
        "rate": nums[2],
        "channel": nums[3],
        "csi_len": nums[4],
    }

    iq = nums[5:]

    if len(iq) % 2 != 0:
        iq = iq[:-1]

    imag = np.array(iq[0::2], dtype=np.float32)
    real = np.array(iq[1::2], dtype=np.float32)

    return {
        "metadata": metadata,
        "imag": imag,
        "real": real,
    }