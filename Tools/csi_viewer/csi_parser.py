import re
import numpy as np


def parse_csi_line(line: str):
    """
    Parses CSI lines from ESP32 serial output.

    Supported examples:
    CSI,<timestamp_us>,<rssi>,<rate>,<channel>,<len>,imag0,real0,imag1,real1,...
    CSI,len=384,rssi=-51,rate=11,channel=6,...
    """

    if "CSI" not in line:
        return None

    values = re.findall(r"-?\d+", line)

    if len(values) < 12:
        return None

    nums = list(map(int, values))

    # Heuristic:
    # First fields are metadata. CSI payload usually starts after len field.
    # Format: CSI,timestamp,rssi,rate,channel,len,iq...
    if len(nums) > 20:
        metadata = {
            "timestamp_us": nums[0],
            "rssi": nums[1],
            "rate": nums[2],
            "channel": nums[3],
            "csi_len": nums[4],
        }
        iq = nums[5:]
    else:
        return None

    if len(iq) % 2 != 0:
        iq = iq[:-1]

    imag = np.array(iq[0::2], dtype=np.float32)
    real = np.array(iq[1::2], dtype=np.float32)

    amplitude = np.sqrt(real**2 + imag**2)
    phase = np.arctan2(imag, real)

    return {
        "metadata": metadata,
        "imag": imag,
        "real": real,
        "amplitude": amplitude,
        "phase": phase,
    }