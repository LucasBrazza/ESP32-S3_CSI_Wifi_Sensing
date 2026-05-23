def create_csi_packet(
    label,
    pc_timestamp,
    rssi,
    rate,
    channel,
    csi_len,
    imag,
    real,
):
    return {
        "label": label,
        "pc_timestamp": pc_timestamp,
        "rssi": rssi,
        "rate": rate,
        "channel": channel,
        "csi_len": csi_len,
        "imag": imag,
        "real": real,
    }


def get_num_subcarriers(packet):
    return min(
        len(packet["imag"]),
        len(packet["real"]),
    )