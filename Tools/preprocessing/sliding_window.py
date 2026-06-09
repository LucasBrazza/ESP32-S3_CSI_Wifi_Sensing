def create_sliding_windows(matrix, window_size=20, step_size=5):
    """
    Cria janelas deslizantes a partir de uma matriz CSI.

    Entrada:
        matrix[pacote][subportadora]

    Saída:
        windows[janela][pacote][subportadora]
    """

    windows = []

    if not matrix:
        return windows

    if window_size <= 0:
        return windows

    if step_size <= 0:
        return windows

    total_packets = len(matrix)

    if total_packets < window_size:
        return windows

    start = 0

    while start + window_size <= total_packets:
        end = start + window_size

        window = []

        for packet_index in range(start, end):
            window.append(matrix[packet_index][:])

        windows.append(window)

        start += step_size

    return windows


def create_labeled_windows(matrix, label, window_size=20, step_size=5):
    """
    Cria janelas deslizantes já associadas a uma classe.

    Saída:
        [
            {
                "label": "empty",
                "data": janela
            }
        ]
    """

    raw_windows = create_sliding_windows(
        matrix,
        window_size=window_size,
        step_size=step_size,
    )

    labeled_windows = []

    for window in raw_windows:
        labeled_windows.append(
            {
                "label": label,
                "data": window,
            }
        )

    return labeled_windows


def count_windows(total_packets, window_size=20, step_size=5):
    """
    Calcula quantas janelas serão geradas.
    """

    if total_packets < window_size:
        return 0

    count = 0
    start = 0

    while start + window_size <= total_packets:
        count += 1
        start += step_size

    return count


