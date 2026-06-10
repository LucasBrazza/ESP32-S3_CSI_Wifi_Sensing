"""
Sliding Window Generation

This module converts a continuous CSI amplitude matrix into overlapping
temporal windows.

The CSI matrix follows the format:

    matrix[packet][subcarrier]

Each generated window follows the format:

    window[packet][subcarrier]

and the full output follows:

    windows[window][packet][subcarrier]

Sliding windows are used because the classifier should not analyze a
single CSI packet in isolation. Human presence and movement are temporal
phenomena, so each sample must represent a short sequence of packets.

Example:

    window_size = 20
    step_size = 5

Generated windows:

    packets 0  - 19
    packets 5  - 24
    packets 10 - 29
    packets 15 - 34

Formula for the number of windows:

    number_of_windows = floor((N - window_size) / step_size) + 1

where:

    N = total number of valid packets

This same concept is later adapted to real-time processing using a
fixed-size buffer.
"""

def create_sliding_windows(matrix, window_size=20, step_size=5):
    """
    Create overlapping sliding windows from a CSI matrix.

    Input:

        matrix[packet][subcarrier]

    Output:

        windows[window][packet][subcarrier]

    The window size defines how many packets are included in each
    sample. The step size defines how far the window moves after each
    generated sample.

    A smaller step size increases overlap between windows and generates
    more training samples. A larger step size reduces overlap and lowers
    computational cost.
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
    Create sliding windows and assign the same class label to each one.

    This is used during supervised training, where each collected file
    corresponds to a known class such as:

        empty
        static_presence
        movement

    Output format:

        {
            "label": class_name,
            "data": window
        }

    The label is kept together with the window so that feature extraction
    and classification training can preserve the class information.
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
    Compute how many sliding windows will be generated.

    Formula:

        number_of_windows = floor((N - window_size) / step_size) + 1

    where:

        N = total number of valid packets

    This helper is useful for diagnostics, especially when comparing
    files with different packet counts.
    """

    if total_packets < window_size:
        return 0

    count = 0
    start = 0

    while start + window_size <= total_packets:
        count += 1
        start += step_size

    return count


