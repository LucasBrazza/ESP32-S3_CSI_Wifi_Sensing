import math


def get_num_subcarriers(packet: dict) -> int:
    return min(
        len(packet["imag"]),
        len(packet["real"]),
    )


def compute_amplitude(real: float, imag: float) -> float:
    return math.sqrt(real * real + imag * imag)


def packet_to_amplitudes(packet: dict) -> list[float]:
    n = get_num_subcarriers(packet)

    return [
        compute_amplitude(
            packet["real"][i],
            packet["imag"][i],
        )
        for i in range(n)
    ]


def select_structural_subcarriers(amplitudes: list[float]) -> list[float]:
    n = len(amplitudes)

    if n == 192:
        block = amplitudes[64:128]
        indexes = list(range(6, 32)) + list(range(33, 59))

        return [
            block[i]
            for i in indexes
            if i < len(block)
        ]

    if n == 64:
        indexes = list(range(6, 32)) + list(range(33, 59))

        return [
            amplitudes[i]
            for i in indexes
            if i < len(amplitudes)
        ]

    return amplitudes


def mean(values: list[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def variance(values: list[float]) -> float:
    if not values:
        return 0.0

    avg = mean(values)

    return sum(
        (value - avg) ** 2
        for value in values
    ) / len(values)


def std(values: list[float]) -> float:
    return math.sqrt(variance(values))


def energy(values: list[float]) -> float:
    return sum(
        value * value
        for value in values
    )


def extract_basic_features(amplitudes: list[float]) -> dict:
    useful = select_structural_subcarriers(amplitudes)

    if not useful:
        return {
            "mean_amplitude": 0.0,
            "std_amplitude": 0.0,
            "variance_amplitude": 0.0,
            "energy_amplitude": 0.0,
            "max_amplitude": 0.0,
            "min_amplitude": 0.0,
            "range_amplitude": 0.0,
            "num_subcarriers": 0,
        }

    max_value = max(useful)
    min_value = min(useful)

    return {
        "mean_amplitude": mean(useful),
        "std_amplitude": std(useful),
        "variance_amplitude": variance(useful),
        "energy_amplitude": energy(useful),
        "max_amplitude": max_value,
        "min_amplitude": min_value,
        "range_amplitude": max_value - min_value,
        "num_subcarriers": len(useful),
    }