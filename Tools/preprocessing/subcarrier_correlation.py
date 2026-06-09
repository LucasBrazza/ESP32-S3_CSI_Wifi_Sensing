from core_math import mean, std, get_column


def covariance(values_a, values_b):
    if not values_a or not values_b:
        return 0.0

    n = min(len(values_a), len(values_b))

    if n == 0:
        return 0.0

    avg_a = mean(values_a[:n])
    avg_b = mean(values_b[:n])

    total = 0.0

    for i in range(n):
        total += (values_a[i] - avg_a) * (values_b[i] - avg_b)

    return total / n


def pearson_correlation(values_a, values_b, eps=1e-8):
    """
    Correlação de Pearson entre dois sinais.
    Retorna valor entre -1 e 1.
    """

    n = min(len(values_a), len(values_b))

    if n == 0:
        return 0.0

    values_a = values_a[:n]
    values_b = values_b[:n]

    cov = covariance(values_a, values_b)
    std_a = std(values_a)
    std_b = std(values_b)

    return cov / ((std_a * std_b) + eps)


def select_non_redundant_subcarriers(matrix, threshold=0.95):
    """
    Remove subportadoras redundantes com base na correlação.

    Entrada:
        matrix[pacote][subportadora]

    Saída:
        selected_subcarriers: índices das subportadoras mantidas
    """

    if not matrix:
        return []

    num_subcarriers = len(matrix[0])

    selected_subcarriers = []

    for candidate_sc in range(num_subcarriers):
        candidate_signal = get_column(matrix, candidate_sc)

        is_redundant = False

        for selected_sc in selected_subcarriers:
            selected_signal = get_column(matrix, selected_sc)

            corr = pearson_correlation(
                candidate_signal,
                selected_signal,
            )

            if abs(corr) >= threshold:
                is_redundant = True
                break

        if not is_redundant:
            selected_subcarriers.append(candidate_sc)

    return selected_subcarriers


def filter_matrix_by_subcarriers(matrix, selected_subcarriers):
    """
    Mantém somente as subportadoras selecionadas.

    Entrada:
        matrix[pacote][subportadora]

    Saída:
        filtered_matrix[pacote][subportadora_selecionada]
    """

    filtered = []

    for row in matrix:
        new_row = []

        for sc in selected_subcarriers:
            if sc < len(row):
                new_row.append(row[sc])

        filtered.append(new_row)

    return filtered


def print_selected_subcarriers(selected_subcarriers):
    print()
    print("Subportadoras após remoção de redundância:")
    print(selected_subcarriers)
    print("Total:", len(selected_subcarriers))
