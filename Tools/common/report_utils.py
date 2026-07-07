def print_header(title):
    print()
    print("=" * 70)
    print(title.upper())
    print("=" * 70)


def print_section(title):
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


def print_result(label, value):
    print(f"[RESULT] {label}: {value}")


def print_info(message):
    print(f"[INFO] {message}")


def print_debug(message):
    print(f"[DEBUG] {message}")