from csi_preprocessor import extract_features


def main():
    # Simulated CSI amplitude vector with 192 subcarriers
    amplitudes_192 = [float(i % 50 + 1) for i in range(192)]

    features = extract_features(amplitudes_192)

    print("Extracted features:")
    for key, value in features.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()