from pathlib import Path
from collections import Counter

BASE = Path("datasets/raw_bin")

for quad in sorted(BASE.glob("quad*")):

    print("=" * 60)
    print(quad.name)

    arquivos = list((quad / "raw_bin").glob("*.bin"))

    classes = Counter()

    for arq in arquivos:

        nome = arq.stem.lower()

        if nome.startswith("empty"):
            classes["empty"] += 1

        elif nome.startswith("static"):
            classes["static"] += 1

        elif nome.startswith("movement"):
            classes["movement"] += 1

        else:
            classes["desconhecido"] += 1

    print(classes)

    print()

    for classe in ["empty", "static", "movement"]:
        print(f"{classe:10} : {classes[classe]}")