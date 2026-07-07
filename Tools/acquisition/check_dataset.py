from pathlib import Path

BASE = Path("Tools/datasets/raw_bin")

quadrantes = ["quad1", "quad2", "quad3", "quad4", "quad5"]

for quad in quadrantes:
    pasta = BASE / quad 

    print(f"\n=== {quad} ===")

    if not pasta.exists():
        print("Pasta não encontrada:", pasta)
        continue

    arquivos = list(pasta.rglob("*"))

    arquivos_bin = [
        arq for arq in arquivos
        if arq.is_file()
    ]

    print("Total de arquivos:", len(arquivos_bin))

    for arq in arquivos_bin[:5]:
        print("Exemplo:", arq.name, "| tamanho:", arq.stat().st_size, "bytes")