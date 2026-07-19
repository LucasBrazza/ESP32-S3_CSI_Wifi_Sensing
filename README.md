# ESP32-S3 CSI Wi-Fi Sensing

Projeto de TCC para aquisição e classificação de estados do ambiente por meio de CSI (*Channel State Information*) usando dois ESP32-S3 em uma arquitetura AP/STA.

O sistema atual trabalha com três classes:

- `empty`;
- `static_presence`;
- `movement`.

A aquisição é executada a aproximadamente **50 pacotes por segundo**, com canal Wi-Fi fixo, largura de banda **HT20** e transporte serial binário a **921600 baud**.

## Arquitetura

```text
AP_controller
    │
    │ UDP unicast a cada 20 ms
    ▼
STA_CSI_receiver
    │
    │ CSI2 binário pela UART a 921600 baud
    ▼
Tools/acquisition/gui/csi_viewer.py
    │
    │ arquivos CSIBIN1 versão 2
    ▼
Tools/datasets
    │
    ├── pré-processamento
    ├── extração e seleção de features
    ├── treinamento e validação
    └── inferência em tempo real
```

## Estado atual

A cadeia de aquisição foi validada com:

- AP enviando tráfego UDP periódico próximo de 50 Hz;
- Wi-Fi fixado em 20 MHz;
- CSI com `csi_len=256`, equivalente a 128 pares complexo I/Q na configuração atual;
- callback CSI leve, com cópia imediata para uma fila FreeRTOS;
- filtragem dos quadros pelo BSSID do AP e pelo MAC do STA;
- transporte serial binário com CRC-16/CCITT-FALSE;
- diagnóstico de falhas de sequência e descartes no ESP32 e no computador;
- arquivos de dataset binários na versão 2, mantendo leitura retrocompatível da versão 1.

Em uma aquisição saudável de 5 segundos, o esperado é aproximadamente:

```text
240 a 260 pacotes
48 a 52 Hz
Sequence gaps: 0
CRC errors: 0
PC drops: 0
ESP drops: 0
csi_len: 256
bandwidth: 20 MHz
```

O valor de MCS pode variar durante a aquisição sem alterar o tamanho do vetor CSI.

## Estrutura principal

```text
ESP32-S3_CSI_Wifi_Sensing/
├── AP_controller/          # AP, largura HT20 e tráfego UDP controlado
├── STA_CSI_receiver/       # recepção UDP, captura CSI e protocolo CSI2
├── Tools/
│   ├── acquisition/        # GUI, parser serial e diagnósticos de dataset
│   ├── csi/                # leitura, escrita e conversão dos arquivos binários
│   ├── preprocessing/      # limpeza, filtros, janelas e features
│   ├── training/           # seleção, treinamento e validação
│   ├── classification/     # árvore de decisão implementada no projeto
│   ├── real_time/          # inferência em fluxo contínuo
│   └── datasets/           # dados, artefatos intermediários e resultados
├── requirements.txt        # dependências diretas do ambiente Python
└── requirements-lock.txt   # versões exatas do ambiente validado
```

## Pipeline de processamento

```text
CSI bruto I/Q
    ↓
amplitude
    ↓
limpeza de subportadoras
    ↓
filtro de Hampel
    ↓
média móvel
    ↓
normalização z-score
    ↓
remoção de redundância por correlação
    ↓
janelas deslizantes
    ↓
extração de features estatísticas
    ↓
Fisher Score / Top-K
    ↓
classificação e validação por arquivo
```

O pipeline principal mantém uma implementação própria de árvore de decisão para facilitar a futura exportação do modelo. Na branch `research/literature-guided-pipeline-review`, os experimentos comparativos usam também `scikit-learn` e `xgboost`; o candidato provisório mais recente é um Gradient Boosting compacto com Top-K 126, 20 estimadores, profundidade 3 e taxa de aprendizado 0,1. Essa escolha deve ser revalidada após a coleta do Dataset v2.

## Requisitos

- 2 placas ESP32-S3;
- ESP-IDF 6.0;
- Python 3.11;
- cabos USB para gravação e comunicação serial;
- Windows PowerShell ou outro terminal compatível com ESP-IDF e Python.

### Ambiente Python

Na raiz do repositório:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para reproduzir exatamente o ambiente validado:

```powershell
python -m pip install -r requirements-lock.txt
```

O arquivo `Tools/acquisition/gui/requirements.txt` contém apenas as dependências mínimas da interface de aquisição.

## Preparação do ESP-IDF no Windows

Em um novo PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
. "C:\esp\v6.0\esp-idf\export.ps1"
```

## Compilação e gravação

### AP

```powershell
cd AP_controller
idf.py build
idf.py -p COM3 flash
idf.py -p COM3 monitor
```

O AP deve mostrar uma taxa próxima de 50 pacotes por segundo e zero erros de envio.

### STA

```powershell
cd STA_CSI_receiver
idf.py build
idf.py -p COM4 flash
```

Após o início do fluxo CSI binário, a COM do STA deve ser aberta pela GUI a 921600 baud. Não mantenha o `idf.py monitor` conectado à mesma porta durante a aquisição.

## Executar a interface de aquisição

Na raiz do repositório, com o ambiente virtual ativo:

```powershell
python Tools/acquisition/gui/csi_viewer.py
```

Configuração padrão atual:

```text
Baud: 921600
Duração: 5 s
Classes: empty, static_presence, movement
Formato salvo: <pasta selecionada>/raw_bin/<classe>_AAAAMMDD_HHMMSS.bin
```

Para organizar uma nova base sem misturá-la ao dataset antigo, selecione pastas separadas por sessão e quadrante, por exemplo:

```text
Tools/datasets/raw_v2/session_01/quad1/
Tools/datasets/raw_v2/session_01/quad2/
...
```

A GUI acrescentará automaticamente a subpasta `raw_bin`.

## Documentação específica

- [`AP_controller/README.md`](AP_controller/README.md): configuração do AP e geração do tráfego UDP;
- [`STA_CSI_receiver/README.md`](STA_CSI_receiver/README.md): captura, filtro e protocolo binário CSI2;
- [`Tools/README.md`](Tools/README.md): aquisição no computador, formato do dataset e pipeline Python.

## Cuidados experimentais

- inicialize o AP antes do STA;
- mantenha posição, canal e largura de banda dos dispositivos constantes;
- não misture diretamente o Dataset v2 com coletas antigas de densidade diferente;
- registre sessões independentes para reduzir dependência temporal;
- confira os diagnósticos da GUI antes de aceitar uma coleta;
- preserve os arquivos binários brutos e gere CSV apenas para inspeção;
- reavalie o tamanho das janelas quando a frequência de aquisição for alterada.
