# Tools

Ferramentas Python para aquisição, armazenamento, inspeção, pré-processamento, treinamento, validação e inferência usando os dados CSI enviados pelo `STA_CSI_receiver`.

## Requisitos

O ambiente validado utiliza Python 3.11.

Na raiz do repositório:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Somente para a interface de aquisição:

```powershell
python -m pip install -r Tools/acquisition/gui/requirements.txt
```

## Fluxo de aquisição atual

```text
AP: UDP unicast a 50 Hz e HT20
    ↓
STA: callback CSI + fila FreeRTOS
    ↓
CSI2 binário a 921600 baud
    ↓
CSIFrameParser
    ↓
CSIViewer
    ↓
CSIBIN1 versão 2
```

A implementação textual antiga foi substituída no fluxo oficial pelo protocolo binário. Os scripts de CLI antigos permanecem úteis para histórico e depuração, mas a GUI é a ferramenta principal para novas coletas.

## Interface gráfica

Execute a partir da raiz do repositório:

```powershell
python Tools/acquisition/gui/csi_viewer.py
```

Configurações principais:

| Campo | Valor ou função |
|---|---|
| Porta | porta serial do STA, normalmente COM4 no ambiente de desenvolvimento |
| Baud | `921600` |
| Classe | `empty`, `static_presence` ou `movement` |
| Offset | atraso antes do início da janela de coleta |
| Duração | 5 s por padrão |
| Output folder | pasta-base escolhida para a sessão/quadrante |

O arquivo é salvo em:

```text
<output folder>/raw_bin/<classe>_AAAAMMDD_HHMMSS.bin
```

Exemplo de organização do Dataset v2:

```text
Tools/datasets/raw_v2/
└── session_01/
    ├── quad1/raw_bin/
    ├── quad2/raw_bin/
    ├── quad3/raw_bin/
    ├── quad4/raw_bin/
    └── quad5/raw_bin/
```

A classe já é armazenada dentro de cada pacote e também aparece no nome do arquivo. A sessão e o quadrante são representados atualmente pela estrutura de pastas.

## Diagnósticos da GUI

A barra de status apresenta:

| Indicador | Significado |
|---|---|
| `Rate` | amostras recebidas no último segundo |
| `Seq gaps` | amostras ausentes detectadas pelo campo de sequência |
| `CRC` | frames com CRC inválido |
| `PC drops` | eventos descartados porque a fila do computador encheu |
| `ESP drops` | amostras descartadas porque a fila FreeRTOS encheu |
| `ESP pending` | amostras aguardando serialização no STA |

Uma coleta oficial deve ser aceita somente quando os contadores de erro e descarte estiverem em zero e a taxa estiver próxima de 50 Hz.

## Protocolo serial

O parser em `acquisition/gui/csi_parser.py` processa frames com:

```text
Magic: CSI2
Versão: 1
Tipos: sample e stats
Ordem de bytes: little-endian
Integridade: CRC-16/CCITT-FALSE
```

A amostra contém sequência, timestamp do ESP32, RSSI, rate, canal, flags, tamanho e CSI bruto. O parser separa automaticamente o vetor intercalado em arrays `imag` e `real`.

Na configuração atual HT20:

```text
csi_len = 256 inteiros int8
imag = 128 valores
real = 128 valores
```

## Formato de dataset binário

O módulo `csi/csi_binary_io.py` usa:

```text
Magic do arquivo: CSIBIN1
Versão atual de escrita: 2
Versões aceitas na leitura: 1 e 2
```

Cada pacote da versão 2 preserva:

- `label`;
- `pc_timestamp`;
- `capture_timestamp`, alinhado ao relógio do ESP32;
- `esp_timestamp_us`;
- `sequence`;
- `packet_index`;
- `rssi`;
- `rate`;
- `channel`;
- `csi_len`;
- `flags`;
- vetores `imag` e `real` em `int16`.

A leitura de arquivos versão 1 continua suportada, mas os campos que não existiam nessa versão recebem valores de compatibilidade.

## Conversão para CSV

O binário é o formato bruto oficial. CSV deve ser usado apenas para inspeção:

```powershell
python Tools/csi/bin_to_csv.py arquivo.bin arquivo.csv
```

## Organização das ferramentas

### `acquisition/`

- `gui/csi_parser.py`: parser incremental CSI2 e diagnósticos;
- `gui/csi_viewer.py`: visualização e coleta binária;
- `check_dataset.py`: verificações de integridade do dataset;
- `analysis_dataset.py`: análises gerais das coletas;
- `cli/`: loggers anteriores e ferramentas de depuração.

### `csi/`

- `csi_binary_io.py`: leitura e escrita CSIBIN1 v1/v2;
- `csi_packet.py`: representação interna de pacote;
- `bin_to_csv.py`: conversão auxiliar.

### `preprocessing/`

Fluxo numerado atual:

```text
00_diagnose_dataset_packets.py
01_process_dataset.py
02_extract_features.py
03_subcarrier_variance_diagnostics.py
04_correlation_threshold_diagnostics.py
```

Módulos auxiliares implementam:

- amplitude a partir de I/Q;
- limpeza e filtragem de Hampel;
- média móvel;
- normalização z-score;
- correlação entre subportadoras;
- janelas deslizantes;
- extração e seleção de features.

Os parâmetros ficam centralizados em:

```text
Tools/preprocessing/pipeline_parameters.json
```

### `training/`

Os scripts numerados realizam, entre outras tarefas:

- Fisher Score e seleção Top-K;
- treinamento da árvore de decisão;
- análise das features selecionadas;
- holdout estratificado e holdout por arquivo;
- tuning de profundidade, divisão mínima e Top-K;
- diagnóstico binário;
- validação hierárquica;
- análise específica de `static_presence` versus `movement`.

A separação por arquivo deve ser preferida para evitar que janelas do mesmo arquivo apareçam simultaneamente em treino e teste.

Na branch `research/literature-guided-pipeline-review`, os experimentos adicionais com Regressão Logística, Gradient Boosting e XGBoost usam `scikit-learn` e `xgboost`. O candidato provisório selecionado antes da nova coleta é Gradient Boosting com Top-K 126, 20 estimadores, profundidade 3 e `learning_rate=0.1`; ele não deve ser tratado como modelo final até a repetição dos testes com o Dataset v2.

### `classification/`

`decision_tree.py` contém uma árvore de decisão própria, sem dependência obrigatória de `scikit-learn` na inferência. Essa escolha facilita a inspeção da estrutura e a futura tradução do modelo para execução embarcada.

### `real_time/`

`01_realtime_inference.py` concentra o fluxo de inferência em tempo real. Os parâmetros de normalização, subportadoras selecionadas, features e modelo devem ser os mesmos utilizados no treinamento.

## Pipeline conceitual

```text
raw_bin
    ↓
leitura dos pacotes
    ↓
amplitude
    ↓
limpeza
    ↓
Hampel
    ↓
média móvel
    ↓
z-score
    ↓
remoção de redundância
    ↓
janelas deslizantes
    ↓
features
    ↓
Fisher Score / Top-K
    ↓
classificador
    ↓
validação por arquivo e por quadrante
```

## Observações

- preserve sempre os arquivos binários brutos;
- não misture automaticamente arquivos v1 e v2 em um mesmo experimento;
- coletas antigas tinham frequência efetiva diferente e exigem reavaliação do tamanho da janela;
- a janela deve ser definida em segundos e convertida para pacotes conforme a taxa real;
- arquivos em `datasets/processed` são regeneráveis;
- resultados finais em `datasets/results` fazem parte do histórico experimental do TCC;
- não versione `.venv`, `build`, `__pycache__`, `.pyc` ou coletas temporárias.
