# ESP32-S3 CSI Wi-Fi Sensing

Sistema experimental de detecção passiva de presença humana usando informações CSI (*Channel State Information*) obtidas por dois ESP32-S3.

O projeto classifica o estado do ambiente em três classes:

- `empty`: ambiente vazio;
- `static_presence`: presença humana sem movimento significativo;
- `movement`: presença humana em movimento.

A versão atual reúne aquisição binária, organização do Dataset v2, pré-processamento, treinamento, validação por arquivo, inferência contínua, máquina de estados temporal e aplicação gráfica em tela cheia.

## Visão geral

```text
ESP32-S3 AP
    │
    │ tráfego UDP controlado em HT20
    ▼
ESP32-S3 receptor
    │
    │ CSI2 binário pela UART a 921600 baud
    ▼
Aplicação Python
    │
    ├── aquisição do dataset
    ├── treinamento e avaliação
    └── detecção realtime
```

O conteúdo dos pacotes UDP não é utilizado diretamente na classificação. O tráfego mantém um fluxo controlado de recepções para que o ESP32-S3 receptor disponibilize amostras CSI.

## Estado consolidado

A configuração experimental atual utiliza:

| Parâmetro | Valor |
|---|---:|
| Dispositivos | 2 ESP32-S3 |
| Largura de banda | HT20 |
| Taxa esperada | aproximadamente 50 pacotes/s |
| Baud rate | 921600 |
| `csi_len` esperado | 256 inteiros |
| Subportadoras complexas | 128 |
| Classes | 3 |
| Janela | 2,0 s |
| Passo | 0,5 s |
| Classificador selecionado | Extra Trees |
| Árvores | 100 |
| Subportadoras utilizadas | 97 |
| Features selecionadas | 160 |

Na avaliação consolidada do Dataset v2, o candidato selecionado obteve macro F1 médio de aproximadamente `0,8402`. Na simulação de cenário contínuo utilizada durante o desenvolvimento, a classificação bruta atingiu acurácia de aproximadamente `0,8914`. A configuração temporal v4 elevou esse resultado para aproximadamente `0,9140` no mesmo cenário de desenvolvimento. Esse último valor não substitui uma validação independente em ambiente distinto.

## Estrutura principal

```text
ESP32-S3_CSI_Wifi_Sensing/
├── AP_controller/
│   └── firmware que cria a rede e gera o tráfego UDP
├── STA_CSI_receiver/
│   └── firmware que recebe os pacotes, captura CSI e envia frames CSI2
├── Tools/
│   ├── app/
│   │   └── menu principal da aplicação
│   ├── acquisition/
│   │   └── interface de aquisição, parser e diagnósticos
│   ├── csi/
│   │   └── leitura e escrita do formato CSIBIN1
│   ├── preprocessing/
│   │   └── preparação dos sinais e extração de características
│   ├── training/
│   │   └── treinamento, comparações e validações
│   ├── realtime/
│   │   └── inferência contínua e máquina de estados
│   └── datasets/
│       └── dados locais, artefatos e resultados
├── requirements.txt
├── requirements-lock.txt
├── run_app.bat
└── run_app.vbs
```

## Requisitos

- Windows 10 ou 11;
- Python 3.11;
- ESP-IDF compatível com ESP32-S3;
- duas placas ESP32-S3;
- cabos USB para gravação e comunicação serial.

O ambiente de firmware foi validado com ESP-IDF 6.0. As portas `COM3` e `COM4` apresentadas nos exemplos devem ser ajustadas conforme o computador utilizado.

## Instalação do ambiente Python

Na raiz do repositório:

```powershell
py -3.11 -m venv .venv

.\.venv\Scripts\python.exe -m pip install --upgrade pip

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Para reproduzir as versões exatas registradas no ambiente validado:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

## Execução do programa

O uso normal exige apenas:

```powershell
.\run_app.bat
```

Também é possível abrir `run_app.bat` por duplo clique.

O arquivo `run_app.vbs` inicia o mesmo programa sem manter a janela do terminal visível e pode ser usado como destino de um atalho do Windows.

A aplicação abre em tela cheia e apresenta:

- **Aquisição do Dataset**;
- **Detecção Realtime**;
- **Treinar Modelo**;
- **Resultados e Gravações**;
- **Verificar instalação**;
- **Encerrar**.

As interfaces auxiliares também são exibidas em tela cheia. A tecla `Esc` ou o botão **Voltar ao menu** retorna à tela inicial.

## Artefatos necessários para o realtime

A detecção contínua utiliza:

```text
Tools/datasets/processed/realtime_model_extra_trees.joblib
Tools/datasets/processed/realtime_pipeline_config_extra_trees.json
Tools/realtime/state_machine_config_candidate_v4.json
```

O arquivo `.joblib` é um artefato binário gerado localmente pelo treinamento e pode não estar incluído no repositório. A opção **Verificar instalação** informa quando algum desses arquivos está ausente.

## Calibração opcional

As médias, os desvios, as subportadoras e as features são exportados pelo treinamento. A detecção realtime usa esses valores diretamente e não recalibra o pipeline a cada inicialização.

O fluxo normal é:

```text
contagem regressiva
    ↓
abertura da porta serial
    ↓
preenchimento do buffer
    ↓
monitoramento contínuo
```

A opção **Calibração opcional** registra como o modelo treinado responde a uma condição de referência. Ela pode ser usada após alterar o ambiente, a posição dos dispositivos ou os artefatos do modelo. Essa verificação não modifica parâmetros e não bloqueia o realtime.

## Fluxo de aquisição

A interface de aquisição permite:

- selecionar a porta serial;
- acompanhar CSI, RSSI e diagnósticos;
- escolher sessão, quadrante e classe;
- executar coletas manuais;
- executar uma sequência programada com avisos sonoros;
- salvar os pacotes no formato binário CSIBIN1.

Classes aceitas:

```text
empty
static_presence
movement
```

Uma aquisição saudável de 5 segundos deve apresentar, aproximadamente:

```text
240 a 260 pacotes
48 a 52 Hz
Sequence gaps: 0
CRC: 0
PC drops: 0
ESP drops: 0
csi_len: 256
```

## Pipeline de processamento

```text
CSI complexo I/Q
    ↓
amplitude
    ↓
remoção de posições não informativas
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
extração de características
    ↓
seleção Fisher / Top-K
    ↓
classificador
    ↓
máquina de estados temporal
```

Médias, desvios, subportadoras e features são ajustados usando apenas os arquivos de treinamento e depois exportados para o realtime. A aplicação ao vivo não recalcula esses parâmetros.

## Treinamento

O menu **Treinar Modelo** executa o pipeline consolidado do Dataset v2 em processo separado.

O comando equivalente é:

```powershell
.\.venv\Scripts\python.exe -m Tools.training.20_retrain_dataset_v2
```

A configuração principal está em:

```text
Tools/training/dataset_v2_training_config.json
```

O protocolo mantém todas as janelas de um mesmo arquivo no mesmo conjunto, evitando que uma aquisição apareça simultaneamente no treino e no teste.

## Compilação dos firmwares

### AP

```powershell
cd AP_controller
idf.py set-target esp32s3
idf.py build
idf.py -p COM3 flash
```

### Receptor

```powershell
cd STA_CSI_receiver
idf.py set-target esp32s3
idf.py build
idf.py -p COM4 flash
```

O AP deve ser iniciado antes do receptor. Feche `idf.py monitor` na porta do receptor antes de abrir a aplicação, pois a UART passa a transportar frames binários CSI2.

## Arquivos gerados

As execuções realtime são armazenadas em:

```text
Tools/datasets/realtime_runs/run_AAAAMMDD_HHMMSS/
├── calibration.json
├── metadata.json
├── raw_stream.bin
└── realtime_predictions.csv
```

Os resultados do treinamento ficam em:

```text
Tools/datasets/results/
```

## Documentação dos módulos

- [`AP_controller/README.md`](AP_controller/README.md)
- [`STA_CSI_receiver/README.md`](STA_CSI_receiver/README.md)
- [`Tools/README.md`](Tools/README.md)

## Cuidados experimentais

- mantenha AP e receptor em posições fixas;
- preserve canal, largura de banda e orientação dos dispositivos;
- não use simultaneamente `idf.py monitor` e a aplicação na mesma porta;
- confira taxa, CRC, falhas de sequência e descartes antes de aceitar uma coleta;
- preserve os arquivos binários brutos;
- mantenha arquivos de uma mesma aquisição no mesmo conjunto durante a validação;
- trate resultados da máquina temporal obtidos no cenário de desenvolvimento como ajuste, não como validação independente.
