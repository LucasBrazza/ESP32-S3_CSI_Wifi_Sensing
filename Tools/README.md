# Tools

Aplicação e ferramentas Python do sistema ESP32-S3 CSI Wi-Fi Sensing.

A pasta concentra aquisição, armazenamento, pré-processamento, treinamento, validação e inferência contínua.

## Uso normal

Na raiz do repositório:

```powershell
.\run_app.bat
```

O programa abre em tela cheia. Não é necessário executar cada script manualmente.

Menu disponível:

```text
Aquisição do Dataset
Detecção Realtime
Treinar Modelo
Resultados e Gravações
Verificar instalação
Encerrar
```

A tecla `Esc` retorna de uma tela auxiliar ao menu principal.

## Estrutura

```text
Tools/
├── app/
│   └── gui_menu.py
├── acquisition/
│   ├── gui/
│   │   ├── csi_viewer.py
│   │   ├── csi_parser.py
│   │   └── requirements.txt
│   └── ferramentas auxiliares
├── csi/
│   └── csi_binary_io.py
├── preprocessing/
│   └── filtros, janelas e features
├── training/
│   ├── 20_retrain_dataset_v2.py
│   └── dataset_v2_training_config.json
├── realtime/
│   ├── 01_realtime_inference.py
│   ├── 02_tune_state_machine.py
│   ├── 03_realtime_gui.py
│   ├── realtime_inference_engine.py
│   ├── temporal_state_machine.py
│   └── state_machine_config_candidate_v4.json
└── datasets/
    ├── processed/
    ├── results/
    └── realtime_runs/
```

## Ambiente Python

O ambiente validado utiliza Python 3.11.

```powershell
py -3.11 -m venv .venv

.\.venv\Scripts\python.exe -m pip install --upgrade pip

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

O arquivo `Tools/acquisition/gui/requirements.txt` contém apenas as dependências mínimas da interface de aquisição.

## Menu principal

`Tools/app/gui_menu.py` é o ponto de entrada da aplicação.

Ele:

- verifica a existência do modelo e das configurações;
- abre a aquisição em tela cheia;
- abre a detecção realtime em tela cheia;
- executa o treinamento em processo separado;
- permite abrir pastas de resultados;
- verifica bibliotecas e portas seriais;
- mantém apenas uma tela auxiliar ativa por vez.

## Aquisição do Dataset

A tela de aquisição usa:

```text
SerialReader
    ↓
CSIFrameParser
    ↓
eventos sample e stats
    ↓
visualização
    ↓
CSIBIN1 versão 2
```

Configuração atual:

| Campo | Valor |
|---|---|
| Baud | `921600` |
| Classes | `empty`, `static_presence`, `movement` |
| Duração padrão | 5 s |
| Offset | atraso antes de iniciar |
| Organização | sessão, quadrante e classe |

Diagnósticos apresentados:

- taxa de pacotes;
- falhas de sequência;
- erros CRC;
- descartes no computador;
- descartes no ESP32;
- itens pendentes na fila do ESP32.

Uma coleta deve ser aceita somente quando a taxa estiver próxima de 50 Hz e os contadores de erro e descarte estiverem em zero.

## Formato CSIBIN1

`Tools/csi/csi_binary_io.py` mantém o formato binário do dataset.

A versão 2 preserva:

- rótulo;
- timestamp do computador;
- timestamp alinhado à captura;
- timestamp do ESP32;
- sequência;
- índice do pacote;
- RSSI;
- rate;
- canal;
- tamanho;
- flags;
- vetores imaginário e real.

A leitura da versão 1 continua disponível para compatibilidade.

## Dataset v2

A organização recomendada é:

```text
Tools/datasets/raw_v2/
└── session_XX/
    ├── quad1/
    │   └── raw_bin/
    ├── quad2/
    ├── quad3/
    ├── quad4/
    └── quad5/
```

As três classes devem ser coletadas em aquisições independentes. O arquivo de origem é mantido como grupo durante a divisão de treino e teste.

## Pré-processamento

```text
pacotes binários
    ↓
matriz de amplitude
    ↓
remoção de colunas não informativas
    ↓
Hampel
    ↓
média móvel
    ↓
z-score
    ↓
correlação
    ↓
janelas
    ↓
11 descritores por subportadora
    ↓
Fisher / Top-K
```

Os parâmetros estatísticos são ajustados somente no conjunto de treinamento e exportados para a inferência.

## Treinamento consolidado

O menu executa:

```powershell
.\.venv\Scripts\python.exe -m Tools.training.20_retrain_dataset_v2
```

Configuração:

```text
Tools/training/dataset_v2_training_config.json
```

O pipeline:

- valida os arquivos;
- identifica sessão, quadrante e classe;
- separa treino e teste por arquivo;
- compara janelas, correlação, orçamento de features e classificadores;
- treina o candidato final;
- exporta tabelas, figuras, relatório e artefatos realtime.

Candidato atual:

```text
Extra Trees
100 árvores
correlação 0,95
janela 2,0 s
passo 0,5 s
160 features
97 subportadoras
```

## Detecção realtime

A tela realtime usa:

```text
CSI2 serial
    ↓
buffer incremental
    ↓
mesmo pré-processamento do treinamento
    ↓
Extra Trees
    ↓
probabilidades
    ↓
máquina de estados temporal
    ↓
estado final e TTS
```

Artefatos:

```text
Tools/datasets/processed/realtime_model_extra_trees.joblib
Tools/datasets/processed/realtime_pipeline_config_extra_trees.json
Tools/realtime/state_machine_config_candidate_v4.json
```

O motor espera:

```text
128 subportadoras de entrada
100 pacotes por janela
25 pacotes por passo
112 pacotes no buffer inicial
160 features selecionadas
```

## Calibração opcional

A detecção realtime utiliza diretamente os parâmetros exportados pelo treinamento. Médias, desvios, subportadoras e features não são recalculados na inicialização.

A opção **Calibração opcional** do menu executa apenas uma verificação de referência:

1. abre a serial;
2. preenche o buffer;
3. avalia algumas janelas;
4. registra classes e probabilidades;
5. salva um resumo na pasta da execução.

Essa verificação não aprova ou rejeita o modelo, não altera os artefatos e não é necessária para iniciar o realtime.

## Máquina de estados

A configuração v4 selecionada usa:

```text
sem suavização adicional
1 confirmação para transições normais
1 confirmação para transições diretas
limiar normal 0,0
limiar direto 0,6
estado inicial empty
```

O TTS anuncia apenas alterações aceitas no estado final.

## Saídas realtime

```text
Tools/datasets/realtime_runs/run_AAAAMMDD_HHMMSS/
├── calibration.json
├── metadata.json
├── raw_stream.bin
└── realtime_predictions.csv
```

O rótulo real pode ser informado manualmente durante uma execução para permitir avaliação posterior, mas não influencia a classificação.

## Comandos de desenvolvimento

Verificar sintaxe:

```powershell
.\.venv\Scripts\python.exe -m py_compile `
    Tools/app/gui_menu.py `
    Tools/acquisition/gui/csi_viewer.py `
    Tools/realtime/03_realtime_gui.py `
    Tools/realtime/realtime_inference_engine.py
```

Abrir somente a aquisição:

```powershell
.\.venv\Scripts\python.exe Tools/acquisition/gui/csi_viewer.py
```

Abrir somente o realtime:

```powershell
.\.venv\Scripts\python.exe -m Tools.realtime.03_realtime_gui
```

Esses comandos são destinados a desenvolvimento e diagnóstico. No uso normal, execute apenas `run_app.bat`.

## Arquivos locais

Não devem ser versionados:

- `.venv`;
- `__pycache__`;
- modelos temporários;
- coletas de teste;
- execuções realtime;
- resultados intermediários;
- arquivos `.pyc`.

Os arquivos finais de relatório, tabelas e figuras podem ser preservados em `Tools/datasets/results`.
