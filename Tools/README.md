# Tools

A pasta `Tools` contém todas as ferramentas executadas no computador durante a aquisição, organização, visualização e armazenamento dos dados CSI coletados pelos ESP32-S3.

O fluxo principal do projeto funciona da seguinte forma:

```text
AP_controller
        │
        │ envia pacotes Wi-Fi
        ▼
STA_CSI_receiver
        │
        │ coleta CSI
        │
        │ envia dados pela serial
        ▼
Tools/acquisition
        │
        │ interpreta os dados recebidos
        ▼
Tools/csi
        │
        │ organiza e salva os pacotes
        ▼
Tools/datasets
```

---

# Estrutura CSI utilizada no projeto

Os dados enviados pelo ESP32-S3 possuem o seguinte formato:

```text
CSI,<timestamp_us>,<rssi>,<rate>,<channel>,<len>,imag0,real0,imag1,real1,...
```

Os primeiros campos representam os metadados do pacote:

| Campo | Descrição |
|---|---|
| `timestamp_us` | Timestamp interno do ESP32-S3. |
| `rssi` | Intensidade do sinal recebido. |
| `rate` | Taxa física do pacote Wi-Fi. |
| `channel` | Canal Wi-Fi utilizado. |
| `len` | Quantidade total de inteiros CSI. |

Após os metadados, os dados CSI são enviados em formato intercalado:

```text
imag0, real0,
imag1, real1,
imag2, real2,
...
```

Atualmente o projeto normalmente trabalha com:

```text
len = 384
```

Isso significa:

```text
384 inteiros
        ↓
192 pares imag/real
        ↓
192 subportadoras complexas
```

O pipeline oficial do projeto trabalha exclusivamente com:

```text
imag
real
metadata
```


---

# Organização das ferramentas

A pasta `acquisition/cli/` contém os loggers executados pelo terminal.

O arquivo `serial_logger_raw.py` é o logger principal do projeto. Sua função é abrir a serial do ESP32-S3, identificar linhas CSI válidas e salvar os dados brutos em arquivos binários.

O arquivo `serial_logger_parsed.py` realiza aquisição utilizando os dados já organizados em estruturas internas do projeto, permitindo validar o fluxo de parsing durante a coleta.

O arquivo `serial_logger_parsed_debug.py` é utilizado para depuração do pipeline de aquisição e validação detalhada dos pacotes CSI recebidos.

---

A pasta `acquisition/gui/` contém as ferramentas gráficas utilizadas durante a aquisição.

O arquivo `csi_parser.py` recebe as linhas CSI vindas da serial e transforma os dados textuais em estruturas organizadas contendo:

- metadata;
- imag;
- real.

Esse parser não realiza cálculos derivados.

O arquivo `csi_viewer.py` é a interface gráfica principal do projeto. Sua função é visualizar os sinais CSI em tempo real durante a coleta, permitindo validar estabilidade dos dados, comportamento do sinal e funcionamento geral do sistema.

O arquivo `requirements.txt` contém as dependências Python utilizadas pelas ferramentas gráficas.

---

A pasta `csi/` contém o núcleo responsável por representar, interpretar, converter e salvar os pacotes CSI.

O arquivo `csi_packet.py` define a estrutura utilizada para representar um pacote CSI dentro do projeto, padronizando os dados utilizados internamente.

O arquivo `csi_binary_io.py` é responsável pelo salvamento e leitura dos arquivos binários do projeto, realizando serialização e desserialização dos pacotes CSI.

O arquivo `bin_to_csv.py` converte arquivos binários para CSV, permitindo inspeção manual dos dados coletados.

---

A pasta `datasets/` contém os datasets gerados durante os experimentos.

A pasta `datasets/bin/` armazena os datasets binários originais, preservando a coleta exatamente como foi realizada.

A pasta `datasets/csv/` contém versões CSV dos datasets, utilizadas principalmente para validação, inspeção manual e compatibilidade com ferramentas externas.

---

A pasta `experiments/` contém ferramentas e scripts que não fazem parte do pipeline oficial do projeto.

A pasta `experiments/deprecated/` armazena arquivos antigos ou removidos do fluxo principal.

A pasta `experiments/features/` é destinada para estudos relacionados à extração de features e processamento derivado.

A pasta `experiments/subcarrier_analysis/` contém experimentos relacionados à análise e seleção de subportadoras.

Essas etapas não fazem parte do pipeline oficial neste momento.

---

# Fluxo oficial do projeto

O fluxo principal atualmente é:

```text
ESP32
        ↓
serial_logger_raw
        ↓
RAW BIN
        ↓
bin_to_csv
        ↓
RAW CSV
```

O formato binário é o formato principal do projeto.

O CSV é utilizado apenas como formato auxiliar para inspeção e análise.

---

# Execução das ferramentas

Todos os comandos abaixo devem ser executados dentro da pasta `Tools`.

```bash
cd Tools
```

Instalar dependências do viewer:

```bash
pip install -r acquisition/gui/requirements.txt
```

Executar o logger principal:

```bash
python acquisition/cli/serial_logger_raw.py
```

Executar interface gráfica:

```bash
python acquisition/gui/csi_viewer.py
```

Converter binário para CSV:

```bash
python csi/bin_to_csv.py input.bin output.csv
```

---

# Objetivo arquitetural

A arquitetura atual foi organizada para separar claramente:

```text
aquisição
↓
representação CSI
↓
armazenamento
↓
visualização
↓
experimentos
```

Essa separação facilita:

- manutenção;
- expansão do projeto;
- reprodutibilidade;
- evolução futura do pipeline.

---

# Observações importantes

- O dataset bruto deve sempre ser preservado.
- O formato binário é o formato principal do projeto.
- O CSV é utilizado apenas como formato auxiliar.
- O parser não deve calcular dados derivados.
- O processamento pesado deve ocorrer fora do pipeline oficial.
- O pipeline principal deve permanecer simples e reproduzível.