# STA_CSI_receiver

Firmware responsável por configurar um ESP32-S3 como estação Wi-Fi (*Station Mode*) e realizar a coleta de dados CSI (*Channel State Information*).

Este módulo é o principal responsável pela aquisição dos dados utilizados nos experimentos de Wi-Fi Sensing.

Enquanto o `AP_controller` cria a rede Wi-Fi e envia pacotes, o `STA_CSI_receiver` recebe esses pacotes e extrai os dados CSI associados a eles.

---

# Ideia geral

O ESP32-S3 receptor conecta na rede criada pelo `AP_controller`.

Após conectado, ele:

- recebe os pacotes UDP enviados pelo AP;
- habilita a coleta CSI;
- executa a callback CSI sempre que um pacote é recebido;
- envia os dados coletados pela serial para o computador.

Fluxo simplificado:

```text
AP_controller
        │
        │ envia pacotes UDP
        ▼
STA_CSI_receiver
        │
        │ coleta CSI
        ▼
Serial USB
        │
        ▼
Computador
```

---

# O que é CSI?

CSI (*Channel State Information*) representa informações do estado do canal Wi-Fi.

Esses dados carregam informações relacionadas ao ambiente físico, como:

- presença de pessoas;
- movimento;
- obstáculos;
- reflexões do sinal;
- variações do canal sem fio.

O ESP32-S3 permite acessar esses dados através da callback CSI disponibilizada pelo ESP-IDF.


---

# Como a coleta acontece

A coleta CSI depende da recepção de pacotes Wi-Fi.

Sempre que um pacote é recebido:

```text
Pacote Wi-Fi recebido
        ↓
ESP32-S3 executa callback CSI
        ↓
Dados CSI ficam disponíveis
        ↓
Firmware imprime os dados pela serial
```

Por isso o `AP_controller` envia pacotes continuamente.

Sem tráfego Wi-Fi, não há coleta CSI consistente.

---

# Saída serial

Os dados CSI são enviados pela serial em formato textual.

Formato conceitual:

```text
CSI,<esp_timestamp_us>,<rssi>,<rate>,<channel>,<len>,<csi_data...>
```

Exemplo:

```text
CSI,123456789,-51,11,6,384,12,-3,8,...
```

---

# Significado dos campos

| Campo | Descrição |
|---|---|
| `esp_timestamp_us` | Timestamp interno do ESP32-S3 em microssegundos. |
| `rssi` | Intensidade do sinal recebido. |
| `rate` | Taxa física do pacote Wi-Fi recebido. |
| `channel` | Canal Wi-Fi utilizado. |
| `len` | Tamanho do vetor CSI. |
| `csi_data` | Valores brutos do CSI. |

---

# Funcionamento interno

O fluxo principal do firmware é:

```text
Inicialização do ESP32-S3
        ↓
Configuração Wi-Fi em modo STA
        ↓
Conexão ao AP
        ↓
Recebimento de IP
        ↓
Inicialização do CSI
        ↓
Inicialização do receptor UDP
        ↓
Recepção contínua de pacotes
        ↓
Execução da callback CSI
        ↓
Impressão dos dados na serial
```

---

# Componentes principais do firmware

O firmware é dividido em três partes principais:

| Componente | Responsabilidade |
|---|---|
| Wi-Fi Manager | Realiza a conexão ao AP. |
| UDP Receiver | Recebe os pacotes enviados pelo AP. |
| CSI Manager | Configura e processa a coleta CSI. |

---

# Wi-Fi Manager

Responsável por:

- inicializar o Wi-Fi;
- configurar o ESP32-S3 em modo STA;
- conectar ao AP;
- monitorar eventos de conexão;
- iniciar o restante do fluxo após receber IP.

Fluxo simplificado:

```text
Inicializa Wi-Fi
        ↓
Conecta ao AP
        ↓
Recebe IP
        ↓
Inicia CSI e UDP
```

---

# UDP Receiver

Responsável por:

- abrir a porta UDP;
- receber os pacotes enviados pelo AP;
- manter o tráfego necessário para coleta CSI.

Os pacotes recebidos não são processados profundamente nesta etapa.

Seu objetivo principal é apenas gerar tráfego Wi-Fi contínuo.

---

# CSI Manager

Responsável por:

- configurar os parâmetros CSI;
- registrar a callback CSI;
- habilitar o CSI no ESP32-S3;
- processar os dados recebidos na callback;
- imprimir os dados pela serial.

Este é o componente mais importante do firmware.

---

# Callback CSI

A callback CSI é executada automaticamente pelo ESP-IDF sempre que um pacote Wi-Fi é recebido.

Fluxo:

```text
Pacote recebido
        ↓
ESP-IDF chama callback CSI
        ↓
Firmware lê os dados CSI
        ↓
Dados são enviados pela serial
```

A callback deve permanecer leve para evitar perda de pacotes e instabilidades.

---

# Como executar

Os comandos abaixo devem ser executados dentro da pasta do `STA_CSI_receiver`.

```bash
cd STA_CSI_receiver
```

Definir o alvo ESP32-S3:

```bash
idf.py set-target esp32s3
```

Abrir o menu de configuração:

```bash
idf.py menuconfig
```

Configure principalmente:

- SSID da rede criada pelo AP;
- senha da rede;
- demais parâmetros disponíveis.

Essas informações devem coincidir com as utilizadas no `AP_controller`.

Compilar o firmware:

```bash
idf.py build
```

Gravar o firmware no ESP32-S3:

```bash
idf.py flash -p COMx
```

Abrir o monitor serial:

```bash
idf.py monitor -p COMx
```

Também é possível gravar e abrir o monitor em um único comando:

```bash
idf.py flash -p COMx monitor
```

---


# Resultado esperado

Quando tudo estiver funcionando corretamente:

```text
STA conecta ao AP
        ↓
Pacotes UDP começam a chegar
        ↓
Callback CSI é acionada continuamente
        ↓
Dados CSI aparecem na serial
        ↓
Computador consegue salvar os dados
```

O funcionamento correto normalmente é identificado por:

- conexão Wi-Fi bem sucedida;
- recepção contínua de pacotes;
- linhas CSI aparecendo continuamente na serial.

---

# Observações importantes

- O AP deve estar iniciado antes do STA.
- O SSID e senha devem coincidir com os do AP.
- O canal Wi-Fi deve permanecer fixo durante os testes.
- A callback CSI deve permanecer leve.
- Logs excessivos podem reduzir a estabilidade da coleta.
- O computador será responsável pelo salvamento e processamento inicial dos dados.