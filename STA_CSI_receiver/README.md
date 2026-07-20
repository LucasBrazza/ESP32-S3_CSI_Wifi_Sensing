# STA_CSI_receiver

Firmware do ESP32-S3 que se conecta ao `AP_controller`, recebe o tráfego UDP e captura o CSI associado aos quadros controlados.

## Fluxo interno

```text
STA conecta ao AP
    ↓
power saving desabilitado
    ↓
BSSID do AP e MAC do STA registrados
    ↓
CSI habilitado
    ↓
callback recebe um quadro válido
    ↓
filtro AP para STA
    ↓
cópia para fila FreeRTOS
    ↓
tarefa serializa o frame
    ↓
UART0 a 921600 baud
    ↓
parser CSI2 no computador
```

## Componentes

| Arquivo | Responsabilidade |
|---|---|
| `wifi_manager.c` | conexão ao AP, eventos Wi-Fi e configuração de energia |
| `udp_receiver.c` | recepção dos pacotes UDP |
| `csi_manager.c` | filtro dos quadros, fila, CSI2, UART e estatísticas |
| `main.c` | inicialização geral |

## Callback CSI

A callback executa no contexto da pilha Wi-Fi e deve permanecer curta. Ela:

1. valida os ponteiros e o tamanho;
2. aceita apenas quadros com origem no BSSID do AP e destino no STA;
3. copia os metadados e o vetor CSI para uma estrutura local;
4. tenta inserir a estrutura em uma fila FreeRTOS sem bloqueio.

A conversão para bytes e a escrita na UART são executadas por outra tarefa. Essa separação evita bloquear a recepção Wi-Fi.

## Configuração atual

| Parâmetro | Valor |
|---|---:|
| UART | `UART_NUM_0` |
| Baud rate | `921600` |
| Buffer RX UART | 1024 bytes |
| Buffer TX UART | 16384 bytes |
| Capacidade máxima CSI | 384 inteiros |
| Fila CSI | 64 amostras |
| Estatísticas | aproximadamente 1 s |
| Power saving | `WIFI_PS_NONE` |

Com AP em HT20:

```text
csi_len = 256
num_subcarriers = 128
bandwidth = 20 MHz
```

O MCS pode variar sem modificar necessariamente o tamanho do vetor.

## Protocolo CSI2

Todos os campos numéricos usam little-endian.

### Cabeçalho comum

| Campo | Tamanho | Descrição |
|---|---:|---|
| Magic | 4 bytes | `CSI2` |
| Versão | 1 byte | atualmente `1` |
| Tipo | 1 byte | `1` para amostra, `2` para estatísticas |
| Tamanho | 2 bytes | frame completo |

### Frame de amostra

| Campo | Tamanho |
|---|---:|
| `sequence` | 4 bytes |
| `timestamp_us` | 8 bytes |
| `rssi` | 1 byte assinado |
| `rate` | 1 byte |
| `channel` | 1 byte |
| `flags` | 1 byte |
| `csi_len` | 2 bytes |
| CSI bruto | `csi_len` bytes `int8` |
| CRC-16/CCITT-FALSE | 2 bytes |

O vetor bruto é intercalado:

```text
imag0, real0, imag1, real1, ...
```

### Flags

| Bits | Conteúdo |
|---|---|
| 0 | largura de banda |
| 1 e 2 | `sig_mode` |
| 3 | STBC |
| 4 | primeiro valor CSI inválido |
| 5 a 7 | bits menos significativos do MCS |

### Estatísticas

O frame periódico de estatísticas informa:

- amostras recebidas;
- amostras colocadas na fila;
- amostras serializadas;
- descartes por fila cheia;
- frames inválidos;
- frames acima do limite;
- itens pendentes na fila.

A aplicação usa esses valores para exibir `ESP drops` e `ESP pending`.

## Integridade

O parser do computador:

- procura a magic `CSI2`;
- valida versão e tamanho;
- verifica CRC;
- detecta saltos de sequência;
- recupera a sincronização após bytes inválidos;
- separa automaticamente os pares imaginário e real.

## Compilar e gravar

```powershell
cd STA_CSI_receiver
idf.py set-target esp32s3
idf.py menuconfig
idf.py build
idf.py -p COM4 flash
```

A porta `COM4` é apenas um exemplo. SSID e senha devem coincidir com o AP.

## Uso da serial

Durante a inicialização podem aparecer logs textuais. Depois que o CSI é habilitado, a UART passa a transportar frames binários.

- use `idf.py monitor` apenas para diagnóstico de inicialização;
- feche o monitor antes de abrir a aplicação;
- não interprete o fluxo CSI como texto;
- utilize o menu principal do projeto para aquisição ou realtime.

## Resultado esperado

Em uma coleta de 5 segundos:

```text
240 a 260 amostras
48 a 52 Hz
Sequence gaps: 0
CRC: 0
ESP drops: 0
csi_len: 256
bandwidth: 20 MHz
```
