# STA_CSI_receiver

Firmware do ESP32-S3 que se conecta ao `AP_controller`, recebe o tráfego UDP e captura os dados CSI associados aos quadros unicast controlados.

A implementação atual não imprime o vetor CSI como texto. Os dados são enviados em um protocolo binário chamado **CSI2**, a **921600 baud**, para reduzir a carga da callback e sustentar aproximadamente 50 amostras por segundo.

## Fluxo interno

```text
Wi-Fi STA conecta ao AP
    ↓
power saving desabilitado
    ↓
BSSID do AP e MAC do STA registrados
    ↓
CSI habilitado
    ↓
callback recebe um quadro válido
    ↓
filtro AP → STA
    ↓
cópia imediata para fila FreeRTOS
    ↓
tarefa de saída serializa o frame
    ↓
UART0 a 921600 baud
    ↓
parser Python CSI2
```

## Componentes

| Componente | Responsabilidade |
|---|---|
| `wifi_manager.c` | conexão ao AP, eventos Wi-Fi e desativação de power saving |
| `udp_receiver.c` | recepção dos pacotes destinados à porta UDP |
| `csi_manager.c` | filtro, fila, protocolo CSI2, UART e estatísticas |
| `main.c` | inicialização geral do firmware |

## Callback CSI

A callback roda no contexto da tarefa Wi-Fi. Por isso ela executa apenas operações curtas:

1. valida ponteiros e tamanho;
2. aceita somente quadros cujo MAC de origem é o BSSID do AP e cujo destino é o próprio STA;
3. copia os metadados e o buffer CSI para uma estrutura local;
4. tenta inserir a estrutura em uma fila FreeRTOS sem bloqueio.

A conversão para bytes e a escrita na UART são realizadas por uma tarefa separada. Isso evita o gargalo provocado pela antiga impressão textual de centenas de valores com `printf`.

## Configuração atual

| Parâmetro | Valor |
|---|---:|
| UART | `UART_NUM_0` |
| Baud rate | `921600` |
| Buffer RX UART | 1024 bytes |
| Buffer TX UART | 16384 bytes |
| Capacidade máxima CSI | 384 inteiros |
| Fila CSI | 64 amostras |
| Intervalo das estatísticas | 1 s |
| Power saving | desabilitado (`WIFI_PS_NONE`) |

Com o AP em HT20, o valor esperado atualmente é:

```text
csi_len = 256
num_subcarriers = 128
bandwidth = 20 MHz
```

O MCS pode variar, por exemplo entre 6 e 7, sem alterar o tamanho do vetor.

## Protocolo serial CSI2

Todos os frames usam little-endian.

### Cabeçalho comum

| Campo | Tamanho | Descrição |
|---|---:|---|
| Magic | 4 bytes | `CSI2` |
| Versão | 1 byte | atualmente `1` |
| Tipo | 1 byte | `1` para amostra, `2` para estatísticas |
| Tamanho do frame | 2 bytes | inclui cabeçalho, payload e CRC |

### Frame de amostra

Após o cabeçalho comum:

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

O vetor CSI bruto é intercalado como:

```text
imag0, real0, imag1, real1, ...
```

### Bits de `flags`

| Bits | Conteúdo |
|---|---|
| 0 | largura: 0 = 20 MHz, 1 = 40 MHz |
| 1–2 | `sig_mode` |
| 3 | STBC |
| 4 | primeiro valor CSI inválido |
| 5–7 | MCS, três bits menos significativos |

### Frame de estatísticas

Enviado aproximadamente uma vez por segundo, contém:

- amostras recebidas;
- amostras inseridas na fila;
- amostras serializadas;
- descartes por fila cheia;
- frames inválidos;
- frames maiores que o limite;
- quantidade pendente na fila.

A GUI usa essas informações para exibir `ESP drops` e `ESP pending`.

## Integridade do transporte

O CRC cobre o frame a partir do campo de versão; a magic `CSI2` fica fora do cálculo. O parser no computador:

- procura a magic mesmo quando existem logs textuais misturados;
- valida versão e tamanho;
- valida o CRC;
- detecta saltos no campo `sequence`;
- se recupera de bytes corrompidos procurando o próximo frame válido.

## Compilar e gravar

```powershell
cd STA_CSI_receiver
idf.py set-target esp32s3
idf.py menuconfig
idf.py build
idf.py -p COM4 flash
```

A porta `COM4` é apenas o exemplo usado no ambiente de desenvolvimento.

O SSID e a senha devem coincidir com o AP. O canal é definido pelo AP durante a associação.

## Uso da porta serial

Durante a inicialização podem aparecer logs textuais. Depois que o CSI é habilitado, a UART passa a transportar frames binários a 921600 baud.

Portanto:

- use `idf.py monitor` somente para diagnóstico de inicialização;
- feche o monitor antes de abrir a GUI;
- não tente interpretar o fluxo CSI como texto;
- use `Tools/acquisition/gui/csi_viewer.py` para aquisição normal.

## Resultado esperado

Em uma coleta de 5 segundos com o AP saudável:

```text
aproximadamente 240–260 amostras
48–52 Hz
Sequence gaps: 0
CRC: 0
ESP drops: 0
csi_len: 256
bandwidth: 20 MHz
```

A ausência de saltos de sequência comprova que os frames aceitos pela callback chegaram ao computador sem perdas detectadas no caminho monitorado.
