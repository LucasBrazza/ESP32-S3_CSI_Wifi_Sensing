# AP_controller

Firmware do ESP32-S3 responsável por criar a rede Wi-Fi experimental e gerar tráfego UDP unicast controlado para o `STA_CSI_receiver`.

O AP não coleta CSI. Sua função é manter as condições de transmissão reproduzíveis para que o STA capture uma amostra CSI por pacote recebido.

## Configuração atual

| Parâmetro | Valor atual | Origem |
|---|---:|---|
| Modo Wi-Fi | Access Point | `main.c` |
| Largura de banda | 20 MHz (`WIFI_BW20`) | `main.c` |
| Intervalo UDP | 20 ms | `UDP_INTERVAL_MS` |
| Taxa solicitada | 50 pacotes/s | derivada do intervalo |
| IP de destino | `192.168.4.2` | `UDP_TARGET_IP` |
| Porta UDP | `3333` | `UDP_TARGET_PORT` |
| Canal padrão | 6 | `Kconfig.projbuild` |
| Máximo de clientes | 1 | `Kconfig.projbuild` |

SSID, senha, canal e número máximo de clientes são configuráveis por `idf.py menuconfig` no menu **Wi-Fi CSI Project Configuration**.

## Funcionamento

```text
Inicialização do NVS
    ↓
Wi-Fi em modo AP
    ↓
SSID, senha e canal configurados
    ↓
interface iniciada e fixada em HT20
    ↓
tarefa UDP periódica
    ↓
pacote enviado ao STA a cada 20 ms
```

A tarefa usa `xTaskDelayUntil()` para manter o período referenciado ao instante anterior. Assim, o tempo gasto na montagem e no envio do pacote não é somado continuamente ao intervalo de 20 ms.

O payload contém uma sequência e o timestamp interno do AP:

```text
CSI_PKT,<sequence>,<timestamp_us>
```

Esses campos são úteis para diagnóstico do emissor, mas o dataset principal é formado pelos metadados e pelo CSI capturados no STA.

## Por que o AP é fixado em 20 MHz?

Durante os testes em HT40, os quadros recebidos alternavam entre 20 e 40 MHz, produzindo vetores CSI com tamanhos diferentes (`256` e `384`). O uso de `WIFI_BW20` estabiliza a forma do dado na configuração atual:

```text
csi_len = 256 inteiros int8
          ↓
128 pares imag/real
          ↓
128 valores complexos
```

## Arquivos principais

| Arquivo | Responsabilidade |
|---|---|
| `main/main.c` | inicialização do AP, largura de banda e tarefa UDP |
| `main/Kconfig.projbuild` | SSID, senha, canal e número máximo de clientes |
| `sdkconfig` | configuração gerada pelo ESP-IDF |

## Compilar e gravar

Carregue primeiro o ambiente ESP-IDF 6.0. Depois, dentro da pasta do projeto:

```powershell
cd AP_controller
idf.py set-target esp32s3
idf.py menuconfig
idf.py build
idf.py -p COM3 flash
```

A porta `COM3` é apenas o exemplo usado no ambiente de desenvolvimento; ajuste conforme o computador.

## Monitorar

```powershell
idf.py -p COM3 monitor
```

O firmware informa as estatísticas aproximadamente uma vez por segundo. O resultado esperado é semelhante a:

```text
UDP stats: rate=50.00 pkt/s, sent=50, errors=0
```

Pequenas oscilações ao redor de 50 pacotes/s são aceitáveis. Erros contínuos ou ausência de envios indicam que o STA ainda não recebeu o IP esperado ou que há problema na conexão.

Para sair do monitor do ESP-IDF:

```text
Ctrl + ]
```

## Ordem de inicialização

1. ligue e valide o `AP_controller`;
2. ligue ou reinicie o `STA_CSI_receiver`;
3. aguarde o STA receber o IP `192.168.4.2`;
4. abra a GUI de aquisição na porta serial do STA.

## Critérios de validação

- AP criado no canal configurado;
- largura de banda informada como HT20;
- STA conectado;
- taxa UDP próxima de 50 pacotes/s;
- `errors=0` nas estatísticas do AP;
- no STA, apenas `bandwidth=20MHz` e `csi_len=256` na configuração atual.
