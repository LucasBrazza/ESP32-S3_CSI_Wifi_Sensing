# AP_controller

Firmware do ESP32-S3 responsável por criar a rede Wi-Fi experimental e gerar o tráfego usado na obtenção das amostras CSI.

O AP não realiza classificação nem captura CSI. Sua função é manter uma comunicação controlada com o receptor.

## Fluxo

```text
AP_controller
    │
    │ UDP unicast periódico
    ▼
STA_CSI_receiver
    │
    │ captura CSI dos quadros recebidos
    ▼
Computador
```

## Responsabilidades

- configurar o ESP32-S3 no modo Access Point;
- criar a rede com SSID e senha conhecidos;
- manter o canal e a largura de banda definidos;
- aguardar a conexão do receptor;
- enviar pacotes UDP ao STA;
- registrar estatísticas de envio.

O conteúdo UDP não participa diretamente da classificação. Ele apenas cria eventos regulares de recepção no STA.

## Configuração experimental

| Parâmetro | Valor atual |
|---|---:|
| Alvo | ESP32-S3 |
| Modo Wi-Fi | Access Point |
| Largura de banda | HT20 |
| Intervalo UDP | aproximadamente 20 ms |
| Taxa esperada | aproximadamente 50 pacotes/s |
| Destino | endereço IP do STA |
| Transporte | UDP unicast |

SSID, senha, canal, IP e porta devem coincidir com a configuração do receptor.

## Componentes

A implementação é dividida em arquivos com responsabilidades específicas:

```text
main/
├── main.c
├── wifi_manager.c
├── wifi_manager.h
├── udp_sender.c
└── udp_sender.h
```

Os nomes podem variar conforme a revisão do firmware, mas o fluxo permanece dividido entre inicialização, configuração Wi-Fi e geração do tráfego.

## Compilar e gravar

Abra um terminal ESP-IDF e execute:

```powershell
cd AP_controller
idf.py set-target esp32s3
idf.py menuconfig
idf.py build
idf.py -p COM3 flash
```

A porta `COM3` é apenas um exemplo.

Para diagnóstico:

```powershell
idf.py -p COM3 monitor
```

## Ordem de inicialização

1. ligue ou reinicie o AP;
2. confirme que a rede foi criada;
3. ligue o receptor;
4. aguarde a conexão do STA;
5. confirme o início do envio UDP;
6. abra a aplicação no computador.

## Resultado esperado

Com o receptor conectado, o AP deve apresentar:

```text
aproximadamente 50 pacotes enviados por segundo
erros de envio próximos de zero
destino UDP correspondente ao STA
canal e largura de banda estáveis
```

## Cuidados

- não altere o canal durante uma sessão;
- mantenha a largura de banda em HT20 para reproduzir a configuração validada;
- não altere a taxa UDP sem reavaliar a taxa CSI e o tamanho das janelas;
- inicialize o AP antes do receptor;
- mantenha posição e orientação do dispositivo durante as coletas.
