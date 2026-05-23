# ESP32-S3 CSI Wi-Fi Sensing

Projeto para coleta e análise de dados CSI (*Channel State Information*) utilizando placas ESP32-S3 em uma arquitetura AP/STA.

O objetivo principal é construir uma base experimental para Wi-Fi Sensing, inicialmente com processamento no computador, mas mantendo a estrutura do projeto preparada para futuras etapas de pré-processamento e classificação embarcada.

---

# Objetivo do projeto

Este projeto tem como objetivo capturar dados CSI de sinais Wi-Fi usando ESP32-S3 para investigar aplicações de sensoriamento, como:

- detecção de presença;
- detecção de movimento;
- identificação de mudanças no ambiente;
- classificação de estados do ambiente.

A proposta inicial do sistema é:

1. configurar um ESP32-S3 como ponto de acesso Wi-Fi;
2. configurar outro ESP32-S3 como estação receptora;
3. gerar tráfego controlado entre os dispositivos;
4. coletar os dados CSI no receptor;
5. enviar os dados pela serial para o computador;
6. salvar os dados em arquivos;
7. analisar os dados posteriormente usando Python.

---

# Estrutura do projeto

```text
ESP32-S3_CSI_Wifi_Sensing/
│
├── AP_controller/
│   └── Firmware do ESP32-S3 responsável por criar a rede Wi-Fi e gerar tráfego.
│
├── STA_CSI_receiver/
│   └── Firmware do ESP32-S3 responsável por conectar ao AP e coletar CSI.
│
├── Tools/
│   └── Scripts auxiliares para coleta, conversão, visualização e análise dos dados.
│
├── README.md
└── .gitignore
```

---

# Visão geral do fluxo

O fluxo básico do projeto é dividido em três partes principais:

```text
AP_controller  --->  STA_CSI_receiver  --->  Tools
```

---

# 1. AP_controller

O `AP_controller` executa no primeiro ESP32-S3.

Ele é responsável por:

- criar uma rede Wi-Fi própria;
- manter SSID, senha e canal controlados;
- permitir a conexão do receptor;
- gerar tráfego UDP periódico.

Esse tráfego é necessário porque o receptor CSI precisa receber pacotes Wi-Fi para que o ESP32-S3 consiga disponibilizar informações CSI.

---

# 2. STA_CSI_receiver

O `STA_CSI_receiver` executa no segundo ESP32-S3.

Ele é responsável por:

- conectar à rede criada pelo `AP_controller`;
- habilitar a coleta CSI;
- receber pacotes UDP;
- executar a callback CSI;
- imprimir os dados CSI pela serial.

A saída serial contém metadados do pacote e o vetor CSI bruto.

Formato conceitual:

```text
CSI,<esp_timestamp_us>,<rssi>,<rate>,<channel>,<len>,<csi_data...>
```

---

# 3. Tools

A pasta `Tools` contém os scripts executados no computador.

Ela é responsável por:

- ler os dados enviados pela serial;
- adicionar rótulos às coletas;
- salvar os dados em arquivos;
- converter formatos quando necessário;
- permitir visualização futura;
- preparar os dados para pré-processamento e análise.

Nesta etapa inicial, o computador é usado para facilitar testes, validação e desenvolvimento do pipeline.

---

# Fluxo completo de aquisição

```text
1. Ligar o ESP32-S3 com o firmware AP_controller
   │
   └── O dispositivo cria a rede Wi-Fi experimental.

2. Ligar o ESP32-S3 com o firmware STA_CSI_receiver
   │
   └── O dispositivo conecta ao AP.

3. O AP envia pacotes UDP para o STA
   │
   └── Esses pacotes geram eventos de recepção no receptor.

4. O STA coleta CSI dos pacotes recebidos
   │
   └── A callback CSI captura os dados brutos.

5. O STA envia os dados pela serial
   │
   └── Cada pacote CSI é impresso em uma linha.

6. O computador executa scripts da pasta Tools
   │
   └── Os dados são salvos em arquivos de dataset.

7. Os dados salvos são analisados posteriormente
   │
   └── Pré-processamento, seleção de subportadoras, features e classificação.
```

---

# Organização por responsabilidade

| Pasta | Responsabilidade |
|---|---|
| `AP_controller/` | Controlar o ponto de acesso Wi-Fi e gerar tráfego para o experimento. |
| `STA_CSI_receiver/` | Conectar ao AP, capturar CSI e enviar os dados pela serial. |
| `Tools/` | Coletar, salvar, converter, visualizar e analisar os dados no computador. |

---

# Estado atual do projeto

O projeto está em fase de construção do pipeline de aquisição.

As etapas principais atualmente são:

- firmware AP funcional;
- firmware STA funcional;
- coleta CSI habilitada;
- envio dos dados pela serial;
- scripts iniciais para salvar dados no computador;
- organização inicial para datasets.

---


# Documentação dos módulos

Cada pasta principal possui ou deverá possuir seu próprio `README.md` com detalhes específicos:

```text
AP_controller/README.md
STA_CSI_receiver/README.md
Tools/README.md
```

Esses arquivos devem explicar com mais detalhes o funcionamento interno de cada parte do projeto.

---

# Requisitos gerais

Para utilizar este projeto são necessários:

- 2 placas ESP32-S3;
- ESP-IDF instalado;
- Python instalado no computador;
- cabo USB para gravação e leitura serial;
- ambiente configurado para execução dos scripts de coleta.

---

# Observações importantes

- O canal Wi-Fi deve ser mantido fixo durante os experimentos.
- O receptor CSI depende da recepção de pacotes para gerar amostras.
- A callback CSI deve permanecer leve.
- O processamento mais pesado deve ser feito inicialmente no computador.
- O formato CSV é útil para testes, mas o formato binário será mais adequado para coletas maiores.