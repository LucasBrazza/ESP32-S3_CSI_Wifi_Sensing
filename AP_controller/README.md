# AP_controller

Firmware responsável por configurar um ESP32-S3 como ponto de acesso Wi-Fi, também chamado de AP (*Access Point*).

No fluxo deste projeto, o `AP_controller` não coleta CSI. Sua função é criar a rede Wi-Fi e gerar pacotes para que outro ESP32-S3, configurado como receptor, consiga capturar os dados CSI.

---

# Ideia geral

A coleta CSI depende da recepção de pacotes Wi-Fi.

Por isso, o projeto utiliza dois dispositivos:

```text
ESP32-S3 AP_controller
        │
        │ envia pacotes UDP
        ▼
ESP32-S3 STA_CSI_receiver
        │
        │ coleta CSI dos pacotes recebidos
        ▼
Computador
```

O `AP_controller` é o primeiro bloco desse fluxo.

Ele cria uma rede Wi-Fi própria e envia pacotes UDP continuamente para o receptor.

---

# O que este firmware faz

De forma simples, este firmware executa as seguintes etapas:

```text
1. Liga o ESP32-S3
2. Inicializa os recursos internos necessários
3. Configura o Wi-Fi em modo Access Point
4. Cria uma rede Wi-Fi
5. Aguarda o receptor conectar
6. Envia pacotes UDP continuamente
```

Esses pacotes UDP servem apenas para gerar tráfego Wi-Fi.

Eles não carregam dados importantes para análise neste momento.

---

# Por que enviar pacotes UDP?

O ESP32-S3 receptor só consegue coletar CSI quando recebe pacotes Wi-Fi.

Então, se não houver tráfego entre os dispositivos, a coleta CSI fica limitada ou irregular.

O envio UDP resolve isso criando um fluxo simples e constante de pacotes:

```text
Pacotes UDP constantes
        ↓
Recepção Wi-Fi constante no STA
        ↓
Callback CSI acionada
        ↓
Dados CSI disponíveis para coleta
```

---

# Relação com o receptor CSI

O `AP_controller` e o `STA_CSI_receiver` trabalham juntos.

O AP cria a rede e envia pacotes.

O STA conecta nessa rede e coleta CSI.

```text
AP_controller
   └── cria a rede e envia pacotes

STA_CSI_receiver
   └── conecta na rede e coleta CSI
```

Portanto, para reproduzir o experimento, o AP deve ser iniciado antes do receptor.

---

# Configurações principais

As configurações mais importantes deste firmware são:

| Configuração | Função |
|---|---|
| SSID | Nome da rede Wi-Fi criada pelo AP. |
| Senha | Senha usada pelo receptor para conectar. |
| Canal Wi-Fi | Canal usado no experimento. Deve ser mantido fixo. |
| IP do receptor | Endereço para onde os pacotes UDP são enviados. |
| Porta UDP | Porta usada para envio dos pacotes. |
| Intervalo de envio | Tempo entre um pacote UDP e outro. |

O canal Wi-Fi é especialmente importante, pois alterações no canal podem mudar o comportamento dos dados CSI.

---

# Como executar

Os comandos abaixo devem ser executados dentro da pasta do `AP_controller`.

```bash
cd AP_controller
```

---

# 1. Selecionar o alvo ESP32-S3

```bash
idf.py set-target esp32s3
```

Esse comando informa ao ESP-IDF que o projeto será compilado para o ESP32-S3.

---

# 2. Configurar o projeto

```bash
idf.py menuconfig
```

No menu de configuração, ajuste os parâmetros principais do AP, como:

- nome da rede Wi-Fi;
- senha;
- canal Wi-Fi;
- demais parâmetros disponíveis no projeto.

Depois de configurar, salve e saia do menu.

---

# 3. Compilar o firmware

```bash
idf.py build
```

Esse comando compila o projeto e gera o firmware que será gravado no ESP32-S3.

---

# 4. Gravar o firmware

Substitua `COMx` pela porta serial correta do seu ESP32-S3.

```bash
idf.py flash -p COMx
```

Exemplo no Windows:

```bash
idf.py flash -p COM4
```

---

# 5. Abrir o monitor serial

```bash
idf.py monitor -p COMx
```

Exemplo:

```bash
idf.py monitor -p COM4
```

Também é possível gravar e abrir o monitor em um único comando:

```bash
idf.py flash -p COMx monitor
```

---

# Ordem recomendada de execução

Para reproduzir o experimento, siga esta ordem:

```text
1. Grave e execute o AP_controller
2. Verifique se a rede Wi-Fi foi criada
3. Grave e execute o STA_CSI_receiver
4. Aguarde o STA conectar ao AP
5. Verifique se o AP começou a enviar pacotes UDP
6. Verifique no receptor se os dados CSI estão sendo impressos
7. Execute as ferramentas no computador para salvar os dados
```

---

# Resultado esperado

Quando o firmware estiver funcionando corretamente:

```text
O ESP32-S3 cria uma rede Wi-Fi
        ↓
O receptor conecta nessa rede
        ↓
O AP envia pacotes UDP continuamente
        ↓
O receptor passa a receber pacotes
        ↓
A coleta CSI pode ser realizada no STA
```

O sucesso do `AP_controller` é observado quando:

- a rede Wi-Fi aparece disponível;
- o receptor consegue se conectar;
- os pacotes UDP são enviados continuamente;
- o receptor passa a gerar dados CSI.

---

# Observações importantes

- Inicie o AP antes do receptor.
- Mantenha o canal Wi-Fi fixo durante os testes.
- Use o mesmo SSID e senha configurados no receptor.
- Confirme se o IP de destino do UDP corresponde ao IP do receptor.
- Evite alterar a taxa de envio UDP sem observar o impacto na coleta CSI.