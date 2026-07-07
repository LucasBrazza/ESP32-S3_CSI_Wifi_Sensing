## Revisões futuras por etapa

### 1. Aquisição dos dados

Revisar depois que a coleta estiver estável.

* A taxa de pacotes está muito baixa para 60 s.
* Hoje você teve algo como 50–55 pacotes por coleta, o que dá menos de 1 pacote/s.
* Precisamos verificar se o AP está gerando tráfego suficiente.
* Conferir se o `csi_viewer` está salvando durante todo o tempo configurado.
* Conferir se há perda de pacotes ou se só está chegando CSI quando há algum tráfego.

### 2. Pré-processamento

Revisar antes de coletar dataset grande.

* A limpeza precisa ser padronizada entre treino e tempo real.
* Hoje decidimos limpar antes de montar janelas.
* Avaliar se pacotes inválidos devem ser descartados ou interpolados.
* Confirmar se Hampel e média móvel com os parâmetros atuais fazem sentido com uma taxa de pacotes maior.
* Revisar `window_size=20` e `step_size=5` quando soubermos a taxa real de amostragem.

### 3. Normalização

Revisar quando houver mais coletas.

* Hoje o z-score usa média/desvio calculados no conjunto de calibração.
* Precisamos separar treino/teste para evitar vazamento de informação.
* Verificar se `means` e `stds` continuam estáveis em dias diferentes.
* Tratar subportadoras com `std = 0`.

### 4. Correlação e redução de subportadoras

Revisar depois de coletar mais dados.

* O threshold `0.40` funcionou muito bem no dataset pequeno.
* Não considerar esse valor definitivo ainda.
* Reavaliar thresholds com dataset maior.
* Registrar tabela: threshold, subportadoras mantidas, acurácia, matriz de confusão.
* Confirmar se as subportadoras escolhidas são estáveis entre dias/coletas.

### 5. Janelas deslizantes

Revisar quando a taxa de pacotes estiver correta.

* Hoje `20 pacotes` não significa necessariamente `20 segundos` ou `0,2 s`; depende da taxa real.
* Definir janela em tempo real, por exemplo 2 s ou 3 s.
* Converter isso para número de pacotes.
* Avaliar overlap: `step_size=5` pode ser muito baixo ou alto dependendo da frequência.

### 6. Features

Revisar antes do modelo final.

* Hoje temos: média, desvio, mínimo, máximo, pico-a-pico e energia.
* Avaliar se todas são necessárias.
* Possivelmente remover features redundantes.
* Considerar adicionar variância, mediana, IQR ou energia normalizada.
* Evitar features caras se a ideia for embarcar.

### 7. Fisher Score

Revisar na validação final.

* Hoje o Fisher está sendo calculado antes da validação LOOCV, usando todo o conjunto.
* Isso pode deixar o resultado otimista.
* No teste final, Fisher deve ser calculado apenas no treino de cada fold.
* Também comparar com seleção fixa de subportadoras/features.

### 8. Árvore de classificação

Revisar depois de coletar mais dados.

* A árvore atual é ótima para embarcado, mas pode estar superajustada.
* Testar diferentes profundidades: 2, 3, 4, 5.
* Comparar árvore simples com Random Forest pequeno.
* Salvar a árvore em formato fácil de portar para C/MicroPython.

### 9. Simulação de tempo real

Revisar antes de ir para serial real.

* Hoje a simulação usa `.bin` já salvo.
* Próximo passo é testar com arquivos novos.
* Depois adaptar para stream serial real.
* Garantir que o buffer só receba pacotes válidos.
* Remover prints de debug ou trocar por modo verbose.

### 10. Máquina de estados

Implementar depois que o classificador estiver estável.

* Criar regra para evitar transições falsas.
* Exemplo: só mudar de estado após 3 janelas consecutivas.
* Medir atraso de detecção.
* Medir número de transições espúrias.

### 11. Refactor do código

Fazer depois que a simulação com arquivos novos funcionar.

* Separar melhor:

  * `csi_preprocessing.py`
  * `train_pipeline.py`
  * `realtime_pipeline.py`
  * `evaluation.py`
* Reduzir o tamanho do `csi_pipeline_core.py`.
* Evitar imports circulares.
* Atualizar README da pasta `Tools`.

### 12. Validação final do TCC

Fazer no final.

* Coletar mais amostras por classe.
* Separar treino/teste.
* Testar em dias diferentes.
* Testar com posições diferentes da pessoa.
* Reportar matriz de confusão, acurácia, precisão, recall e F1-score.
* Comparar custo computacional antes/depois da redução de subportadoras.
