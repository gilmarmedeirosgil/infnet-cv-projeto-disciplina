## A4.2 — Transfer Learning para Análise de Tráfego Urbano

### Contexto

O sistema analisado usa uma ResNet-50 (He et al., 2016) pré-treinada no ImageNet (Deng et al., 2009), ajustada com 800 frames rotulados de 12 câmeras urbanas em três classes: *livre*, *moderado* e *congestionado*. Obteve 78% de accuracy na validação, foi implantado e passou a falhar de forma sistemática em chuva, à noite e em câmeras com ângulos que não estavam no treino.

O argumento que justificou o projeto ("o ImageNet reconhece objetos, veículos são objetos, logo as features servem") junta duas tarefas diferentes. O ImageNet treina **reconhecimento de um objeto dominante, em geral centrado, em fotos diurnas feitas por pessoas**. Classificar fluxo pede outra coisa: estimar **densidade e ocupação** de faixas numa cena ampla, com dezenas de veículos pequenos e parcialmente ocultos, e interpretar o **contexto espacial** (quais faixas, que sentido, onde fica a retenção). A mesma avenida cheia de carros pode ser "congestionado" (carros parados) ou "moderado" (carros andando juntos). A diferença está no movimento, e um frame isolado não mostra movimento. Yosinski et al. (2014) mostraram que as camadas finais de uma CNN são as mais específicas da tarefa de origem e as que transferem pior quando as tarefas se afastam.

A referência indicada no enunciado (Meegle) trata dos desafios gerais de transfer learning em análise de tráfego. Esta análise se apoia na literatura citada abaixo.

### Problemas identificados

#### P1. *Domain shift* e *task shift* do ImageNet para o fluxo

**Descrição técnica.** As features de alto nível da ResNet-50 codificam "o que é o objeto". A tarefa exige "quantos, onde e se estão parados". Com 800 imagens, o fine-tuning completo dos ~25 milhões de parâmetros tende a reaproveitar os atalhos mais fáceis que distinguem as classes no treino (iluminação, horário, a própria câmera) em vez de aprender densidade. É o *shortcut learning* (Geirhos et al., 2020), transposto dos "5 problemas técnicos" discutidos na Aula 7 (visão biomédica).

**Impacto operacional.** O modelo acerta enquanto o atalho acompanha o rótulo (por exemplo, "câmera X no fim da tarde = congestionado") e erra quando a correlação se quebra. O operador passa a confiar num indicador que não mede tráfego.

**Como eu abordaria.** (i) Reformular a saída como uma grandeza física: estimar **contagem ou ocupação por faixa** (detector de veículos pré-treinado em dados de tráfego, ou mapa de densidade, como em Zhang et al., 2017) e derivar a classe por limiares. (ii) Se a classificação direta for mantida, comparar *feature extraction* (backbone congelado) com fine-tuning parcial dos últimos blocos, porque com 800 imagens o congelamento reduz o sobreajuste ao fundo da câmera. (iii) Usar Grad-CAM para verificar se a atenção cai sobre faixas e veículos ou sobre céu, prédios e o carimbo de horário.

#### P2. Poucos dados e vazamento entre frames e câmeras no split

**Descrição técnica.** São 800 frames de 12 câmeras, cerca de 67 por câmera. Frames consecutivos da mesma câmera são quase idênticos. Se o split foi aleatório por frame, a validação contém "gêmeos" das imagens de treino, com o mesmo fundo, a mesma iluminação e os mesmos veículos. Os 78% medem então **memorização da cena**, não generalização. É o análogo do vazamento por paciente da Aula 7. Beery et al. (2018) documentaram esse efeito em câmeras fixas (armadilhas fotográficas): o desempenho cai de forma acentuada em locais nunca vistos.

**Impacto operacional.** A decisão de implantar foi tomada com uma estimativa otimista. A accuracy real em câmeras novas é desconhecida e provavelmente bem menor. Foi isso que apareceu em produção.

**Como eu abordaria.** Refazer a avaliação com **split *leave-camera-out*** (validação cruzada agrupada por câmera, *GroupKFold*) e, dentro de cada câmera, blocos temporais separados (dias diferentes, não minutos adjacentes). Deduplicar frames quase idênticos (hash perceptual) antes do split. Com 12 câmeras e ~67 frames cada, o *leave-one-camera-out* gera *folds* pequenos e de alta variância. Por isso o resultado deve ser reportado como **média ± desvio entre *folds***, nunca como um número único. A cifra que vale para a decisão de implantação é a de câmeras não vistas.

#### P3. Cobertura insuficiente de condições (chuva, noite, ângulos)

**Descrição técnica.** A distribuição de entrada em produção difere da de treino (*covariate shift*): faróis e reflexos no asfalto molhado, ruído do sensor à noite, gotas na lente, perspectivas e alturas de montagem diferentes. CNNs treinadas em ImageNet são reconhecidamente frágeis a corrupções como ruído, desfoque, neblina, neve e baixo contraste (Hendrycks & Dietterich, 2019).

**Impacto operacional.** As falhas se concentram nos horários e condições em que o trânsito é mais crítico: pico noturno, chuva forte. Se o sinal alimenta o controle semafórico, um congestionamento não detectado mantém planos semafóricos e rotas recomendadas inadequados e atrasa o despacho de agentes em incidentes.

**Como eu abordaria.** (i) **Coleta direcionada** de frames noturnos, com chuva e de novas câmeras, com rotulação priorizada pelos erros observados. (ii) **Augmentation fotométrica e climática** coerente com o domínio: brilho, gama e contraste, ruído, desfoque de movimento, chuva e neblina sintéticas. Evitar *flips* verticais e rotações grandes, que não ocorrem em câmera fixa; perspectiva leve simula variação de ângulo. (iii) **Adaptação de domínio no nível de pixel com CycleGAN** (Zhu et al., 2017; Aula 7): traduzir frames diurnos rotulados para "noite" e "chuva", preservando a estrutura da cena pela consistência cíclica e herdando o rótulo. A ressalva da Aula 7 vale aqui: o gerador pode **alucinar** ou apagar veículos, então as amostras sintéticas só entram no treino, e o ganho é medido num teste **100% real** das condições-alvo. Alternativa no espaço de features: adaptação adversarial (Ganin & Lempitsky, 2015) com frames não rotulados das câmeras novas. (iv) **Avaliação estratificada** por condição e por câmera.

#### P4. Frame único, sem informação temporal

**Descrição técnica.** Congestionamento é um estado dinâmico, definido por velocidade baixa e fila persistente. Um frame não distingue "parado no sinal vermelho" de "parado por retenção", nem "denso e fluindo" de "denso e travado".

**Impacto operacional.** Se o sinal alimenta o controle semafórico, haverá alarmes falsos a cada ciclo (fila momentânea classificada como congestionamento) e oscilação da classe de um minuto para o outro, o que torna o sinal inútil para controle adaptativo.

**Como eu abordaria.** Usar **sequências curtas** (8 a 16 frames), com optical flow como canal adicional, uma CNN 3D inflada a partir de pesos 2D (Carreira & Zisserman, 2017) ou um *pipeline* **detector + rastreamento** (SORT; Bewley et al., 2016) que produza contagem e velocidade média por faixa. Na saída, suavizar a decisão (janela deslizante ou histerese).

#### P5. Rótulos subjetivos, classes desbalanceadas e métrica única

**Descrição técnica.** A fronteira entre *moderado* e *congestionado* depende de quem rotulou, e cada câmera tem capacidade diferente. Sem critério escrito, o ruído de rótulo se concentra nas classes vizinhas. A classe *congestionado* é provavelmente minoritária, e 78% de accuracy pode esconder recall baixo justamente nela (o "paradoxo da acurácia" da Aula 7). Se, por hipótese, 60% dos frames fossem *livre*, um classificador constante já faria 60%. Além disso, as classes são **ordinais**, e a accuracy trata todos os erros como iguais.

**Impacto operacional.** O erro é assimétrico e depende da distância: classificar *congestionado* como *livre* (nenhuma intervenção) é muito pior do que confundi-lo com *moderado*, e ambos são piores que uma intervenção desnecessária.

**Como eu abordaria.** (i) Definir **critério objetivo** de rótulo por ocupação de faixa e/ou velocidade média, calibrado por câmera, com protocolo escrito e medição de concordância por **kappa ponderado** (ou kappa de Fleiss, se houver mais de dois anotadores). (ii) Reportar **matriz de confusão, recall e F1 por classe e por condição**, macro-F1 e uma métrica ordinal (MAE sobre as classes ou kappa ponderado entre predição e rótulo). (iii) Treinar com perda que respeite a ordem (regressão ordinal ou matriz de custo em que *livre↔congestionado* pesa mais que erros entre vizinhas), tratar o desbalanceamento com ponderação e fixar meta explícita de recall para *congestionado*.

#### P6. Implantação sem monitoramento de *drift* nem de incerteza

**Descrição técnica.** O modelo foi para produção sem detecção de mudança de distribuição, sem limiar de confiança e sem *fallback*. As probabilidades softmax de CNNs modernas costumam ser mal calibradas (Guo et al., 2017), e o sistema erra com alta confiança fora da distribuição. Mudanças físicas (câmera reposicionada, obra, lente suja, estação do ano) alteram a entrada sem aviso.

**Impacto operacional.** As falhas foram descobertas pelos efeitos no trânsito, não por um alerta. Não há como saber quais câmeras estão degradadas, e o erro pode persistir por semanas.

**Como eu abordaria.** (i) **Calibração** (*temperature scaling*) e abstenção: abaixo de um limiar de confiança, a câmera é marcada como "indeterminada" e o sistema volta ao plano semafórico padrão ou a outra fonte de dados. (ii) **Monitoramento por câmera** da distribuição de entrada (brilho, estatísticas de embeddings) e de saída (proporção de classes), com testes de *drift* (Rabanser et al., 2019). (iii) Auditoria amostral com rotulação humana, **retreino** com os casos difíceis e *shadow deployment* antes da troca de modelo (Sculley et al., 2015).

### Riscos do transfer learning ImageNet → classificação de fluxo

1. **Viés do domínio de origem.** O ImageNet é composto de fotos diurnas, bem expostas e centradas no objeto. O pré-treino dá pouca robustez prévia a noite, chuva ou vista oblíqua e fixa estatísticas de baixo nível (cor, textura) que não valem nessas condições.
2. **Convergência rápida não é adequação.** Um backbone forte atinge accuracy alta em poucas épocas e dá aparência de solução pronta. Em alguns domínios, porém, o pré-treino ImageNet acelera a convergência mais do que melhora o resultado final (Kornblith et al., 2019; He, Girshick & Dollár, 2019).
3. **Resolução e escala.** A entrada padrão de 224×224 reduz uma cena ampla, e os veículos distantes somem em poucos pixels. Soluções: entrada de resolução maior (a ResNet aceita por ter *global average pooling*), recorte de ROI por faixa e *tiling* da cena.
4. **Esquecimento das features úteis.** Um fine-tuning agressivo com 800 imagens pode destruir as features genéricas de baixo nível (bordas, texturas), que são a parte que realmente transfere (*catastrophic forgetting*). Taxas de aprendizado diferenciadas por camada e descongelamento gradual reduzem esse risco.

A conclusão não é abandonar o transfer learning, e sim escolher **a fonte e a tarefa certas**: backbone pré-treinado em detecção de veículos ou cenas de tráfego, saída física (contagem, ocupação, velocidade) e validação em câmeras e condições nunca vistas.

### Tabela-resumo

| # | Problema | Impacto operacional | Proposta | Prioridade |
|---|---|---|---|---|
| P2 | Vazamento entre frames/câmeras; poucos dados | 78% superestimado; decisão de implantar sem base | Split *leave-camera-out* + blocos temporais; deduplicação; média ± desvio | **Alta** (sem isso nenhuma melhoria é mensurável) |
| P3 | Cobertura de chuva/noite/ângulos (*covariate shift*) | Falhas nos horários mais críticos; semáforos e rotas mal ajustados | Coleta direcionada; augmentation climática; CycleGAN dia→noite/chuva validada em teste real; avaliação estratificada | **Alta** |
| P5 | Rótulos subjetivos, desbalanceamento, só accuracy | Congestionamento não detectado escondido na média | Critério por ocupação/velocidade; kappa ponderado; métricas por classe, condição e ordinais; custo assimétrico | **Alta** |
| P6 | Sem monitoramento de *drift* e incerteza | Erros silenciosos; sem *fallback* | Calibração + abstenção; *drift* por câmera; auditoria e retreino; *shadow deployment* | **Alta** |
| P1 | *Domain/task shift* ImageNet → densidade | Modelo aprende atalhos (câmera, horário) | Reformular como contagem/ocupação; backbone de tráfego; congelamento parcial; Grad-CAM | Média |
| P4 | Frame único, sem tempo | Alarmes falsos por fila de semáforo; classe oscilante | Sequências + optical flow/3D-CNN ou detector + rastreamento; suavização temporal | Média |

A ordem reflete dependências: P2 e P5 definem **como medir**, e sem isso não se sabe se P1, P3 ou P4 melhoraram alguma coisa. P6 protege a operação enquanto o modelo é refeito.

### Referências

- Beery, S., Van Horn, G., & Perona, P. (2018). Recognition in Terra Incognita. *ECCV*.
- Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016). Simple Online and Realtime Tracking. *IEEE ICIP*.
- Carreira, J., & Zisserman, A. (2017). Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset. *CVPR*.
- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A Large-Scale Hierarchical Image Database. *CVPR*.
- Ganin, Y., & Lempitsky, V. (2015). Unsupervised Domain Adaptation by Backpropagation. *ICML*.
- Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut Learning in Deep Neural Networks. *Nature Machine Intelligence*, 2, 665–673.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML*.
- He, K., Girshick, R., & Dollár, P. (2019). Rethinking ImageNet Pre-training. *ICCV*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Hendrycks, D., & Dietterich, T. (2019). Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. *ICLR*.
- Kornblith, S., Shlens, J., & Le, Q. V. (2019). Do Better ImageNet Models Transfer Better? *CVPR*.
- Rabanser, S., Günnemann, S., & Lipton, Z. C. (2019). Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift. *NeurIPS*.
- Sculley, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems. *NeurIPS*.
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014). How Transferable Are Features in Deep Neural Networks? *NeurIPS*.
- Zhang, S., Wu, G., Costeira, J. P., & Moura, J. M. F. (2017). FCN-rLSTM: Deep Spatio-Temporal Neural Networks for Vehicle Counting in City Cameras. *ICCV*.
- Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks. *ICCV*.
- Material da disciplina: Aulas 1 e 6 (*feature extraction* vs. fine-tuning; *catastrophic forgetting*), Aula 7 (GANs; "5 problemas técnicos"; CycleGAN para tradução entre domínios) e Aula 8 (síntese de imagens).
