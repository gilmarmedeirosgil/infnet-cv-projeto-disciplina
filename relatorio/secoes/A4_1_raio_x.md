## A4.1 — Estudo de caso: triagem de COVID-19 em raio-X de tórax com cGAN para augmentation sintética

### Cenário e objetivo

Um grupo anterior treinou uma ResNet-18 do zero para triar radiografias de tórax em Normal / Pneumonia / COVID-19, com 1.200 imagens na proporção 7:2:1 (840/240/120), split 80/20, SGD com LR fixo 0,01 e 15 épocas, sem augmentation, e relatou só a accuracy (93% no treino, 61% na validação). Este notebook (1) diagnostica os problemas metodológicos desse projeto e o impacto clínico de cada um; (2) reproduz o baseline e constrói um pipeline corrigido; (3) treina uma **GAN condicional (cGAN)** para gerar radiografias COVID sintéticas, documentando a instabilidade do treino adversarial e sua mitigação; (4) mede, com várias seeds, se os sintéticos melhoram o **recall de COVID** num teste 100% real; (5) propõe um plano de melhoria com critério de adoção clínica. Notebook: `notebooks/A4_estudo_caso_raio_x.ipynb`, executado com "Executar tudo" numa T4 do Colab em **809,5 s (~13,5 min)**; VRAM de pico 3.274 MB.

### Diagnóstico do projeto anterior

| # | Problema | Impacto clínico |
|---|---|---|
| 1 | Desbalanceamento 7:2:1 não tratado | Falsos negativos de COVID: paciente infectado é liberado como "normal" |
| 2 | Split 80/20 sem estratificação, sem teste separado | O recall que embasaria a decisão de implantar não se reproduz |
| 3 | Só accuracy global | Ninguém sabe quantos COVID escaparam nem quantos alarmes falsos houve |
| 4 | ResNet-18 sem pré-treino com 1.200 imagens | Features frágeis, decoram o treino, usam atalhos |
| 5 | Overfitting (93%×61%) sem regularização | Modelo final pior que o melhor já alcançado no treino |
| 6 | Sem augmentation | Pouca robustez a variações rotineiras de aquisição |
| 7 | SGD com LR fixo, 15 épocas arbitrárias | Resultado não reprodutível entre seeds |
| 8 | Vazamento por paciente e viés de fonte | Falha silenciosa em produção; o modelo pode aprender a fonte, não a doença |

**Confirmação numérica.** Prever "Normal" para tudo já dá **70,0%** de accuracy na distribuição enviesada de desenvolvimento (acima dos 61% de validação e abaixo dos 93% de treino do relatório do grupo) — o paradoxo da acurácia direto — mas só **33,3%** de accuracy balanceada e **0% de recall de COVID** num teste equilibrado. Reproduzindo a receita do grupo (seção seguinte): treino 100%, validação aleatória 89,6% (um número que parece bom), mas **teste 70,5%**, com recall de COVID de apenas **23,5%** — a validação, embaralhada na mesma distribuição enviesada do treino, não expôs o problema.

### Dados

`tawsifurrahman/covid19-radiography-database`: COVID 3.616, Normal 10.192, Viral Pneumonia 1.345 arquivos (Lung_Opacity ignorada, fora do escopo). Após MD5 (**54 duplicatas exatas removidas, nenhuma cruzando classes**) e pHash (Hamming ≤ 4, mais permissivo: **31 rejeições adicionais**, 18 em COVID e 13 em Normal, das quais **4 cruzando classes/partições** — sem essa checagem, até 4 radiografias quase idênticas poderiam aparecer em dev e teste ao mesmo tempo).

**Confusão classe × fonte, confirmada.** Pneumonia vem de **uma única fonte** (`paultimothymooney/chest-xray-pneumonia`, 440 imagens, o repositório pediátrico de Guangzhou); Normal vem de duas fontes (RSNA 893, o mesmo repositório pediátrico 147); COVID vem de **6 fontes** (BIMCV 217, Eurorad 23, COVID-CXNet 43, ieee8023 17, ml-workgroup 12, SIRM 8), cada uma com seu equipamento e faixa etária. O modelo pode estar aprendendo a fonte, não a doença; como o teste vem dos mesmos repositórios do treino, o desempenho medido aqui é um **limite superior** do que se veria num hospital novo.

**Brilho médio** (atalho possível): COVID 139,7 ± 26,7, Normal 129,8 ± 22,3, Pneumonia 127,9 ± 18,4 — COVID é ~10-12 unidades mais claro em média, pequeno frente ao desvio de cada classe, mas coerente com o viés de fonte/equipamento.

**Grade 64 px** (decisão anti-atalho: reduz o texto sobreposto a ilegível): mantém forma do tórax, silhueta cardíaca, clavículas e dispositivos grandes; perde opacidades finas em vidro fosco — a marca radiológica mais discutida do COVID — e textura fina do parênquima.

![Amostras em 299 px e a versão 64 px usada pelos modelos.](figuras/A4_samples_grid.png)

### Baseline vs. pipeline corrigido

| | Baseline (receita do grupo) | Pipeline corrigido |
|---|---|---|
| Pré-treino | Não (do zero) | ImageNet |
| Otimizador | SGD, LR 0,01 fixo | AdamW, cosseno, *warmup* |
| Augmentation | Não | Sim (rotação, zoom, deslocamento, brilho/contraste) |
| Pesos de classe | Não | Sim |
| Seleção de época | Última (15) | Melhor macro-F1 val, *patience* 7 |
| **Accuracy teste** | 70,5% | **87,5%** |
| **Macro-F1 teste** | 0,665 | **0,874** |
| **Recall COVID teste** | 23,5% (47/200) | **67,5%** (135/200) |
| IC 95% (Wilson) do recall | — | **[60,7%; 73,6%]** |
| Especificidade COVID | 99,0% | **100%** |
| VPP a 10% de prevalência | 72,3% | **100%** |

![Matrizes de confusão: baseline vs. corrigido (teste).](figuras/A4_baseline_vs_corrected.png)

O pipeline corrigido não isola cada correção por ablação — a melhoria é do **pacote completo**; a leitura mais defensável é que pré-treino e pesos de classe atacam os dois problemas mais graves do baseline (features do zero com poucos dados; fronteira deslocada para Normal). **A meta de recall ≥ 0,90 da Aula 7 não foi atingida**: 0,675 no teste, limite inferior do IC (0,607) abaixo até de 0,85. O pipeline corrigido é uma melhoria grande e real, mas **não está pronto para triagem clínica** por esse critério.

### cGAN condicional: instabilidade e mitigação

Duas runs, mesma arquitetura condicional (embedding de classe, ruído 100-d), 250 épocas cada:

**Run A (ingênua).** Sem *spectral norm*, rótulo real = 1,0, LRs iguais (2e-4). Não houve divergência catastrófica (0 iterações não-finitas), mas os sintomas de **D dominando** são claros: $D(x)=0{,}847$ contra $D(G(z))=0{,}17/0{,}04$ na época final; $\text{loss}_G$ termina em 4,40. O **gap de *overfitting* do D** cresce de −0,04 (época 1) a um pico de **0,70** (época 225). O **KID melhora até a época 150 (0,288) e piora até o fim (0,296)**; a diversidade (LPIPS) sobe até a época 100 (0,337) e cai ~13% até o fim (0,292) — um enfraquecimento real, não um colapso total.

**Run B (mitigada: *spectral norm* + *label smoothing* 0,9 + TTUR 4×).** $\text{loss}_D=1{,}10$, $\text{loss}_G=1{,}30$; $D(x)=0{,}55$, $D(G(z))=0{,}29$ — o D ainda vence, com margem bem menor. O gap de *overfitting* do D fica entre −0,05 e **0,11** o treino todo. O **KID melhora monotonicamente até o fim** (0,264 na época 250, contra 0,296 da Run A) — sem a piora tardia da Run A. Não houve ablação fator a fator: a melhoria é do **pacote das três mitigações**, não atribuível a uma só.

![Curvas de treino da cGAN: Run A vs. Run B.](figuras/A4_gan_training_curves.png)

![Qualidade (KID) e diversidade ao longo do treino.](figuras/A4_gan_quality_curves.png)

### Qualidade, condicionamento e memorização das sintéticas

| | Run A | Run B |
|---|---|---|
| KID — COVID / Normal / Pneumonia | 0,282 / 0,379 / 0,444 | **0,268 / 0,362 / 0,368** |
| FID (COVID, comparativo) | 261,8 | **250,6** |
| Condicionamento COVID | 0,987 | **1,000** |
| Condicionamento Normal | 0,0067 | 0,0033 |
| Condicionamento Pneumonia | **0,780** | 0,203 |
| Razão de memorização (NN sint./NN real) | 0,970 | 0,972 |
| Fração de sintéticas mais próximas do treino que qualquer real | 0,0 | 0,0 |

A Run B melhora o KID nas 3 classes (mais em Pneumonia), mas **piora bastante o condicionamento de Pneumonia** (0,78 → 0,20) — um *trade-off* real que o KID sozinho não mostra. **Normal é o pior condicionamento nas duas runs** (perto de zero). A leitura direta é que o gerador não captura bem a ausência de opacidade que define "normal" — mas há uma explicação alternativa não descartada: como o controle real×sintético (abaixo) dá AUC 1,00, o classificador usado para medir condicionamento (que pondera COVID em ~3,3×) pode rotular qualquer imagem "com cara de sintética" como COVID, inflando o condicionamento de COVID às custas de Normal/Pneumonia. Não foi medida a distribuição completa de predições das sintéticas de Normal/Pneumonia, o que distinguiria as duas hipóteses — fica como limite da análise. A razão de memorização perto de 1,0 e a fração 0,0 de sintéticas "cópia" descartam decoreba do treino.

**Controle real vs. sintético: AUC = 1,00 ± 0,00** (300 por classe, com uma regressão logística sobre features do Inception e com uma CNN pequena treinada direto sobre os pixels). Separação **perfeita** nos dois métodos — alerta forte de atalho: um classificador treinado com as duas populações juntas pode aprender "cara de sintético" em vez da doença.

![Amostras finais da cGAN.](figuras/A4_gan_final_samples.png)

![Teste de memorização: vizinho mais próximo real vs. sintético.](figuras/A4_gan_memorization.png)

### Com vs. sem sintéticos: o experimento decisivo

*Sweep* com multiplicadores 0×/1×/3× de sintéticos COVID sobre o treino real (96 imagens), 3 seeds cada, pipeline corrigido:

| Multiplicador | Recall COVID (média ± desvio, 3 seeds) | Macro-F1 |
|---|---|---|
| 0× (só reais) | **72,3% ± 4,5** | 0,885 ± 0,010 |
| 1× (+96 sintéticas) | 66,5% ± 3,9 | 0,867 ± 0,012 |
| 3× (+288 sintéticas) | 66,3% ± 4,3 | 0,860 ± 0,023 |

A média do recall **cai** com sintéticos. A diferença pareada (1×−0×) tem IC 95% t (gl=2) de **[−22,0; +10,4] p.p.** e (3×−0×) de **[−15,4; +3,4] p.p.** — os dois incluem zero, mas o McNemar por seed revela a tendência: em 1×, 2 das 3 seeds pioram de forma significativa (p=0,014 e p=0,0016); em 3×, 1 de 3 (p=0,0055). **Nenhuma seed melhora de forma significativa.**

Precisão e especificidade de COVID **sobem** (não descem) com sintéticos (98,7%→99,5%→99,0%), enquanto o recall cai — o oposto do que um deslocamento de fronteira "para COVID" previria. A leitura mais provável: os sintéticos **diluem** o sinal de treino, gastando capacidade do modelo em características do COVID sintético que não transferem para o COVID real do teste — coerente com o AUC de 1,00 do controle real-vs-sintético. O ponto 0× já treina com pesos de classe; os sintéticos não têm ganho incremental sobre o que os pesos já entregam.

![Métricas de COVID no sweep 0×/1×/3×.](figuras/A4_sweep_covid_metrics.png)

![Matrizes de confusão do sweep, por multiplicador e seed.](figuras/A4_sweep_confusion.png)

**Conclusão honesta.** Neste experimento (3 classes, 64 px, pipeline já corrigido, 3 seeds), a GAN **não ajudou** — há indícios reais, embora não conclusivos na média de 3 seeds, de que **atrapalhou** o recall de COVID. Contrasta com a Aula 7 (+13 p.p., 1 seed): a diferença mais provável é o número de seeds, não o pipeline — um único treino pode acertar por sorte de inicialização/split.

### Plano de melhoria e critério de adoção clínica

Priorização guiada pelos resultados: com o recall do corrigido em 67,5% (abaixo da meta 0,90) e a GAN sem ganho no sweep, o gargalo **não é validação externa ainda** — é fechar a distância até a meta no próprio teste interno.

1. **Subir a resolução de entrada** (224–512 px, com ou sem GAN retreinada na mesma resolução) — o teto mais provável, pela perda de opacidades finas em 64 px.
2. **Auditar e mitigar atalhos de fonte** — Pneumonia de uma única fonte, COVID de 6, é confusão classe × fonte real, não hipotética.
3. **Reduzir a variância entre seeds** (desvio de ±4-5 p.p. no recall) — ensemble ou *test-time augmentation*.
4. **GAN/sintéticos pausados**: não adotar nesta configuração — Δ recall médio −5,8 p.p. (1×) e −6,0 p.p. (3×), sem ganho e com o controle real-vs-sintético falhando o pré-requisito de AUC baixo (deu 1,00).
5. Só depois de bater a meta de sensibilidade no teste interno, calcular a amostra para teste externo: **92 a 127 casos COVID confirmados**, dependendo da sensibilidade real, para que o limite inferior do IC fique acima de 0,85–0,90.

Critério de adoção clínica proposto (a validar com o time clínico): sensibilidade ≥ 0,90 com IC 95% inferior ≥ 0,85; especificidade ≥ 0,80 na prevalência local; calibração (ECE ≤ 0,05); humano no *loop* (o modelo prioriza a fila, não decide sozinho); estudo prospectivo silencioso antes de qualquer uso; monitoramento pós-implantação com auditoria mensal e plano de *rollback*.

### Uso de IA

Notebook (`notebooks/src/A4_estudo_caso_raio_x.py` → `.ipynb`) escrito com Claude Code (`construtor-notebook`): diagnóstico do baseline, EDA com MD5/pHash e tabela de fontes, baseline e pipeline corrigido, cGAN condicional com *spectral norm*/*label smoothing*/TTUR, métricas de qualidade (KID/FID/LPIPS via torch-fidelity e torchmetrics), controle real-vs-sintético, *sweep* com 3 seeds e testes pareados (t, McNemar), plano de melhoria com cálculo de amostra. Análise (`analista-resultados`) escrita a partir do JSON de métricas da execução na T4 (28/09/2026) e das figuras geradas — todos os números deste texto foram conferidos contra `A4_metrics.json`. Verificação humana: pendente de revisão por Gilmar; leituras sobre atalhos de fonte e a causa da queda de recall com sintéticos mantidas como hipótese, apoiadas no controle real-vs-sintético (AUC 1,00) mas não isoladas experimentalmente.
