# Plano de Execução — Projeto da Disciplina

**Disciplina:** Visão Computacional com CNNs e Transformers [26E3_3] — Infnet
**Aluno:** Gilmar Oliveira de Medeiros
**Prazo:** segunda, **05/10/2026 23:59** (2 tentativas de envio no Moodle)
**Enunciado + rubrica:** `../../26T3-LIA-C3-DEEPLE_ Projeto da Disciplina [Obrigatório].pdf` (a rubrica completa está nas págs. 5–7; o `.md` convertido está truncado)

Este plano deve ser executado **em etapas por agentes de IA**, com um **gate de aprovação do Gilmar ao fim de cada etapa**. Nenhuma etapa começa sem o OK da anterior.

---

## 0. Decisões e convenções

### 0.1 Decisões tomadas
| Tema | Decisão |
|---|---|
| A1: dataset | **NEU Surface Defects** (indústria: 6 classes, 1.800 imagens 200×200 em tons de cinza, Kaggle) |
| A4.1: abordagem generativa | **GAN condicional (cGAN)**. Base: `../../Aula-7/aula_07_gans_generative_adversarial_networks.ipynb` |
| A4.1: dados | COVID-19 Radiography Database (Kaggle), amostra **Normal 840 / Pneumonia 240 / COVID 120** replicando o cenário, mais um **test set fixo e separado** |
| Acesso a dados | **`kagglehub` + Colab Secrets** (`KAGGLE_USERNAME`, `KAGGLE_KEY`) |
| Workflow | **O local é a fonte da verdade** → repo GitHub público → abrir no Colab pelo link do GitHub → executar via `colab-mcp` |
| Relatório | **Markdown + pandoc** → PDF em PT-BR |
| Checkpoints | **Gate ao fim de cada etapa** |
| Coordenação dos agentes | **Supervisor**: a sessão principal orquestra e executa no Colab (ver §2.1) |
| Contexto das aulas | **Base de conhecimento** em `docs/base_conhecimento/` + agentes custom que podem abrir o material original |
| Modelos | **Opus** para código, análise e revisão; **Sonnet** para redação e formatação do relatório |
| Paralelismo | Permitido para trabalho **sem GPU** enquanto o Colab treina |
| Colab | **Plus**, mas **sempre com runtime T4** (exigência do enunciado; os tempos do cabeçalho são medidos na T4) |
| Nome dos arquivos | `gilmar_medeiros` |
| Cronograma | Sem metas de data: executar em sequência tão rápido quanto os gates permitirem |

### 0.2 Convenções para todos os notebooks
- **Autocontidos**: nenhum import de módulo local; cada notebook roda "Executar tudo" do início ao fim no Colab T4 sem modificações, além de montar o Drive (exigência do enunciado).
- **Primeira célula (markdown)**: título, objetivo, **requisitos de memória (RAM/VRAM) e tempo estimado de execução** (exigência do enunciado), link "Open in Colab".
- **Célula de setup**: `pip install` mínimo, seed 42 (`random`, `numpy`, `torch`, `cudnn.deterministic` quando viável), checagem da GPU, montagem do Drive, leitura dos Secrets do Kaggle via `google.colab.userdata`.
- **Persistência**: figuras, checkpoints e CSVs de métricas em `MyDrive/infnet_cv_projeto/outputs/<Ax>/`, para sobreviver a desconexões. Treinos longos retomam do checkpoint quando ele existe.
- **Idioma**: explicações e análises em PT-BR; código e nomes de variáveis podem ficar em inglês.
- **Estrutura interna**: Problema → Decisão técnica (com justificativa) → Código → Resultado → Análise. A avaliação pesa **análise e justificativa**, não só código.
- Figuras usadas no relatório também são salvas em PNG com nome estável (ex.: `A1_attention_head3.png`).

---

## 1. Estrutura do repositório
```
projeto-disciplina/
  agents/                      # cópia versionada dos agentes (os ativos ficam em ../.claude/agents/)
  docs/PLANO_PROJETO.md        # este plano + log de progresso
  docs/decisoes.md             # decisões, pendências e registro de uso de IA
  docs/base_conhecimento/      # README.md (índice) + partes geradas pelo pesquisador-aulas
  scripts/py2ipynb.py          # .py percent -> .ipynb, com checagem de sintaxe (--selftest)
  scripts/extract_figures.py   # PNGs das saídas de um .ipynb executado -> relatorio/figuras/
  notebooks/src/<Ax>.py        # FONTE editável dos notebooks (formato percent)
  notebooks/A1_vision_transformers.ipynb   # gerados a partir de notebooks/src/
  notebooks/A2_clip_ads16.ipynb
  notebooks/A3_cnn_kaggle.ipynb
  notebooks/A4_estudo_caso_raio_x.ipynb
  notebooks/00_smoke_test.ipynb   # Etapa 0 (não vai na entrega)
  resultados/<Ax>/             # outputs.json (saídas capturadas do Colab) + metrics.json
  relatorio/secoes/            # seções por atividade e textos teóricos
  relatorio/relatorio.md       # documento único montado a partir das seções
  relatorio/figuras/
  entregas/                    # .ipynb executados baixados do Colab, PDF e ZIP final
```

---

## 2. Protocolo de execução (vale para todo agente)
1. **Editar localmente** o fonte `notebooks/src/<Ax>.py` → `python3 scripts/py2ipynb.py notebooks/src/<Ax>.py -o notebooks/<Ax>.ipynb` → `git commit` → `git push`.
2. No Chrome, abrir `https://colab.research.google.com/github/<user>/<repo>/blob/main/notebooks/<A>.ipynb`, confirmar o **runtime T4** (mesmo com o Plus) e chamar `open_colab_browser_connection`.
3. Executar célula a célula com `run_code_cell`, lendo as saídas. Em caso de erro, corrigir e **replicar a correção no fonte `.py` local** (nunca deixar a correção só no Colab), depois fazer push de novo.
4. Salvar as saídas: `get_cells(includeOutputs)` → `resultados/<Ax>/outputs.json`; o JSON impresso pela última célula → `resultados/<Ax>/metrics.json`.
5. Ao terminar, rodar **"Executar tudo" desde um runtime limpo** (prova de reprodutibilidade), anotar o tempo real no cabeçalho, baixar o `.ipynb` executado para `entregas/` e extrair as figuras com `python3 scripts/extract_figures.py entregas/<Ax>.ipynb -o relatorio/figuras --prefix <Ax>`.
5. A execução via `colab-mcp` fica com a **sessão principal**, porque o MCP está preso à aba do Chrome. Subagentes escrevem código e texto e fazem revisão.
6. **Gate de fim de etapa**: o agente apresenta (a) as métricas principais, (b) o checklist dos itens de rubrica da etapa com a evidência de cada um, (c) dúvidas e conflitos. Depois **para e espera o OK do Gilmar**.
7. **Uso de IA**: registrar em `docs/decisoes.md` o que foi gerado ou assistido por IA (Claude Code), para a seção obrigatória de citação no relatório.
8. Sempre que faltar referência ou houver conflito entre abordagens, **perguntar ao Gilmar** em vez de assumir.

### 2.1 Time de agentes e fluxo
Definições em `../.claude/agents/` (raiz do curso, onde o Claude Code as carrega), com cópia em `agents/`. Pode ser preciso reiniciar a sessão ou usar `/agents` para carregá-las.

| Agente | Modelo | Ferramentas | Papel | Entrada → Saída |
|---|---|---|---|---|
| **Orquestrador** (sessão principal) | Opus | todas + colab-mcp + git | Chama os agentes, **executa no Colab**, faz git push, conduz os gates, atualiza o log | — |
| `pesquisador-aulas` | Opus | Read, Glob, Grep, Bash (só leitura), Write (só em `docs/base_conhecimento/`) | Destila das aulas os conceitos, o código reaproveitável (arquivo + célula), a terminologia e o estilo do professor | `Aula-*` → `docs/base_conhecimento/<parte>.md` |
| `construtor-notebook` | Opus | Read, Write, Edit, Glob, Grep, Bash | Escreve e corrige o notebook da etapa seguindo §0.2 e a base de conhecimento; checagem estática local | Etapa + base → `notebooks/src/<Ax>.py` → `.ipynb` |
| `analista-resultados` | Opus | Read, Write, Edit, Glob, Grep | Interpreta as saídas executadas, escreve as células de análise e o rascunho da seção; aponta anomalias | `resultados/<Ax>/` + figuras → análise + `relatorio/secoes/<Ax>.md` |
| `redator-relatorio` | Sonnet | Read, Write, Edit, Glob, Grep, Bash (pandoc/zip) | Textos sem GPU (A4.2 e teoria), montagem do relatório, PDF, ZIP | Seções → `entregas/gilmar_medeiros_*` |
| `revisor-rubrica` | Opus | Read, Glob, Grep (**somente leitura**) | Auditor adversarial: PASSA/FRACO/FALHA por item de rubrica, com evidência; confere convenções e rigor | Artefatos da etapa → parecer no chat |

```
construtor ──► orquestrador executa no Colab (T4) ──► analista ──► revisor ──► GATE Gilmar
   ▲                 │ erro simples: corrige e replica no .py      │ FALHA/FRACO
   └─────────────────┴──── erro estrutural / item reprovado ◄──────┘
Em paralelo (sem GPU): construtor prepara o notebook da próxima etapa; redator escreve A4.2 e textos teóricos.
```
- Os agentes **não** executam no Colab nem fazem commit; só o orquestrador faz isso.
- Dúvidas dos agentes voltam ao orquestrador, que as leva ao Gilmar (AskUserQuestion).
- Figuras para o relatório: a célula começa com `# fig: nome_estavel`.

---

## 3. Etapas

### Etapa 0: Setup (sem GPU)
**Gilmar**
- [ ] `! gh auth login` (o token atual do `gh` está inválido).
- [ ] Criar os Secrets `KAGGLE_USERNAME` e `KAGGLE_KEY` no Colab (ícone de chave) e liberar o acesso aos notebooks.
- [ ] Aceitar, no site do Kaggle, os termos dos datasets, se algum pedir.

**0.1 Base de conhecimento (agentes, em paralelo, sem GPU)**
- [ ] 3 instâncias do `pesquisador-aulas` em paralelo:
  - `Aula-1` + `Aula-6` → `docs/base_conhecimento/A3_A2_cnn_clip.md`
  - `Aula-2` … `Aula-5` → `docs/base_conhecimento/A1_transformers_vit.md`
  - `Aula-7` + `Aula-8` → `docs/base_conhecimento/A4_gans.md`
- [ ] O orquestrador cria `docs/base_conhecimento/README.md` (índice, lacunas e dúvidas consolidadas) e leva as dúvidas ao Gilmar.

**0.2 Infraestrutura (orquestrador)**
- [ ] Criar o repo **público** no GitHub (sugestão: `infnet-cv-projeto-disciplina`), com `.gitignore` (dados, checkpoints, `.DS_Store`) e a estrutura da seção 1.
- [ ] Criar `docs/decisoes.md`.
- [ ] Criar `notebooks/00_smoke_test.ipynb`: lê os Secrets, baixa e lista cada dataset com `kagglehub` e imprime a contagem de arquivos por classe.
- [ ] **Validar os slugs** (pendência 5.1) e registrar os slugs finais em `docs/decisoes.md`.

**Gate 0**: base de conhecimento revisada, o repo abre no Colab pelo link do GitHub e os 4 datasets baixam, com contagens por classe conferidas.

---

### Etapa 1: A3, classificador com CNN pré-treinada
Primeira etapa por ser curta: valida o pipeline de ponta a ponta. Dataset: Kaggle `pavansanagapati/images-dataset` (~1.800 imagens).
Base de código: `../../Aula-1/aula_01_cnn_architectures.ipynb` (seção 5, feature extraction).

- [ ] EDA: contagem por classe, grade de amostras, tamanhos e canais das imagens, checagem de desbalanceamento.
- [ ] Split **estratificado** train/val/test (ex.: 70/15/15), com seed fixa.
- [ ] **Justificar a escolha do modelo** (ResNet-50 vs. EfficientNet-B0: parâmetros, FLOPs, VRAM e batch possível na T4, número de classes, tamanho do dataset). **Rubrica 1.5**
- [ ] Transforms com a normalização ImageNet do modelo escolhido.
- [ ] Congelar o backbone e substituir o head por `Linear(in_features, n_classes)`; conferir que só o head tem `requires_grad`, imprimindo params treináveis vs. totais. **Rubrica 1.1**
- [ ] **Um único treino**; registrar loss e accuracy de train/val por epoch.
- [ ] Curvas de loss e accuracy por epoch, **accuracy global + accuracy por classe** no test, matriz de confusão e exemplos de erros. **Rubrica 1.2**
- [ ] **3.2, análise escrita**: para cada opção (a) geométrica, (b) cor, (c) escala, (d) normalização, dizer se ajuda neste dataset e **para quais classes pode distorcer ou prejudicar** (ex.: grayscale em classes que se distinguem pela cor, flip em classes com orientação ou texto). Pelo menos 3 estratégias com justificativa de domínio. **Rubrica 1.3**
- [ ] Discussão escrita: **feature extraction vs. fine-tuning** conforme o tamanho e o domínio do dataset (proximidade com o ImageNet, risco de overfitting, custo). **Rubrica 1.4**

**Gate 1**: métricas, curvas, checklist 1.1–1.5.

---

### Etapa 2: A1, Vision Transformers no NEU Surface Defects
Etapa mais pesada; cobre os itens de rubrica 2.x e 3.x.
Bases: `../../Aula-4/aula_04_vision_transformers.ipynb`, `../../Aula-5/aula_05_advanced_vision_transformers.ipynb`, `../../Aula-5/aula_05_deit_distillation_transfer_learning.ipynb`, `../../Aula-5/aula_05_dino_self_supervised_vision.ipynb` (mapas de atenção).

**2a. Problema e dados**
- [ ] Definição do problema: inspeção visual de defeitos em chapas de aço laminado a quente (6 classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches) e sua relevância industrial.
- [ ] EDA: contagem por classe, amostras, observação de que as imagens são texturas em tons de cinza, com defeitos distribuídos, não objetos centrados.
- [ ] Split estratificado fixo train/val/test, **o mesmo para todos os modelos**.

**2b. Módulos do zero (PyTorch), com testes**
- [ ] `ScaledDotProductAttention` (retorna saída **e** pesos de atenção; suporte a máscara).
- [ ] `MultiHeadAttention` com **projeções Q/K/V independentes por head** e **concatenação** das saídas mais a projeção de saída. **Rubrica 2.1**
- [ ] `TransformerEncoderBlock`: MHA + **MLP de duas camadas** (GELU) + **LayerNorm** (pre-norm) + **conexões residuais**. **Rubrica 2.3**
- [ ] `PatchEmbedding` (Conv2d com stride = patch), **CLS token aprendível**, **positional embedding** aprendível, pilha de blocos e head → `ViT` que recebe uma imagem e retorna logits. **Rubrica 3.1**
- [ ] **Testes (asserts)**: shapes; linhas da atenção somando 1; equivalência numérica com `torch.nn.functional.scaled_dot_product_attention`; forward do ViT com batch aleatório; contagem de parâmetros.
- [ ] **Demonstração do positional encoding**: sem PE, embaralhar os patches só permuta as saídas dos tokens (a atenção é equivariante a permutações) e o CLS não muda; com PE, muda. Mais um texto explicando por que a atenção sem PE não preserva a posição. **Rubrica 2.4**

**2c. Treino do ViT do zero**
- [ ] ViT pequeno (ex.: img 128, patch 16, dim 192, 6 blocos, 3 ou 6 heads), AdamW, warmup + cosine, label smoothing, augmentations compatíveis com textura de aço (flips e rotações de 90°, que preservam a semântica do defeito; justificar cada uma).
- [ ] Curvas, métricas no test (accuracy, F1 macro, matriz de confusão).
- [ ] **Mapas de atenção de ao menos 1 head** (atenção CLS→patches, sobreposta à imagem) e **identificação escrita das regiões emergentes**. **Rubrica 3.2, 2.2**

**2d. Fine-tuning de ViT pré-treinado**
- [ ] Modelo pré-treinado compatível com a T4 (candidatos: `google/vit-base-patch16-224` ou `facebook/deit-small-patch16-224`; justificar), **substituindo o head** por 6 classes, com as imagens em cinza replicadas para 3 canais e normalização do modelo.
- [ ] Mesmo split, curvas, métricas, mapas de atenção (1 head e, opcionalmente, attention rollout).
- [ ] **Baseline CNN** leve (ResNet-18 ou 50 pré-treinada) com o mesmo split, para embasar a discussão ViT vs. CNN.

**2e. Comparação e análise**
- [ ] **Tabela quantitativa**: ViT do zero vs. ViT pré-treinado (vs. CNN), com accuracy, F1 macro, parâmetros, tempo de treino e VRAM. **Rubrica 3.3, 3.6**
- [ ] Justificativa da arquitetura escolhida para o domínio **com base nos dados**. **Rubrica 3.6**
- [ ] Interpretação escrita da atenção: o que o modelo pondera em cada tipo de defeito (ex.: linhas finas em *scratches*, manchas em *patches*). **Rubrica 2.2**
- [ ] Texto: **DeiT** (eficiência de dados, distillation token) e **Swin** (janelas deslocadas, hierarquia, custo linear), dizendo o que cada um resolve que o ViT original não resolve. **Rubrica 3.4**
- [ ] Texto: **quando ViT supera CNN e quando CNN é preferível**, aplicado a este dataset (pequeno e de textura, onde o viés indutivo de localidade da CNN ajuda). **Rubrica 3.5**
- [ ] Texto: **pré-treinamento do BERT (MLM/NSP, auto-supervisionado em texto) vs. do ViT (supervisionado no ImageNet-21k, ou MAE/DINO)**, dizendo o que cada estratégia maximiza. **Rubrica 2.5**
- [ ] "O que eu mudaria": próximos passos justificados.

**Gate 2**: tabela comparativa, figuras de atenção, checklist 2.1–2.5 e 3.1–3.6.

---

### Etapa 3: A4.1, detecção de COVID-19 em raio-X com cGAN
Base: `../../Aula-7/aula_07_gans_generative_adversarial_networks.ipynb` (cGAN + comparação de recall) e `../../Aula-7/aula_07_dcgan_cifar10_treinamento.ipynb` (diagnóstico de treino).
Referência: Dumakude, A., & Ezugwu, A. E. (2023). Automated COVID-19 detection with convolutional neural networks. *Scientific Reports*, 13, 10607, https://www.nature.com/articles/s41598-023-37743-4 (autores corrigidos em 28/09/2026 — a atribuição "Nour & Tariq" estava errada, ver `docs/decisoes.md`)

**3a. Diagnóstico do projeto anterior (texto)**: pelo menos 5 problemas, cada um com o **impacto clínico**. **Rubrica 5.1**
- [ ] Desbalanceamento 7:2:1 não tratado → o modelo favorece "Normal" e gera falsos negativos de COVID.
- [ ] Split 80/20 **sem estratificação** → a validação pode ter pouquíssimos casos de COVID, com métrica instável; também não há test set separado.
- [ ] **Só accuracy global** → 70% de accuracy é possível prevendo tudo "Normal"; faltam recall/sensibilidade por classe, matriz de confusão e AUC.
- [ ] ResNet-18 **sem pré-treino** com 1.200 imagens → subajuste das features e overfitting.
- [ ] **Overfitting** (93% treino vs. 61% validação) sem regularização nem early stopping.
- [ ] **Sem augmentation**.
- [ ] SGD com **LR fixo 0.01**, sem scheduler, e 15 epochs arbitrárias.
- [ ] Possível vazamento por paciente e viés de fonte/hospital (atalhos do tipo marcações e dispositivos na imagem).

**3b. Dados**
- [ ] Amostrar **840/240/120** (Normal/Pneumonia/COVID) do COVID-19 Radiography Database como conjunto de desenvolvimento, mais um **test set fixo, estratificado e disjunto** (retirado do restante do dataset, mantendo a proporção). Justificar a escolha.
- [ ] As imagens sintéticas entram **só no treino**; nunca em validação ou teste.

**3c. Baseline e pipeline corrigido**
- [ ] Reprodução resumida do baseline "como o grupo fez" (ResNet-18 do zero, SGD 0.01, 15 epochs, sem augmentation, split não estratificado), reportando accuracy **e** recall por classe para evidenciar o problema.
- [ ] Pipeline corrigido: ResNet-18 **pré-treinada**, split estratificado, class weights ou `WeightedRandomSampler`, augmentation leve e clinicamente plausível (rotação pequena, brilho/contraste; **sem flip vertical**), AdamW + scheduler, early stopping por **recall/F1 macro de validação**.

**3d. cGAN condicional**
- [ ] Gerador e discriminador condicionados por embedding de classe, 64–128 px, em tons de cinza, **loop adversarial correto** (`.detach()` no passo de D, heurística não saturante no G, alternância D/G). **Rubrica 5.2**
- [ ] **Run A** (configuração ingênua): documentar a instabilidade observada (**mode collapse** ou **divergência**) com curvas de D/G, grades de amostras por epoch e uma métrica de diversidade (ex.: distância média par a par entre amostras, ou FID/KID em resolução reduzida).
- [ ] **Run B** (mitigação: spectral norm e/ou label smoothing, TTUR, e opcionalmente DiffAugment): mostrar a **evidência de melhoria** com as mesmas métricas. **Rubrica 5.3**
- [ ] Gerar N imagens COVID sintéticas (ex.: igualando a contagem de Pneumonia ou de Normal; justificar) e inspecionar a qualidade visualmente.

**3e. Experimento e plano**
- [ ] Treinar o pipeline corrigido **sem** vs. **com** sintéticos (≥3 seeds se o orçamento de GPU permitir) e reportar **recall, precision e F1 por classe**, com foco no **recall de COVID**, mais a matriz de confusão e média ± desvio. **Rubrica 5.4**
- [ ] Análise crítica: a GAN ajudou? Riscos (artefatos, memorização, viés), limitações do tamanho amostral.
- [ ] **Plano de melhoria integrado** cobrindo **modelo, métrica, augmentation sintética e critério de adoção clínica** (ex.: recall COVID ≥ 0,90 com limite inferior do IC acima de um limiar acordado com o time clínico, validação externa multicêntrica, calibração, humano no loop, monitoramento pós-implantação). **Rubrica 5.6**

**Gate 3**: tabela com vs. sem sintéticos, evidências da GAN, checklist 5.1–5.4 e 5.6.

---

### Etapa 4: A2, CLIP no ADS-16 (sem treino)
Base: `../../Aula-6/aula_06_clip_openai.ipynb` (passos 4–7: álgebra multimodal, prompt ensembling, busca semântica, t-SNE).

- [ ] Carregar o ADS-16 e montar um **subconjunto de ≥500 imagens com amostragem estratificada pelas 16 categorias** (justificar a representatividade).
- [ ] CLIP pré-treinado (ex.: `openai/clip-vit-base-patch32` ou `-patch16`; justificar pela T4), com embeddings de imagem normalizados e **cacheados no Drive**.
- [ ] **2.1a**: ≥20 descrições de objetos e conceitos distintos (templates de prompt / ensembling) → matriz de cosine similarity imagem × conceito.
- [ ] **2.1b**: **threshold definido e justificado** (ex.: percentil da distribuição de similaridades, margem sobre um prompt neutro/"a photo", ou validação manual numa amostra) → ranking dos conceitos por **frequência de ocorrência** e **score médio**.
- [ ] **2.1c**: visualizar os **5 conceitos mais frequentes** com imagens exemplo do corpus. **Rubrica 4.2**
- [ ] **2.2**: busca texto→imagem com top-5; **≥8 consultas** indo do genérico ao específico e do concreto ao abstrato; para cada uma, analisar se o modelo recupera o que a consulta descreve ou interpreta de forma inesperada. **Rubrica 4.3**
- [ ] Texto: **alinhamento das representações visual e textual** no CLIP e por que o pré-treinamento contrastivo (InfoNCE simétrica, 400M pares) permite recuperação semântica sem treino supervisionado. **Rubrica 4.1**
- [ ] Texto e código ilustrativo: **consulta textual do CLIP vs. tokenização do BERT**, com o papel do **padding** e da **attention mask** (CLIP: BPE, 77 tokens, embedding do token EOT; BERT: WordPiece, [CLS]/[SEP], máscara). **Rubrica 4.4**
- [ ] (Opcional) t-SNE/UMAP dos embeddings coloridos pela categoria.

**Gate 4**: ranking, grade top-5, tabela das consultas analisadas, checklist 4.1–4.4.

---

### Etapa 5: A4.2, transfer learning para tráfego urbano (só relatório)
Referência: https://www.meegle.com/en_us/topics/transferlearning/transfer-learning-for-traffic-analysis

Pelo menos 4 problemas, cada um com **impacto operacional** e **como abordaria**, além dos **riscos do transfer learning ImageNet → classificação de fluxo**. **Rubrica 5.5**
- [ ] **Domain/task shift**: o ImageNet ensina a reconhecer objetos centrados, enquanto o fluxo exige densidade, ocupação e contexto espacial, então as features não são adequadas.
- [ ] **Dados pequenos e vazamento entre câmeras**: 800 frames de 12 câmeras, com frames quase idênticos no train e na val, inflam a accuracy de 78%. Proposta: split por câmera (leave-camera-out).
- [ ] **Cobertura de condições**: faltam chuva, noite e ângulos novos. Proposta: coleta direcionada, augmentation fotométrica/climática, domain adaptation, avaliação estratificada por condição.
- [ ] **Frame único, sem informação temporal**: congestionamento é um fenômeno temporal. Proposta: sequências curtas, optical flow, contagem de veículos com detector + rastreamento.
- [ ] **Rótulos subjetivos e fronteiras ambíguas** entre moderado e congestionado, provavelmente desbalanceados. Proposta: critério objetivo (ocupação/velocidade), métrica por classe, custo assimétrico dos erros.
- [ ] **Implantação sem monitoramento**: sem detecção de drift, sem limiar de confiança nem fallback. Proposta: MLOps com monitoramento por câmera e retreino.

**Gate 5**: texto revisado.

---

### Etapa 6: relatório, revisão e entrega
- [ ] `relatorio/relatorio.md`, documento único. Para cada atividade: **definição do problema, decisões técnicas e justificativa, resultados (métricas e gráficos), análise crítica**. Incluir:
  - Introdução, com o link do repo GitHub e o ambiente (Colab T4).
  - A1, A2, A3, A4.1, A4.2 (a A4.2 só aparece aqui).
  - Seção **"Uso de ferramentas de IA"** (obrigatória pelo enunciado): o que foi assistido, como foi verificado.
  - Referências (Dumakude & Ezugwu 2023; Dosovitskiy et al. 2021 ViT; Touvron et al. 2021 DeiT; Liu et al. 2021 Swin; Radford et al. 2021 CLIP; Mirza & Osindero 2014 cGAN; Devlin et al. 2019 BERT; Song & Yan 2013 NEU).
- [ ] Gerar o PDF: `pandoc relatorio/relatorio.md -o entregas/gilmar_medeiros_deep-learning-and-vision_computer-vision.pdf` (definir o engine e o template na etapa).
- [ ] **Revisão final pela rubrica**: percorrer os 26 itens da seção 4 e confirmar a evidência de cada um no notebook ou no relatório.
- [ ] Confirmar que os 4 notebooks em `entregas/` estão **executados**, com outputs, e têm o cabeçalho de memória e tempo.
- [ ] Gerar o ZIP `entregas/gilmar_medeiros_visao-computacional-cnns-transformers_pd.zip` (4 notebooks + PDF) e postar no Moodle (Gilmar).

**Gate 6**: o Gilmar revisa o PDF e o ZIP antes do envio.

---

## 4. Matriz rubrica → evidência (atualizar a cada gate)

| # | Competência | Item da rubrica (resumo) | Onde | Status |
|---|---|---|---|---|
| 1.1 | CNN transfer learning | CNN pré-treinada, head trocado para n classes, backbone congelado (feature extraction) | A3 | ✅ |
| 1.2 | CNN transfer learning | Curvas de treino + accuracy por classe e global | A3 | ✅ |
| 1.3 | CNN transfer learning | ≥3 estratégias de augmentation com justificativa para o domínio | A3 §3.2 | ✅ |
| 1.4 | CNN transfer learning | Quando usar feature extraction vs. fine-tuning (tamanho e domínio) | A3 + relatório | ✅ |
| 1.5 | CNN transfer learning | Escolha do modelo justificada (T4, nº de classes) | A3 | ✅ |
| 2.1 | Transformer | SDPA e MHA do zero, testáveis, com projeções por head e concatenação | A1 2b | ☐ |
| 2.2 | Transformer | Heatmap de atenção de ≥1 exemplo do domínio + interpretação escrita | A1 2c/2e | ☐ |
| 2.3 | Transformer | TransformerEncoderBlock completo (MLP 2 camadas, LayerNorm, residual) como base do ViT | A1 2b | ☐ |
| 2.4 | Transformer | PE aplicado aos tokens + explicação de por que a atenção sem PE não preserva a posição | A1 2b | ☐ |
| 2.5 | Transformer | Pré-treinamento do BERT vs. do ViT: o que cada um maximiza | A1 2e + relatório | ☐ |
| 3.1 | ViT | Patch embedding, CLS aprendível, PE → ViT completo que retorna logits | A1 2b | ☐ |
| 3.2 | ViT | ViT do zero treinado, mapas de atenção de ≥1 head, regiões emergentes por escrito | A1 2c | ☐ |
| 3.3 | ViT | Fine-tuning de ViT pré-treinado com head trocado, comparado com o ViT do zero em tabela | A1 2d/2e | ☐ |
| 3.4 | ViT | DeiT e Swin: o que resolvem que o ViT não resolve | A1 2e + relatório | ☐ |
| 3.5 | ViT | Quando ViT supera CNN e quando CNN é preferível, no domínio | A1 2e + relatório | ☐ |
| 3.6 | ViT | Tabela ViT do zero vs. pré-treinado + escolha de arquitetura baseada nos dados | A1 2e | ☐ |
| 4.1 | CLIP | Alinhamento visual-textual; por que o contrastivo habilita recuperação sem supervisão | A2 + relatório | ☐ |
| 4.2 | CLIP | Ranking por frequência semântica, ≥20 descrições, top-5 com exemplos | A2 2.1 | ☐ |
| 4.3 | CLIP | Busca semântica com ≥8 consultas variadas, documentadas e analisadas | A2 2.2 | ☐ |
| 4.4 | CLIP | Consulta textual do CLIP vs. tokenização do BERT; padding e attention mask | A2 + relatório | ☐ |
| 5.1 | GANs / casos | ≥5 problemas do projeto de raio-X com impacto clínico | A4 3a + relatório | ☐ |
| 5.2 | GANs / casos | GAN (cGAN) implementada com loop adversarial correto | A4 3d | ☐ |
| 5.3 | GANs / casos | Instabilidade diagnosticada (mode collapse/divergência) + mitigação com evidência | A4 3d | ☐ |
| 5.4 | GANs / casos | Impacto dos sintéticos no recall de COVID (com vs. sem) | A4 3e | ☐ |
| 5.5 | GANs / casos | ≥4 problemas do projeto de tráfego + riscos do TL ImageNet → fluxo | Relatório A4.2 | ☐ |
| 5.6 | GANs / casos | Plano integrado para o raio-X: modelo, métrica, sintéticos, critério de adoção clínica | A4 3e + relatório | ☐ |

---

## 5. Pendências a confirmar
1. **Slugs do Kaggle** (validar na Etapa 0):
   - NEU Surface Defects: candidato `kaustubhdikshit/neu-surface-defect-database`
   - ADS-16: a localizar (buscar "ADS-16 computational advertising dataset" no Kaggle)
   - A3: `pavansanagapati/images-dataset` (dado no enunciado)
   - COVID-19 Radiography: candidato `tawsifurrahman/covid19-radiography-database`
2. ✔ **Nome dos arquivos**: `gilmar_medeiros_deep-learning-and-vision_computer-vision.pdf` e `gilmar_medeiros_visao-computacional-cnns-transformers_pd.zip`.
3. ✔ **GPU**: Colab **Plus**, o que permite ≥3 seeds na A4.1 e mais épocas na A1, sempre na T4.
4. ✔ **Ordem das etapas**: A3 → A1 → A4.1 → A2 → A4.2 → relatório, sem metas de data, uma etapa atrás da outra (A4.2 e os textos teóricos adiantados em paralelo).

---

## 6. Log de progresso

| Etapa | Data | Status | Métricas-chave | Decisões do gate |
|---|---|---|---|---|
| Plano | 27/09/2026 | ✅ criado | — | Decisões da seção 0.1 |
| Time de agentes | 27/09/2026 | ✅ definido | `py2ipynb --selftest` OK | Supervisor, 5 agentes custom, base de conhecimento, Opus/Sonnet, paralelismo sem GPU, Plus com T4, `gilmar_medeiros`, sem metas de data |
| 0 Setup | 27/09/2026 | ✅ Gate 0 aprovado | 4 datasets OK (ver decisoes.md); base de conhecimento 657 linhas | D8–D10: 20 recomendações aceitas; corpus A2 = 300 ads + corpus usuários; ADS-16 no PDF com citação |
| 1 A3 | 27/09/2026 | ✅ Gate 1 | test 99,25% (IC Wilson 97,3–99,8), macro-F1 0,993; teste limpo (pHash) 99,24%; 5,7 min na T4 | Rubrica 1.1–1.5 PASSA (revisor); v2 com pHash, alfa medido, RAM/VRAM medidas |
| 2 A1 | 29/09/2026 | **Revisão adversarial (`revisor-gates`) concluída, todas as 7 correções aplicadas** — pronto para Gate 2 | ViT do zero 98,52% (266/270), ViT-B/16 e ResNet-18 100% (270/270); McNemar p=0,125 (compatíveis); ViT do zero converge na época 115 vs 3; 164s na T4 (reexecução com checkpoints) | — |
| 3 A4.1 | 29/09/2026 | **Revisão adversarial concluída, todas as pendências corrigidas** (notebook e relatório) — pronto para Gate 3 | Baseline 70,5% acc / 23,5% recall COVID; corrigido 87,5% acc / 67,5% recall (IC [60,7;73,6]), meta 0,90 não atingida; GAN Run B mais estável que Run A (KID 0,264 ± 0,010 vs 0,296 ± 0,010; piso KID real×real −0,0003 ± 0,0012); *sweep* 3 seeds: sintéticos **pioram** recall médio (−5,8 a −6,0 p.p.), 3 de 6 combos significativos; controle real×sintético AUC 1,00; 809,5s na T4 | Corrigidas nesta rodada: faixas de D(x)/D(G(z)) fabricadas → critério real da aula (D(x) > 0,6; D(G(z)) 0,3–0,5); piso de KID reescrito de "intervalo" para "média ± desvio"; referência cruzada "(item 3)" → "(item 2)"; texto de 5.3 do `.md` desatualizado em relação ao `.py` → sincronizado; citação Nour&Tariq → Dumakude&Ezugwu (2023); TTUR reatribuído a Heusel et al. (2017) |
| 4 A2 | 29/09/2026 | Bug do cache confirmado corrigido; **2ª rodada de correções em andamento** (fingerprint, asserts, JSON, quadrados pretos, cabeçalho) via `construtor-notebook`, aguardando reexecução | Confirmado oficialmente no Colab (122,8s, CPU), números idênticos à reexecução local: corpus 650 imgs (301 anúncios + 349 usuários, 120 usuários); sanidade zero-shot anúncio→categoria 55,8% (acaso 5%); ranking de 25 conceitos e busca com 10 consultas **conferem com o exame visual** na maioria dos casos (ex. "love and dating" 4/5 corretos) | 3 bugs reais corrigidos (formato de retorno do transformers; dataset em 2 partes; **cache de embeddings não invalidado**, achado pelo revisor-rubrica — a leitura "CLIP falha neste corpus" de uma execução anterior era esse bug, não um resultado real); revisor-gates ainda encontrou contagens infladas no ranking, 3 consultas sem análise, explicação errada dos "quadrados pretos" e números faltando no JSON — em correção |
| 5 A4.2 | 29/09/2026 | **Revisão adversarial concluída (PASSA), 3 pendências corrigidas** — pronto para Gate 5 | 6 problemas (acima do mínimo de 4), cada um com impacto operacional e abordagem; seção de riscos do transfer learning; tabela-resumo priorizada; referências reais (Beery 2018, Geirhos 2020, Kornblith 2019, Hendrycks & Dietterich 2019, Yosinski 2014, Zhu et al. 2017, Meegle citada como não-acessível) | Referência Meegle adicionada à lista (não-acessível, motivo registrado); citação de Kornblith (2019) requalificada (a correlação que ele mede é mais fraca justamente para tarefas distantes do domínio de origem, como esta); seção "Uso de IA" adicionada; `teoria_deit_swin.md`/`teoria_bert_vs_vit.md`/`teoria_clip_vs_bert.md` não criados por decisão — conteúdo já inline em A1.md e A2.md |
| 6 Entrega | | ☐ | | |
