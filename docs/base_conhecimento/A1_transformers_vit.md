# Base de conhecimento: A1 (Transformers e ViT)

Fonte: material das Aulas 2 a 5 (notebooks, planos de aula, falas do apresentador), trechos pertinentes dos livros e o enunciado da disciplina (págs. 1 e 5–7).
Convenções deste arquivo:
- Caminhos relativos à raiz do curso (`Visao_Computacional_com_CNNs_Transformers/`).
- "célula [n]" = índice **0-based** da célula no `.ipynb` (igual a `nb['cells'][n]`).
- "Slide n" = numeração do `0X_plano_aula.md` / `0X_falas_apresentador.md`. Os PDFs `aula_0X_apresentacao.pdf` são só imagem (sem texto extraível); para citar, usar o número do slide.
- Livros: **Géron** = *Hands-On ML with Scikit-Learn and PyTorch* (cap. 15 em `Aula-3/`, cap. 16 em `Aula-4/` e `Aula-5/`); **Crash Course** = *Deep Learning Crash Course*, cap. 8 (`Aula-2/3/4`). Páginas = página do PDF do capítulo.

---

## 1. Padrões transversais do professor (Allan Spadini)

**Estrutura dos notebooks**
- Célula 0 (markdown): título "Visão Computacional com CNNs e Transformers", subtítulo Infnet, badge "Open in Colab" apontando para `github.com/allanspadini/curso-vision-transformers-infnet`, roteiro numerado.
- Metodologia declarada em todos: **Situação-Problema → Solução de Engenharia → Fundamento Teórico → Aplicação Prática** (planos de aula §3). As seções em markdown seguem "#### 1. Situação-Problema / #### 2. Solução de Engenharia" antes do código (ex.: `Aula-2/aula_02_transformers.ipynb` células [9], [11], [13]).
- Fórmulas em LaTeX no markdown, antes do código (SDPA, MHA, FFN, CutMix, rollout, perda do DeiT).
- O markdown remete aos slides ("No **Slide 13**…", `Aula-4/aula_04_vision_transformers.ipynb` célula [24]).
- A última seção é sempre "Desafios Práticos" (Aula-2 [33], Aula-4 [28]).

**Setup** (`Aula-2` [2], `Aula-4` [2], `Aula-5/dino` [3])
- `!pip install -q ...`; `SEED = 42` em `random`, `np`, `torch.manual_seed`, `cuda.manual_seed_all`; `cudnn.deterministic = True` (Aula-2); a DINO usa `seed_everything(seed=42)`.
- Imprime o device, o nome da GPU e a **VRAM total** (`get_device_properties(0).total_memory`); em CPU, aviso para ativar a T4.

**Estilo de código**
- Comentários de shape em cada passo (`# [Batch, h, Seq_Q, d_k] x ... -> ...`), type hints nos `forward`, `assert d_model % n_heads == 0`.
- Docstrings e prints em PT-BR, prints com emoji e checkmarks (✅/❌).
- Contagem de parâmetros com `sum(p.numel() ...)`, total e treináveis.

**Avaliação e visualização**
- Métricas: `evaluate.load("accuracy")` + `evaluate.load("f1")` com `average="macro"` (Aula-4 [17]); `classification_report(..., digits=4)` e matriz de confusão `sns.heatmap(cmap="Blues")` (Aula-4 [21]).
- Figuras de atenção em 3 colunas: original | heatmap puro (`magma`, com colorbar) | overlay (`jet`, `alpha=0.55`), com título "✅/❌ Previsto: X, Confiança: Y%" (Aula-4 [27]).
- Por cabeça (DINO [17]): grid 2×3, overlay `inferno` com `alpha=0.55` e normalização min-max por mapa.

---

## 2. Implementações do zero existentes nas aulas

### 2.1 `Aula-2/aula_02_transformers.ipynb` (Transformer seq2seq do zero, PyTorch puro)

| Classe | Célula | O que faz | Observações para a A1 |
|---|---|---|---|
| `PositionalEncoding` | [10] | PE **senoidal** fixo 1D (`register_buffer`), soma + dropout | O ViT usa PE **aprendível**; o Desafio 3 ([33]) pede exatamente trocar por `nn.Embedding(max_len, d_model)` treinável |
| `ScaledDotProductAttention` | [12] | `softmax(QKᵀ/√d_k + M)V`; retorna `(output, attn_weights)`; máscara `masked_fill(mask == 0, -1e9)` | O **dropout é aplicado nos pesos antes de retorná-los**: o teste "linhas somam 1" só vale em `eval()` ou com `dropout=0`. Convenção de máscara: 0 = bloqueado (em `F.scaled_dot_product_attention`, máscara bool `True` = atende) |
| `MultiHeadAttention` | [14] | `w_q, w_k, w_v, w_o = nn.Linear(d_model, d_model, bias=False)`; `view(B, -1, h, d_k).transpose(1, 2)`; concat via `transpose(1, 2).contiguous().view(B, -1, d_model)`; retorna `(w_o(out), attn_weights)` | Projeção **fundida** (uma matriz d×d fatiada por reshape). É matematicamente equivalente a projeções independentes por head (cada head usa um bloco de colunas), mas **não é literal** em relação à rubrica 2.1 ("projeções independentes por attention head") |
| `PositionwiseFeedForward` | [16] | `Linear(d, d_ff) → GELU → Dropout → Linear(d_ff, d)` | MLP de 2 camadas, OK para a rubrica 2.3 |
| `EncoderLayer` | [18] | **Pre-LN**: `x + drop(MHA(LN(x)))`, depois `x + drop(FFN(LN(x)))` | Descarta os pesos (`attn_out, _ = ...`): para os mapas de atenção é preciso guardá-los (ex.: `self.last_attn`) |
| `Encoder` | [18] | Embedding × √d + PE + N camadas + LN final | Substituir Embedding de tokens por PatchEmbedding |
| `DecoderLayer`/`Decoder`/`Seq2SeqTransformer` | [20], [22] | Cross-attention, máscaras causal/padding | Não se aplicam à A1 |
| Heatmap de cross-attention | [27], [30] | `cross_attns[-1][0, 0]` (última camada, **head 0**), `sns.heatmap(cmap='Blues', annot=True)` | Padrão de "1 head" explícito |
| Comparação com `nn.Transformer` | [32] | Só imprime as contagens de parâmetros e afirma "equivalência" | Sem assert numérico: o teste contra `F.scaled_dot_product_attention` na A1 é contribuição nossa |

Treino ([24]): `AdamW(lr=8e-4, weight_decay=1e-4)`, `CosineAnnealingLR(T_max=15, eta_min=1e-5)`, `clip_grad_norm_(1.0)`, 100 épocas.

### 2.2 `Aula-3/aula_03_transformer_pytorch.ipynb`
Seq2seq com `torch.nn.Transformer` nativo ([6]), com checkpointing e retomada ([8]–[10]). Serve só como padrão de **checkpoint/retomada** (convenção do plano §0.2).

### 2.3 ViT do zero: **não existe em nenhum notebook do professor**
- Aula-4 e Aula-5 usam só modelos HF pré-treinados. O resumo oficial da Aula 4 (`Aula-4/Vision Transformers.md`) diz "projetar um ViT do zero", mas o notebook não faz isso.
- **Géron cap. 16, pp. 6–7** ("Implementing a ViT from scratch using PyTorch"):
  - `PatchEmbedding` = `nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)` → `flatten(2)` → `transpose(1, 2)`.
  - `ViT`: `cls_token = nn.Parameter(randn(1,1,E)*0.02)`, `pos_embed = nn.Parameter(randn(1,1+L,E)*0.02)`, dropout, `nn.TransformerEncoder(nn.TransformerEncoderLayer(..., activation="gelu", batch_first=True))`, `LayerNorm(Z[:,0])`, `Linear(E, C)`.
  - Teste: `ViT(...)(torch.randn(4,3,224,224))` → `[4, 1000]`.
  - Limitações para a rubrica: usa o encoder **nativo** (não o nosso bloco; rubrica 2.3 exige o bloco próprio como base do ViT); `TransformerEncoderLayer` é **Post-LN por padrão** (`norm_first=False`); não expõe os pesos de atenção.
- **Crash Course cap. 8, pp. 46–48**: usa `dl.ViT` da biblioteca `deeplay` (caixa-preta), CIFAR-10 32×32, patch 4 (64 patches), `hidden_features=[384]*7`, 12 heads, `Adam(lr=1e-3, wd=5e-5)`, 100 épocas → ~70% val. A nota da p. 47 justifica o patch embedding por Conv2d (kernel = stride = patch). O MHA do zero do livro está na Listing 8-11 (p. 26) e o encoder layer na Listing 8-19 (p. 35).
- **Receita para a A1** (componentes da Aula-2 + esqueleto do Géron): SDPA [12] → MHA reescrito com projeções por head → FFN [16] → `EncoderLayer` Pre-LN [18] que guarda a atenção → `PatchEmbedding` Conv2d + CLS + pos_embed aprendível (Géron) → LN final no CLS → head linear.

### 2.4 O que falta para a rubrica literal (2.1, 2.3, 2.4, 3.1)
1. **MHA com projeções independentes por head** (`nn.ModuleList` de `Linear(d_model, d_k)` para Q, K e V, ou uma classe `AttentionHead`) mais `torch.cat(heads, dim=-1)` e `W_O`. Teste opcional: equivalência com a versão fundida da Aula-2.
2. **Testes (asserts)**, inexistentes nas aulas: shapes; soma das linhas = 1; `torch.allclose` contra `F.scaled_dot_product_attention` (com dropout 0); forward do ViT; contagem de parâmetros.
3. **Blocos que retornam/guardam a atenção** por camada e por head (`[B, h, 1+L, 1+L]`).
4. **Demonstração de PE por permutação**: não existe no material (só o argumento verbal, ver §5.5).
5. **Treino do ViT do zero no domínio**: o material só tem CIFAR-10 via deeplay (Crash Course). Hiperparâmetros de referência na §3.3.

---

## 3. Fine-tuning de ViT/DeiT com Hugging Face

### 3.1 Checkpoints e configuração usados nas aulas

| Notebook | Checkpoint | Classe | Dados | Hiperparâmetros |
|---|---|---|---|---|
| `Aula-4/aula_04_vision_transformers.ipynb` [8], [11], [19] | `google/vit-base-patch16-224` (IN-21k → IN-1k, head de 1000) | `ViTForImageClassification`, `ignore_mismatched_sizes=True`, `attn_implementation="eager"` | `AI-Lab-Makerere/beans` (3 classes, splits prontos) | `lr=3e-5`, `bs=16`, `epochs=3`, `weight_decay=0.01`, `warmup_steps=100`, `eval/save_strategy="epoch"`, `load_best_model_at_end=True`, `metric_for_best_model="accuracy"`, `remove_unused_columns=False`, `report_to="none"`, `logging_steps=15` |
| `Aula-5/aula_05_advanced_vision_transformers.ipynb` [9], [13], [17] | `google/vit-base-patch16-224-in21k` (sem head de classificação) | `AutoImageProcessor`, `AutoModelForImageClassification(num_labels, id2label, label2id)` | `timm/eurosat-rgb`, subsets 2000/400/200 com `.map(batched=True)` | `lr=5e-5`, `bs=16`, `epochs=1`, `wd=0.01`, `logging_steps=25`, `load_best_model_at_end=True` |
| `Aula-5/aula_05_deit_distillation_transfer_learning.ipynb` [8], [11], [14], [16] | Aluno `facebook/deit-tiny-distilled-patch16-224` (5,5M), professor `tangocrazyguy/resnet-50-finetuned-cats_vs_dogs` congelado | `AutoModelForImageClassification` → `DeiTForImageClassificationWithTeacher` (`cls_classifier` e `distillation_classifier`) | `Bingsu/Cat_and_Dog`, 1200 treino / 300 teste | Loop manual: `AdamW(lr=5e-5, wd=0.01)`, 5 épocas, bs 16; **hard distillation** `0.5·CE(z_cls, y) + 0.5·CE(z_dist, argmax z_teacher)`; inferência = média dos logits das duas cabeças [18] |
| `Aula-5/aula_05_dino_self_supervised_vision.ipynb` [5], [9], [11] | `facebook/dino-vits16` (congelado, `attn_implementation="eager"`); `facebook/dinov2-small-imagenet1k-1-layer` via `pipeline` | `AutoModel` | Cat_and_Dog | Sem treino: classificador por média mais próxima sobre o CLS normalizado em L2 |
| Géron cap. 16, pp. 8–9 | `google/vit-base-patch16-224-in21k`; DeiT `facebook/deit-base-distilled-patch16-224` (p. 10) | `ViTForImageClassification(num_labels=37)`, `AutoImageProcessor(use_fast=True)` | Oxford-IIIT Pet | `bs=16`, 3 épocas, `remove_unused_columns=False` (o WARNING da p. 9 explica o porquê), collate com `do_convert_rgb=True`; ~91,8% (ViT), ~94,4% (DeiT) |
| Crash Course cap. 8, pp. 57–59 (Listings 8-41 a 8-44) | `google/vit-base-patch16-224-in21k` | `ViTModel` + `nn.Linear` próprio sobre `last_hidden_state[:, 0]` | CIFAR-10 | Head manual (alternativa ao `ForImageClassification`) |

### 3.2 Padrões de pré-processamento
- Processor: `ViTImageProcessor.from_pretrained(MODEL_NAME)` (Aula-4 [15]) ou `AutoImageProcessor` (Aula-5 [9]). A Aula-5 [8] explica o motivo: tamanho 224 (14×14 = 196 patches), `[C,H,W]` e `image_mean`/`image_std` do pré-treino. Não recriar a normalização à mão.
- **Tons de cinza**: todos os notebooks fazem `img.convert("RGB")` antes do processor, o que replica o canal L em 3 canais. É o padrão a seguir no NEU.
- Aplicação sob demanda com `dataset.with_transform(transform_batch)` (Aula-4 [15], DeiT [9]) ou estática com `.map(batched=True, remove_columns=["image"])` (Aula-5 [11]).
- Collate: `torch.stack` de `pixel_values` + `labels` long (Aula-4 [15], Aula-5 [17]).
- Avaliação: `trainer.predict(test)` → `test_accuracy` / `test_f1_macro` (Aula-4 [21]); `trainer.evaluate(eval_dataset=test_ds)` (Aula-5 [19]).
- Inferência de produção: `pipeline("image-classification", model=OUTPUT_DIR)` depois de `trainer.save_model` + `processor.save_pretrained` (Aula-4 [19], [23]).

### 3.3 Hiperparâmetros de referência para o ViT **do zero**
- Crash Course (pp. 49–55, Listings 8-36 a 8-39): o mesmo ViT com **CutMix** + `GradualWarmup(5 épocas)` + `CosineAnnealingLR(T_max=200, eta_min=1e-5)`, 700 épocas → ~90% no CIFAR-10 (contra ~70% sem CutMix). Mostra a "fome de dados" e o efeito da regularização.
- Aula-4 [13] implementa `apply_cutmix_pil` **só como demonstração visual**: o CutMix não entra no `Trainer`. Fórmula no markdown [12] e no Slide 12.
- Aula-2 [24]: AdamW + cosine + gradient clipping (padrão do professor para treino do zero).

### 3.4 Desafios propostos pelo professor que viram ablações possíveis
- Aula-4 [28]: (1) backbone congelado (`model.vit.parameters()` com `requires_grad=False`) vs fine-tuning completo; (2) patch 16 vs 32 (`google/vit-base-patch32-224`); (3) **Swin** `microsoft/swin-tiny-patch4-window7-224` no mesmo `Trainer`.
- Aula-2 [33], Desafio 3: PE aprendido vs senoidal, comparando a convergência.

---

## 4. Extração e visualização de atenção (como o professor fez)

### 4.1 Pré-requisito HF
Carregar com `attn_implementation="eager"` (Aula-4 [11], [19]; DINO [5]) e chamar `model(**inputs, output_attentions=True)`. `outputs.attentions` é uma tupla de L tensores `[B, h, 1+N, 1+N]`. A nota da DINO [4] justifica o `eager`: com as implementações SDPA/Flash, as matrizes não ficam disponíveis.

### 4.2 Attention Rollout (Abnar & Zuidema 2020): Aula-4 [24]–[27], Slide 13
`compute_attention_rollout(attentions)`:
1. Por camada, média das h heads → `[T, T]`.
2. `A = (W + I)/2`, renormalizado por linha (residual).
3. `R = A_L · … · A_1` (acumulado com `torch.matmul(a, result)`).
4. `R[0, 1:]` → reshape `14×14` → min-max.

`overlay_attention`: resize bicúbico para 224, colormap `jet`, `alpha=0.55`. Amostras: índices fixos do teste, uma por classe.

### 4.3 Atenção de **uma head** (a rubrica 3.2 exige ≥1 head): DINO [12]–[17]
`get_dino_attention_maps`:
- `outputs.attentions[-1]` (última camada) → `[0, :, 0, 1:]` = atenção CLS→patches por head `[h, 196]` → `reshape(h, 14, 14)` → `F.interpolate(size=224, mode="bicubic")`.
- [15]: média das heads + overlay `magma`. [17]: as 6 heads lado a lado ("especialização das cabeças", réplica da Fig. 16-8 do Géron, p. 17).
- Para o ViT **do zero**: o mesmo recorte `[:, :, 0, 1:]` aplicado ao tensor guardado pelo nosso `EncoderLayer`. O grid passa a ser `(img/patch)²` (ex.: 128/16 → 8×8).

### 4.4 Outros exemplos
- Crash Course p. 55 (Listing 8-40): `logs["attention_output"][1][:, 0, 1:].reshape(4, 8, 8)` (última camada, CLS→patches), `skimage.transform.resize` + `cmap="hot", alpha=0.5`.
- Aula-2 [30]: heatmap de matriz token×token com `annot=True`. Útil se quisermos mostrar a matriz 1+N × 1+N de uma head.

### 4.5 Narrativa de interpretação usada pelo professor
- Aula-4, Slide 13 (falas): as camadas iniciais têm atenção dispersa (textura/iluminação); as intermediárias convergem para contornos de alto contraste; as finais concentram-se nos traços distintivos. No ViT a interpretabilidade é "nativa", diferente do Grad-CAM da CNN.
- Aula-5, Slide 14 (falas): o ViT supervisionado tem atenção difusa e correlações espúrias no fundo; o DINO segmenta o objeto e cada head se especializa em uma parte.
- Para o NEU (texturas, sem objeto centrado), é provável que o mapa **não** isole um "objeto". Discutir isso contra a narrativa acima é uma boa análise (rubricas 2.2 e 3.2).

---

## 5. Conceitos para os textos (com fontes)

### 5.1 DeiT (rubrica 3.4)
- **Problema resolvido**: a fome de dados. O ViT original só supera CNNs com JFT-300M; o DeiT fica competitivo **só com ImageNet-1k**, sem dados externos.
- **Mecanismo**: um token `[DIST]` aprendível ao lado do `[CLS]`, cada um com sua head; professor CNN congelado (RegNetY-16GF no paper). Hard distillation: a head DIST aprende `argmax` do professor; loss `½CE(cls, y) + ½CE(dist, ŷ_T)`; na inferência, média das duas heads, sem a CNN. A CNN transfere o **viés indutivo** de localidade via rótulos. Também usa regularização agressiva (RandAugment, Mixup, CutMix, Repeated Aug, Stochastic Depth).
- Fontes: Aula-5 Slide 5 (falas l. 86–104), plano Aula-5 §1; Aula-4 Slide 14; notebook DeiT [10], [15], [17]; Géron cap. 16 p. 10 (Fig. 16-4).
- Atenção: o Géron descreve a head de destilação com **soft targets**, enquanto o notebook e as falas usam **hard**. O paper testa as duas e a hard vence. Ver §7.

### 5.2 Swin (rubrica 3.4)
- **Problemas resolvidos**: (a) custo **quadrático** da atenção global em alta resolução, trocado por **linear** (W-MSA em janelas M×M, M=7); (b) arquitetura **isotrópica**/escala única do ViT, trocada por uma **hierarquia** (patch 4×4, *Patch Merging*, mapas H/4…H/32) compatível com FPN, detecção e segmentação.
- **Janelas deslocadas** (SW-MSA), com deslocamento de ⌊M/2⌋ em blocos alternados, conectam janelas vizinhas. Implementação eficiente: *cyclic shift* (`torch.roll`) + máscara −100 + *reverse shift*.
- Complexidade: Ω(MSA) = 4hwC² + 2(hw)²C; Ω(W-MSA) = 4hwC² + 2M²hwC (plano Aula-5, Slide 9). Exemplo do Géron: 784 tokens × 49 = 38.416 scores, contra 784² = 614.656.
- Fontes: Aula-5 Slides 9–11 (falas l. 169–238); Aula-5 Slides 3 e 6 (as 3 barreiras do ViT); Géron cap. 16 pp. 13–14 (Fig. 16-6). O PVT (Slides 6–8; Géron p. 11) é o contraponto hierárquico com SRA.

### 5.3 ViT vs CNN (rubricas 3.5 e 3.6)
- **Viés indutivo**: a CNN assume localidade e equivariância à translação e aprende rápido com poucos dados. O ViT tem viés quase nulo e campo receptivo global desde o bloco 1, com teto mais alto, mas precisa de muitos dados (Aula-4 Slide 3, falas l. 43–61; Géron cap. 16 pp. 5–6, NOTE sobre inductive bias; Crash Course p. 49).
- **Fome de dados**: resultados de Dosovitskiy. Com IN-1k a ResNet vence, com IN-21k empatam, com JFT-300M o ViT vence (Aula-4 Slide 11, falas l. 194–213; Aula-5 Slide 3).
- **Quando usar cada um** (Aula-5 Slide 17, matriz de decisão): classificação com alto throughput → DeiT/ConvNeXt; detecção/segmentação → Swin/PVT; edge/NPU → ConvNeXt/MobileNet; embeddings sem rótulo → DINOv2 congelado. ConvNeXt (Slide 16): parte do ganho do ViT vem do design macro e do protocolo de treino, não só da atenção.
- Aplicação ao NEU: dataset pequeno (1.800), texturas locais e repetitivas, em cinza. O viés de localidade da CNN tende a ajudar; o ViT do zero sofre com a fome de dados; o ViT pré-treinado compensa via transferência. Argumento a confirmar com a tabela.

### 5.4 Pré-treino BERT vs ViT (rubrica 2.5)
- **BERT** (auto-supervisionado em texto):
  - MLM: 15% dos tokens, regra 80/10/10, CE só nas posições mascaradas.
  - NSP: binário no `[CLS]`; `L = L_MLM + L_NSP`.
  - BookCorpus + Wikipedia (3,3B palavras).
  - Maximiza a verossimilhança do token dado o **contexto bidirecional** (representações contextuais) e, pelo NSP, a coerência entre sentenças no CLS.
  - O RoBERTa mostrou que o NSP ajuda pouco.
  - Fontes: Aula-3 Slides 8, 10, 12, 18 (falas l. 140–159, 181–196, 215–231, 344+); Géron cap. 15 p. 21 (Fig. 15-5).
- **ViT original** (supervisionado): classificação em ImageNet-21k (14M imagens, 21.843 classes) ou JFT-300M e depois fine-tuning. Maximiza a verossimilhança do **rótulo da imagem inteira**, ou seja, features discriminativas alinhadas às classes (Aula-4 Slide 11; Aula-5 [12] "21.843 classes"; Géron cap. 16 pp. 5, 9).
- **Alternativas auto-supervisionadas em visão**:
  - **MAE**: o análogo direto do MLM. Mascara **75%** dos patches (contra 15% no BERT, por causa da redundância espacial), o encoder vê só os 25% visíveis e o decoder leve reconstrói pixels com MSE (Aula-5 Slide 15, falas l. 316–333; Géron p. 18).
  - **DINO**: auto-destilação aluno/professor EMA, multi-crop (local→global), centering + sharpening, sem negativos. Maximiza a **concordância/invariância** entre visões da mesma imagem (Aula-5 Slides 12–13; Géron pp. 15–17).
- Ponte BERT↔ViT: o `[CLS]` e o encoder-only são herdados do BERT (Aula-4 Slide 8; Aula-3 Slide 3, falas l. 50); o plano da Aula-3 cita o MAE como o equivalente do MLM.

### 5.5 Positional encoding (rubrica 2.4)
- A atenção é "operação sobre conjuntos": sem PE, as saídas dos tokens são **equivariantes** a permutações (permutar a entrada só permuta a saída), e a saída do CLS (ou um pooling) é **invariante**. Logo, a posição dos patches se perde ("Paradoxo do Saco de Palavras").
- Fontes: Aula-2 Slide 11 (falas l. 180–190, exemplo "YOUR CAT IS A LOVELY CAT"); Aula-4 Slide 8 (falas: "a atenção é invariante à permutação"); Géron cap. 15 pp. 7–8 ("position-agnostic").
- Nas falas o professor diz "invariante". O termo preciso é equivariante para os tokens e invariante para o CLS; a demonstração do plano (§2b) mostra as duas coisas.
- Senoidal vs aprendido: Aula-2 Slide 12, célula [10]. O ViT usa PE **1D aprendível** `[1, 197, 768]` (Aula-4 [10]–[11]); o paper testou 2D sem ganho (Aula-4 Slide 8, "curiosidade de pesquisa").
- `z₀ = [x_class; x_p¹E; …; x_pᴺE] + E_pos` e as equações Pre-LN do bloco: plano Aula-4 §3.

### 5.6 Outros pontos citáveis
- Custo do pixel: 224² = 50.176 tokens → matriz de ~2,5·10⁹ elementos por head; com patches, 196 (Aula-4 Slide 2, plano §3).
- Patch embedding = `Conv2d(3, D, kernel=P, stride=P)` (Aula-4 Slide 7; Géron p. 6; Crash Course p. 47).
- Por que usar CLS em vez de média: evita que patches irrelevantes diluam a representação (Aula-4 Slide 8).
- MHA tem o mesmo custo de 1 head grande (d_k = d/h); concatena e aplica W_O (Aula-2 Slide 19).
- Pre-LN vs Post-LN (Aula-2 Slide 23; notebook [17]).

---

## 6. Mapeamento rubrica → material

| Rubrica | Exigência (texto do enunciado) | Material base | Gap |
|---|---|---|---|
| 2.1 | SDPA e MHA do zero, testáveis, com projeções independentes por head e concatenação | Aula-2 [12], [14], Slides 14, 17, 19; Crash Course Listing 8-11 | Projeção por head explícita + asserts |
| 2.2 | Heatmap de attention weights em ≥1 exemplo do domínio + interpretação escrita | Aula-4 [25]–[27]; DINO [13]–[17]; Aula-2 [30] | Interpretação específica de defeitos |
| 2.3 | TransformerEncoderBlock (FFN 2 camadas, LayerNorm, residual) como base do ViT | Aula-2 [16], [18], Slide 23; Aula-4 Slide 9 | Usar o bloco próprio no ViT (não o `nn.TransformerEncoder` do Géron) e guardar a atenção |
| 2.4 | PE aplicado aos tokens do ViT + por que a atenção sem PE não preserva posição | Aula-2 [10], Slides 11–12; Aula-4 Slide 8; Géron c.15 pp. 7–8 | Demonstração empírica por permutação (nova) |
| 2.5 | Diferenças entre o pré-treino de BERT e de ViT: o que cada um maximiza | Aula-3 Slides 8, 10, 12; Aula-4 Slide 11; Aula-5 Slides 12–15; Géron c.15 p. 21, c.16 pp. 15–18 | Só texto |
| 3.1 | Patch embedding, CLS aprendível, PE → ViT completo (imagem → logits) | Géron c.16 pp. 6–7; Aula-4 [10]–[11], Slides 6–8, 10 | Montar sobre os módulos próprios |
| 3.2 | ViT do zero treinado no domínio, mapas de ≥1 head, regiões emergentes por escrito | Crash Course pp. 46–56; DINO [13], [17] | Treino no NEU; análise por classe |
| 3.3 | Fine-tuning de ViT pré-treinado, head substituído, tabela contra o do zero | Aula-4 [19]–[21]; Aula-5 adv [13]–[19]; Géron pp. 8–9 | Mesmo split; medir tempo/VRAM |
| 3.4 | DeiT e Swin: o que resolvem que o ViT não resolve | Aula-5 Slides 3–5, 9–11; Aula-4 Slide 14; Géron pp. 10, 13–14 | Só texto (DeiT/Swin como experimento é opcional) |
| 3.5 | Quando ViT supera CNN e quando CNN é preferível, no domínio | Aula-4 Slides 3, 11, 15; Aula-5 Slides 3, 16, 17; Géron p. 5 | Baseline CNN com o mesmo split |
| 3.6 | Tabela ViT do zero vs pré-treinado + escolha de arquitetura com base nos dados | Aula-4 [21] (métricas); Aula-5 Slide 17 | Tabela com acc, F1 macro, params, tempo, VRAM |

---

## 7. Lacunas e armadilhas encontradas no material

1. **Não há ViT do zero em PyTorch puro nas aulas.** As únicas referências são o Géron (encoder nativo, Post-LN, sem pesos de atenção) e o Crash Course (`deeplay`). A montagem é trabalho nosso.
2. **O MHA da Aula-2 é fundido**: não atende à letra da rubrica 2.1 (ver §2.4 e a Dúvida 1).
3. **Nenhum notebook testa numericamente** SDPA/MHA. A célula [32] da Aula-2 afirma "equivalência" só pela contagem de parâmetros.
4. **Dropout nos pesos retornados** pelo SDPA da Aula-2: extrair e testar a atenção sempre em `eval()`.
5. **Convenção de máscara** oposta entre o SDPA da Aula-2 (`mask==0` bloqueia) e o `F.scaled_dot_product_attention` (bool `True` atende). O ViT não precisa de máscara, mas o teste de equivalência sim, se incluir máscara.
6. **Nenhum notebook faz split estratificado próprio**: todos usam os splits prontos do HF ou `shuffle(seed=42).select(range(n))`. O NEU exige split estratificado fixo (ex.: `sklearn.train_test_split(stratify=...)`), salvo e reutilizado por todos os modelos.
7. **Nenhum notebook mede tempo de treino por modelo nem VRAM de pico**: só imprimem a VRAM total. Usar `torch.cuda.reset_peak_memory_stats()` / `max_memory_allocated()` e `time.perf_counter()` (ou `train_result.metrics["train_runtime"]` do `Trainer`).
8. **CutMix não é usado no treino** da Aula-4 (é só demonstração). No NEU, o CutMix pode colar uma região sem defeito e manter o rótulo misto proporcional à área. Isso é questionável para defeitos localizados (inclusion, pitted), mas razoável para defeitos que cobrem a imagem toda (crazing, rolled-in scale). Justificar se for usado.
9. **Inconsistência no notebook DeiT**: o markdown [12] fala em ResNet-18 e o código [14] usa ResNet-50 (`tangocrazyguy/...`). O professor recebe os `pixel_values` normalizados pelo processor do DeiT, não pelo seu próprio. `DeiTForImageClassificationWithTeacher` não calcula a loss de destilação (o notebook faz a loss à mão). Para fine-tuning simples, usar `facebook/deit-small-patch16-224` (não destilado) com `DeiTForImageClassification`/`AutoModelForImageClassification`.
10. **Números dos slides a conferir no paper antes de citar**:
    - Aula-5 Slide 3: ViT-B em IN-1k = 79,9% vs "ResNet-50 bem treinada 83,2%" (valor alto para uma ResNet-50).
    - Aula-5 Slide 5: DeiT-B "85,2%" (no paper, esse valor corresponde ao DeiT-B⚗ a 384 px).
    - Aula-4 Slide 14: ViT-B/16 81,2%, ViT-H/14 88,5%.
    - Citar o paper original, não o slide.
11. **Termo "invariante à permutação"** nas falas: precisar como equivariância para os tokens (§5.5).
12. **O material não trata imagens em cinza nem texturas industriais**: a única adaptação é `convert("RGB")`. A discussão de domínio (NEU) é original nossa.
13. **200×200 do NEU não é múltiplo de 16**: o material só usa 224 (processor HF) ou 32 (CIFAR). O ViT do zero precisa de uma resolução escolhida (ver Dúvida 4).

---

## 8. Dúvidas para o Gilmar

1. **MHA (rubrica 2.1)**: implementar heads **explicitamente independentes** (`ModuleList` de projeções `d_model→d_k` + `torch.cat` + `W_O`), com teste de equivalência contra a versão fundida da Aula-2? É a opção literal e a que recomendo. A alternativa é manter a versão fundida e só explicar a equivalência.
2. **Checkpoint pré-treinado**:
   - `google/vit-base-patch16-224-in21k` (Aula-5, Géron e Crash Course; pré-treino supervisionado puro no IN-21k, sem head de 1000 classes; casa com o texto 2.5);
   - `google/vit-base-patch16-224` (Aula-4; exige `ignore_mismatched_sizes=True`);
   - `facebook/deit-small-patch16-224` (22M, mais leve na T4, e ancora o texto do DeiT).
   - Qual será o principal? Um segundo modelo entra como extra?
3. **Fine-tuning com o `Trainer` do HF** (padrão do professor) ou **loop manual** igual ao do ViT do zero (tempo e VRAM medidos do mesmo jeito, mesma augmentation)?
4. **Resolução do ViT do zero**: 128 com patch 16 (64 tokens, rápido, perde detalhe), 224 com patch 16 (196 tokens, igual ao pré-treinado) ou 200 nativo com patch 20/10? O pré-treinado fica em 224 de qualquer forma.
5. **CutMix no ViT do zero**: seguir a ênfase do professor (Aula-4 Slides 11–12, Crash Course), com a ressalva de domínio da §7.8, ou ficar só com flips e rotações de 90° como está no plano?
6. **Mapas de atenção do pré-treinado**: só 1 head da última camada (padrão DINO, cumpre a rubrica) ou também o **rollout** (padrão da Aula-4, mais robusto)? Recomendo os dois para o pré-treinado e 1 head + rollout para o do zero.
7. **Extras opcionais** propostos nos desafios do professor: Swin-tiny (`microsoft/swin-tiny-patch4-window7-224`) como terceiro modelo na tabela, e/ou backbone congelado vs fine-tuning completo? Custam tempo de GPU, mas reforçam as rubricas 3.4 e 3.5 com dados em vez de só texto.
