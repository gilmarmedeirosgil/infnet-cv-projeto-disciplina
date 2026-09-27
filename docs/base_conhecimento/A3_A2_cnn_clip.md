# Base de conhecimento — A3 (CNN pré-treinada) e A2 (CLIP no ADS-16)

Fontes lidas (caminhos relativos a `Visao_Computacional_com_CNNs_Transformers/`):
- `Aula-1/plano_aula.md`, `Aula-1/aula_01_apresentacao.pdf` (slides 14–17, só imagem), `Aula-1/aula_01_cnn_architectures.ipynb`, cabeçalhos/setup de `aula_01_faster_rcnn.ipynb`, `aula_01_yolo_detection.ipynb`, `aula_01_semantic_segmentation.ipynb`, livro HOML cap. 12 (PDF, seções "Data Augmentation" e "Pretrained Models for Transfer Learning").
- `Aula-6/plano_aula.md`, `Aula-6/Modelos Multimodais: CLIP.md`, `Aula-6/aula_06_apresentacao.pdf` (slides 3, 4, 10, 11, 12), `Aula-6/aula_06_clip_openai.ipynb` (código + outputs salvos), `Aula-6/aula_06_clip_fine_tuning_huggingface.ipynb` (código + outputs), HOML cap. 16 (seção CLIP).
- Pontual: `Aula-3/aula_03_bert_fine_tuning.ipynb`, `Aula-3/Fine Tuning de BERT.md`, `Aula-3/03_plano_aula.md`, `Aula-3/03_falas_apresentador.md`, `Aula-3/aula_03_transformer_pytorch.ipynb` (célula 6, máscaras).

Convenção: `nb[i]` = índice 0-based da célula no `.ipynb` (ordem de `cells`). **[externo]** marca informação que não está no material da aula (conhecimento geral / docs oficiais) e deve ser conferida em execução.

---

## 1. Padrões transversais do professor (Allan Spadini)

**Cabeçalho (célula 0, markdown)**: título "Visão Computacional com CNNs e Transformers / Faculdade Infnet — Pós-Graduação", badge "Open in Colab" apontando para `github.com/allanspadini/curso-vision-transformers-infnet`, bloco "Objetivos de Aprendizagem" numerado. Aula 6 explicita a metodologia: **Situação-Problema → Solução de Engenharia → Teoria Rigorosa** (Aula 1: + "Aplicação Prática"). Os notebooks não têm cabeçalho de RAM/VRAM/tempo (nosso enunciado exige; ver PLANO §0.2).

**Setup (sempre a 1ª/2ª célula de código)** — padrão idêntico em todos os notebooks:
```python
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True   # presente em Aula-1 cnn e Aula-6 fine-tuning; ausente em clip_openai
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# imprime get_device_name(0) e total_memory/(1024**3) em GB
```
- `pip install -q ...` em célula própria (`!pip install -q kagglehub` nas Aula-1 detecção/segmentação; `transformers datasets torchvision scikit-learn ...` na Aula-6).
- Download Kaggle: `kagglehub.dataset_download("owner/slug")` sem autenticação explícita (datasets públicos). Não há uso de `google.colab.userdata`/Secrets nem `drive.mount` em nenhum notebook lido → nosso padrão de Secrets + Drive é adição nossa.
- Seções numeradas `## N. Título` (Aula 1) ou `## 🛠️ Passo N: Título` (Aula 6), com emojis nos títulos e prints (`✅`, `📊`, `🔥`). Markdown antes de cada bloco de código explicando o "porquê".
- Paleta dos gráficos: `#0A345D` (azul Infnet, títulos), `#1BB5D8` (ciano), `#FF7043`/`#DC2626` (erro), `#15803D` (acerto). `plt.suptitle(..., fontweight='bold')`, `grid(linestyle=':')`.
- Prints de verificação: parâmetros treináveis vs totais em %, shapes de tensores comentados (`# [4, 512]`), checagem algébrica (diferença manual vs nativo).
- Seção final "Desafios e Atividades Práticas Propostas" (Aula 1 nb[21]) — é daí que vem o código de fine-tuning com LR diferencial.
- Obs.: glifos emoji em títulos de matplotlib geram `UserWarning: Glyph ... missing from font DejaVu Sans` (visto nos outputs da Aula 6 nb[20], nb[24]) → **evitar emoji em títulos de figuras** nos nossos notebooks.

---

## 2. A3 — CNN pré-treinada, transfer learning por feature extraction

### 2.1 Conceitos cobertos em aula
- **Weights Enum API** (slide 16): `weights = ResNet34_Weights.DEFAULT; model = resnet34(weights=weights)` substitui `pretrained=True`. "Regra de Ouro: sempre utilize `weights.transforms()` para garantir que a imagem de teste receba o mesmo tratamento estatístico usado no treino original" → evita "bugs silenciosos de pré-processamento".
- **Feature extraction vs fine-tuning** (slide 17, lab `TransferLearningSimulator`): 3 modos — *Feature Extraction (backbone fixo)*, *Fine-Tuning Parcial (layer4 + fc)*, *Fine-Tuning Completo*. Mostra ResNet-34 por blocos: conv1+bn1 "low-level edges/colors", layer1 "texturas e padrões básicos", layer2 "motivos e partes simples", layer3 "estruturas de objetos complexos", layer4 "semântica de alto nível / classes", fc "nova cabeça". Dica do slide: "Bloquear camadas iniciais preserva detectores de borda universais e reduz o tempo de treinamento em mais de 70%."
- **Seleção de arquitetura** (slide 14, Tabela 12-3 TorchVision; Top-1 / Params / GFLOPs):
  | Modelo | Top-1 | Params | GFLOPs |
  |---|---|---|---|
  | MobileNet v3 small | 67.7% | 2.5M | 0.1 |
  | **EfficientNet B0** | 77.7% | 5.3M | 0.4 |
  | DenseNet 121 | 74.4% | 8.0M | 2.8 |
  | EfficientNet v2 small | 84.2% | 21.5M | 8.4 |
  | **ResNet 34** (destacado como "padrão da indústria") | 73.3% | 21.8M | 3.7 |
  | ConvNeXt Tiny | 82.6% | 28.6M | 4.5 |
  | ResNet 152 | 82.3% | 60.2M | 11.5 |
  ResNet-50 **não** aparece na tabela do slide. [externo, torchvision docs] ResNet50 `IMAGENET1K_V2`: 25.6M params, 4.09 GFLOPs, 80.86% top-1 (V1: 76.13%). Confirmar com `weights.meta` no notebook.
- **VRAM inferência vs treino** (slide 15, ResNet-50, batch 64): inferência ~120 MB (O(1)); treino ~8.5 GB, dos quais ativações ~7.2 GB (84%), pesos ~100 MB, gradientes ~100 MB, Adam (m,v) ~200 MB. "Regra de ouro: OOM → reduza `batch_size` ou use gradient checkpointing." **Argumento-chave para 1.5**: em feature extraction com backbone congelado (`requires_grad=False` em todo o backbone), o autograd não guarda as ativações do backbone para o backward → pegada muito menor, batch maior cabe na T4 (15 GB; output real da Aula 6: "VRAM Total: 14.56 GB").
- HOML cap. 12: congelar tudo, treinar só o head por algumas épocas (~90% no Flowers102 só com o head), depois opcionalmente descongelar e baixar LR ~10×; LR diferencial por parameter groups; "vale procurar modelos pré-treinados em imagens similares" (satélite → TorchGeo, médico → MONAI) — ImageNet ajuda pouco em domínios distantes.

### 2.2 Código reaproveitável — `Aula-1/aula_01_cnn_architectures.ipynb`
| Célula | Conteúdo | Adaptação para A3 |
|---|---|---|
| nb[2] | imports + seed 42 + device + print GPU/VRAM | copiar; acrescentar Drive/Secrets/kagglehub |
| nb[6] | `unnormalize(tensor, mean, std)` para plot | trocar para média/std ImageNet (0.485,0.456,0.406)/(0.229,0.224,0.225) |
| nb[10] | `train_one_epoch(...)` e `evaluate(...)` (`@torch.no_grad`, loss ponderada por batch, acc %) | reaproveitar direto; `evaluate` deve também devolver `y_true, y_pred` para acc por classe e matriz de confusão |
| nb[12] | loop de épocas com `history = {'train_loss','train_acc','val_loss','val_acc'}`, `AdamW(lr=1e-3, wd=1e-4)`, `CosineAnnealingLR(T_max=EPOCHS)` | base do "único treino" |
| **nb[14]** | **núcleo do 1.1**: `weights.transforms()`, `resnet34(weights=...)`, `for p in model.parameters(): p.requires_grad=False`, `model.fc = nn.Linear(model.fc.in_features, n_classes)`, print de params treináveis vs totais (%) | trocar modelo; para EfficientNet o head é `model.classifier[1]` (Sequential(Dropout, Linear(1280,1000))) [externo]; manter o print |
| nb[16] | otimizador **só com `model_tl.fc.parameters()`** + loop | idem com o novo head |
| nb[18] | curvas val acc/loss lado a lado (scratch vs TL) | adaptar para train vs val do único modelo |
| nb[20] | inferência visual com top-3 softmax, título verde/vermelho acerto/erro, desnormalização ImageNet | reaproveitar para "exemplos de erros" (filtrar `pred != true`) |
| nb[21] | desafio: fine-tuning `layer4` com `AdamW([{layer4, lr 1e-5}, {fc, lr 1e-3}])` | citar na discussão 1.4 (não executar: A3 pede um único treino) |

Split: o notebook usa CIFAR-10 com split fixo train/test (sem val). **Não há código de split estratificado nas aulas** → usar `sklearn.model_selection.train_test_split(stratify=y, random_state=42)` duas vezes (70/15/15) sobre a lista de caminhos.

Dataset a partir de pastas: `aula_01_semantic_segmentation.ipynb` nb[14] tem um `Dataset` custom com `transforms.Normalize(mean=[0.485,...], std=[0.229,...])`; para classificação por pastas, `torchvision.datasets.ImageFolder` + `Subset` por índices é o caminho mais curto (não mostrado em aula).

### 2.3 Normalização e augmentations mostradas
- **Aula 1 nb[4] (scratch, CIFAR)**: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip()`, `ToTensor()`, `Normalize(CIFAR_MEAN, CIFAR_STD)`. Teste: só `ToTensor + Normalize`.
- **Aula 1 nb[14] (transfer)**: **nenhuma augmentation** — usa `pretrained_transforms = weights.transforms()` tanto em treino quanto em teste (resize 256 → center crop 224 → normalização ImageNet para ResNet34 [externo: valores exatos impressos pela célula]).
- **HOML cap. 12** (augmentation *antes* da normalização ImageNet):
  ```python
  T.Compose([T.RandomHorizontalFlip(p=0.5), T.RandomRotation(degrees=30),
             T.RandomResizedCrop(size=(224,224), scale=(0.8,1.0)),
             T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
             T.ToImage(), T.ToDtype(torch.float32, scale=True),
             T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
  ```
  Princípios citados pelo livro (úteis para 1.3): variações devem ser realistas ("um humano não deve distinguir a imagem aumentada"); ruído branco não ajuda; **flip horizontal não serve para texto e objetos assimétricos**; augmentation também ajuda em desbalanceamento; TTA existe; `AutoAugment` é opção pronta.
- Mapeamento para a análise 3.2 do enunciado: (a) geométrica = flip/rotação/`RandomResizedCrop`; (b) cor = `ColorJitter`/grayscale; (c) escala = `RandomResizedCrop(scale=...)`; (d) normalização = **deve** ser a do ImageNet do checkpoint (`weights.transforms()`), não estatísticas do dataset, porque o backbone congelado espera essa distribuição de entrada.

### 2.4 Feature extraction vs fine-tuning (material para 1.4)
Argumentos presentes no material:
- Feature extraction: só o head treina (Aula 1 nb[14] imprime a fração treinável; para ResNet-34 com 10 classes são 512·10+10 = 5.130 pesos, ~0.02% de 21.3M); converge rápido; menor VRAM; menor risco de overfitting em dataset pequeno; preserva features genéricas (slide 17).
- Fine-tuning parcial/completo com LR diferencial (Aula 1 nb[21]; HOML): útil quando o domínio se afasta do ImageNet ou há dados suficientes; LR do backbone ~10–100× menor que a do head.
- Aula 6 slide 12 (mesmo raciocínio, aplicado ao CLIP): linear probe = custo muito baixo, "zero esquecimento"; full fine-tuning = custo alto e risco de *catastrophic forgetting*/perda de robustez OOD; recomendação: "inicie SEMPRE com linear probe antes de arriscar fine-tuning completo".
- Regra prática a escrever (síntese nossa, coerente com o material): dataset pequeno + domínio próximo do ImageNet → feature extraction; pequeno + domínio distante → FT parcial das últimas camadas com LR baixa e augmentation forte; grande + distante → FT completo; grande + próximo → FT completo ou parcial (ganho marginal).

---

## 3. A2 — CLIP no ADS-16 (sem treino)

### 3.1 Conceitos (4.1)
- **Two-tower** (slide 3; nb[5]): image encoder ViT-B/32 (ou ResNet-50/101) → `[N, d_v=768]` → projeção `W_v: Linear(768→512)`; text encoder Transformer, entrada `[N, L=77 tokens]`, representação no **token [EOS] (fim da sentença)** `[N, d_t=512]` → `W_t: Linear(512→512)`; ambos L2-normalizados na hiperesfera `S^{D-1}`, D=512. Sem co-atenção entre as torres (HOML).
- **Similaridade**: `S_ij = Î_i · T̂_j = cos θ_ij`; logits = `S_ij · exp(logit_scale)`; `exp(logit_scale)` = **100.00** no checkpoint (output nb[6]), i.e. τ = 0.01. Inicialização: `logit_scale = log(1/0.07)` (slide 4).
- **InfoNCE simétrica** (plano §2; slide 4): `L_img = -1/B Σ_i log softmax_j(S_ij/τ)[i]`, `L_txt` idem por colunas, `L_CLIP = (L_img + L_txt)/2`. Interpretação HOML: cada linha/coluna é uma classificação B-vias cujo alvo é a diagonal; pares não relacionados tendem a cos ≈ 0 (vetores quase ortogonais em alta dimensão), não −1; exige batch enorme (32.768) para ter negativos suficientes.
- **Por que permite recuperação sem supervisão**: 400M pares (imagem, legenda) da web (WIT) sem anotação humana; o objetivo contrastivo força imagem e texto de mesmo conteúdo para a mesma direção no espaço comum → qualquer frase nova vira um "classificador"/consulta por produto escalar (vocabulário aberto). Ranking = `G @ qᵀ`.
- **Limitações** (HOML; `Modelos Multimodais: CLIP.md` "pontos cegos em domínios especializados"): fraco em imagens especializadas (satélite, médico); forte em cenas do cotidiano da web. Relevante para anúncios: texto dentro da imagem, logos e marcas podem dominar o embedding [hipótese, verificar].
- **Prompt engineering / ensembling** (slide 10; nb[14]): nível 1 palavra isolada (polissemia: "crane" guindaste vs grou) → ImageNet 67.5%; nível 2 template `"a photo of a {c}."` → 68.8% (+1.3); nível 3 ensemble de 80 templates `T̄ = Normalize(Σ T_k)` → 72.5% (+5.0), "equivale a 4× mais dados rotulados, custo zero na inferência da imagem".

### 3.2 Checkpoints usados
- `openai/clip-vit-base-patch32` em todos os notebooks da Aula 6 (e no HOML). 151.3M params (output nb[6]); pesos ~605 MB; input 224×224; patch 32. Não há uso de `-patch16` nem de `ViT-L/14` no material. [externo] patch16 tem ~4× mais tokens visuais (196 vs 49) → mais lento/mais VRAM, geralmente melhor em retrieval; em inferência pura com `no_grad` ambos cabem folgadamente na T4.

### 3.3 Código reaproveitável — `Aula-6/aula_06_clip_openai.ipynb`
| Célula | Conteúdo | Adaptação para A2 |
|---|---|---|
| nb[3] | seed 42 + device + VRAM | copiar |
| nb[4] | `load_image_safely(url)` com User-Agent + fallback sintético | **não usar** no ADS-16 (imagens locais do Kaggle); se usar, o fallback silencioso contaminaria o corpus |
| nb[6] | `CLIPProcessor/CLIPModel.from_pretrained(MODEL_NAME)`, `model.eval()`, prints de `visual_projection`, `text_projection`, `logit_scale.exp()` | copiar (serve de evidência para 4.1) |
| nb[8]–[9] | `pipeline("zero-shot-image-classification", hypothesis_template="a high quality photo of a {}.")` | opcional; o softmax do pipeline é **relativo ao conjunto de rótulos** → não serve para threshold absoluto de ocorrência |
| **nb[12]** | `processor(text=..., images=..., padding=True)`; imprime shapes de `input_ids` e **`attention_mask`** (`[4, 11]` — padding dinâmico até o maior do batch); normaliza, `cosine_sim = image_norm @ text_norm.T`, checa vs `outputs.logits_per_image` (diff 3.81e-06) | base da matriz imagem × conceito (2.1a) e da evidência 4.1/4.4 |
| nb[13] | heatmap seaborn da matriz de cosseno | visualização da matriz (subamostrar imagens) |
| **nb[15]** | `get_image_features_clean(imgs)` / `get_text_features_clean(texts)` → L2-normalizados, com fallback para versões do `transformers` em que `get_*_features` retorna objeto em vez de tensor | **copiar tal qual**; para 500+ imagens, chamar em lotes (ex.: 64) e concatenar; salvar `.pt`/`.npy` no Drive (cache) |
| **nb[16]** | 8 templates OpenAI + média + **re-normalização** do vetor médio | usar para os ≥20 conceitos (2.1a) |
| nb[17] | comparação single word / template / ensemble (softmax com `logit_scale`) | opcional: justificar o uso de ensemble |
| **nb[19]–[20]** | indexação offline `gallery_embeds`; `search_gallery(query, top_k)` com `torch.topk(gallery_embeds @ q.T)` e grade de imagens com cosseno no título | **núcleo do 2.2**: top_k=5, ≥8 consultas |
| nb[22] | t-SNE (`perplexity=10, init="pca", max_iter=2000, random_state=42`) em 3D com plotly | opcional (t-SNE 2D colorido por categoria ADS); usar matplotlib 2D para o PDF |
| nb[24] | anomalia zero-shot: score do prompt "damaged" ≥ 0.50 | ilustra threshold fixo, mas sobre softmax de 2 prompts |
| nb[26]–[28] | linear probe `LogisticRegression` em embeddings congelados | fora do escopo da A2 |

Templates exatos (nb[16]): `"a photo of a {c}."`, `"a centered photo of a {c}."`, `"a close-up photo of a {c}."`, `"a high quality photo of the {c}."`, `"a photo of the clean {c}."`, `"a detailed photo of a {c}."`, `"a cropped photo of a {c}."`, `"a good photo of a {c}."`. Para anúncios convém acrescentar/ajustar ("an advertisement featuring a {c}.", "a poster showing a {c}.") — decisão nossa, justificar.

### 3.4 Escala real das similaridades (base para o threshold 2.1b)
Números **medidos** nos outputs salvos (ViT-B/32, cosseno bruto):
- Fine-tuning nb[10] (baseline zero-shot, 8 pares Pokémon): pares corretos **0.2918**, incorretos **0.2384**, margem **0.0533**.
- O slide 11 (lab simulado) mostra cos 0.78 para o par correto — valor ilustrativo, **não realista** para CLIP; não usar como referência.
Consequência: cossenos corretos ficam tipicamente na faixa ~0.25–0.35 e o "fundo" em ~0.20–0.25 → um threshold absoluto fixo (ex.: 0.5) nunca dispararia. Opções coerentes com o PLANO §Etapa 4: (i) percentil por conceito ou global da distribuição imagem×conceito (ex.: p90); (ii) margem sobre um prompt neutro (`"a photo."`/`"an advertisement."`): conceito ocorre se `s(img,c) − s(img,neutro) > δ`; (iii) z-score por conceito; (iv) validação manual de ~30–50 imagens para calibrar. O material **não** ensina nenhuma dessas; só mostra o limiar 0.50 sobre softmax de 2 prompts (nb[24]) e o lab θ∈[0.20,0.80] (slide 15).

### 3.5 Tokenização CLIP vs BERT (4.4)
O que o material mostra:
- CLIP (nb[5], slide 3): tokens **BPE**, comprimento máximo **L = 77**, representação tomada no token **[EOS]/EOT**. nb[12]: com `padding=True` o `input_ids` sai `[4, 11]` e há `attention_mask` `[4, 11]`. Fine-tuning nb[14]: `padding="max_length", max_length=77, truncation=True` → sempre 77.
- HOML cap. 16: o text encoder é **GPT-2-like (causal)**, "o output do último token é usado como representação da sequência"; o encoder causal permite cachear prefixos comuns ("This is a photo of a").
- BERT (Aula 3 nb[5], nb[7], falas l.132, l.260): **WordPiece**, vocab **30.522**, `[CLS]` (id 101) no início, `[SEP]` (id 102) entre/fim de frases; representação da sequência = `last_hidden_state[:, 0, :]` (pooling do `[CLS]`); `tokenizer(text, truncation=True, max_length=128)` + `DataCollatorWithPadding` (padding dinâmico por batch). Subwords `##` (ex.: "Cupertino" → "Cu", "##pert", "##ino"); rótulo `-100` em subwords/[CLS]/[SEP]/[PAD] no NER (nb[10]–[11]).
- Máscara de padding genérica (Aula 3 `aula_03_transformer_pytorch.ipynb` nb[6]): `src_pad_mask = (src == pad_idx)` passado como `src_key_padding_mask`; `nn.Embedding(..., padding_idx=pad_idx)`; máscara causal `torch.triu(..., diagonal=1)`.

Síntese para o texto (parte [externo], validar imprimindo tokens no notebook):
| | CLIP text encoder | BERT |
|---|---|---|
| Tokenizador | BPE byte-level, lowercase, vocab ~49.408 [externo] | WordPiece, vocab 30.522 (`bert-base-uncased`) |
| Especiais | `<|startoftext|>` (49406) … `<|endoftext|>` (49407) [externo] | `[CLS]` 101 … `[SEP]` 102, `[PAD]` 0 [externo p/ PAD] |
| Comprimento | máx. 77 (truncar consultas longas) | máx. 512; aula usa 64/128 |
| Atenção | **causal** (cada token só vê os anteriores) | **bidirecional** |
| Vetor da sequência | estado no **EOT** → `text_projection` → L2 | estado do **[CLS]** (+ pooler) |
| Padding | no HF, `pad_token` do `openai/clip-vit-base-patch32` é o próprio `<|endoftext|>` [externo]; como o vetor vem do EOT e a atenção é causal, tokens de padding *depois* do EOT não influenciam o resultado; a `attention_mask` é aplicada mesmo assim | `attention_mask` = 0 nos `[PAD]` é **essencial**: atenção bidirecional faria `[CLS]` atender aos PADs e alterar a representação |
| Segmentos | não há | `token_type_ids` (Segment Embeddings E_A/E_B) |
Código ilustrativo sugerido: tokenizar a mesma consulta com `CLIPTokenizer` e `AutoTokenizer("bert-base-uncased")`, com `padding="max_length"` e sem, e imprimir `convert_ids_to_tokens`, `input_ids`, `attention_mask`; demonstrar que o embedding CLIP de uma consulta é idêntico com `padding=True` e `padding="max_length"` (diferença ~1e-6), e que no BERT zerar a `attention_mask` dos PADs muda o `[CLS]` (experimento: mask correta vs mask toda 1).

---

## 4. Mapeamento rubrica → material

| # | Item | Material de apoio | Observação |
|---|---|---|---|
| 1.1 | Head trocado, backbone congelado | Aula-1 cnn nb[14], nb[16]; slide 17 | direto; manter print treináveis vs totais |
| 1.2 | Curvas + acc por classe e global | Aula-1 nb[10], nb[12], nb[18], nb[20] | acc por classe e matriz de confusão **não** estão no material (usar `sklearn.metrics`) |
| 1.3 | ≥3 augmentations justificadas | Aula-1 nb[4]; HOML cap. 12 (Compose + princípios) | análise por classe depende das classes reais do dataset |
| 1.4 | FE vs FT | slide 17; Aula-1 nb[21]; HOML; Aula-6 slide 12 | texto nosso |
| 1.5 | Escolha do modelo (T4, nº classes) | slide 14 (Tabela 12-3), slide 15 (VRAM) | ResNet-50 só via [externo] |
| 4.1 | Alinhamento e contrastivo | Aula-6 slides 3–4, nb[5], nb[10], nb[12]; HOML cap. 16 | evidência: `logit_scale=100`, diff manual vs nativo |
| 4.2 | Ranking ≥20 conceitos + threshold + top-5 | nb[15], nb[16], nb[12]–[13] | threshold é decisão nossa (§3.4) |
| 4.3 | ≥8 consultas analisadas | nb[19]–[20] | top_k=5; tabela de análise é nossa |
| 4.4 | CLIP vs BERT, padding/mask | Aula-6 nb[5], nb[12], FT nb[14]; Aula-3 BERT nb[5], nb[7], nb[10]; transformer nb[6]; HOML cap. 16 | vários detalhes [externo], confirmar por print |

---

## 5. Lacunas (não cobertas pelo material)
1. **Dataset A3 `pavansanagapati/images-dataset`**: classes, contagens e formato não aparecem em nenhuma aula → a análise de augmentation por classe (1.3) só pode ser escrita após a EDA/smoke test.
2. **ADS-16**: não há nenhuma menção nas aulas; slug Kaggle, estrutura (16 categorias), se há imagens de anúncio com texto sobreposto, e licença: tudo a descobrir na Etapa 0.
3. **Split estratificado, acc por classe, matriz de confusão**: não mostrados; usar scikit-learn.
4. **ResNet-50** não está na Tabela 12-3 do slide; EfficientNet-B0 está (5.3M, 0.4 GFLOPs, 77.7%). Números da ResNet-50 e o head da EfficientNet (`classifier[1]`) são [externo].
5. **Transforms de treino com augmentation + `weights.transforms()`**: a aula usa `weights.transforms()` puro; compor augmentation + normalização ImageNet só está no HOML. Atenção: EfficientNet-B0 usa interpolação bicúbica e resize 256 [externo] — reproduzir a partir de `weights.transforms()` impresso.
6. **Threshold de ocorrência** no CLIP: nenhum método calibrado é ensinado (§3.4).
7. **Detalhes de tokenização** (ids especiais do CLIP, vocab 49.408, pad = EOT no HF, `[PAD]`=0 no BERT): não estão no material; verificar imprimindo.
8. **Padding/attention mask do BERT**: o `.md` da Aula 3 promete "tokenização, padding e máscaras de atenção", mas o notebook só usa `DataCollatorWithPadding` e `truncation`, sem imprimir `attention_mask`. A máscara de padding explícita aparece apenas no seq2seq do `aula_03_transformer_pytorch.ipynb` nb[6].
9. Nenhum notebook lido persiste artefatos no Drive nem mede tempo/memória de pico por célula (exigência do nosso enunciado).

## 6. Dúvidas para o Gilmar
1. **A3 — modelo**: ResNet-50 (não está no slide, 25.6M/4.1 GFLOPs) ou EfficientNet-B0 (no slide, 5.3M/0.4 GFLOPs)? Proposta: **EfficientNet-B0** pela eficiência na T4 e por estar na tabela da aula, citando ResNet-34/50 como alternativa; ou ResNet-50 se quiser alinhar com o código da aula (troca de `fc` idêntica). OK?
2. **A3 — augmentation no treino**: o enunciado pede um único treino com feature extraction; aplicamos augmentation no treino (versão HOML) ou mantemos `weights.transforms()` puro (como na aula) e deixamos a 3.2 só como análise escrita?
3. **A2 — checkpoint**: `clip-vit-base-patch32` (o da aula, mais rápido) ou `patch16` (melhor retrieval, ainda cabe na T4)?
4. **A2 — threshold**: preferência entre percentil global, margem sobre prompt neutro, ou calibração com validação manual de ~40 imagens (mais trabalhoso, melhor justificativa)?
5. **A2 — idioma das consultas/descrições**: manter em inglês (CLIP foi treinado em inglês) e traduzir nas tabelas do relatório?
