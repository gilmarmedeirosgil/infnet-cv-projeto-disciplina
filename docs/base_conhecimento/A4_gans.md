# Base de conhecimento A4: GANs (Aula 7 e Aula 8) para A4.1 e A4.2

Fonte: `../../../Aula-7/` e `../../../Aula-8/` (caminhos relativos a este arquivo). Índices de célula são **0-based** (ordem do `nb['cells']`).
Material lido: 3 notebooks da Aula 7, `plano_aula (1).md`, `Arquitetura GAN.md`, `Aula-8/síntese de imagens.md`, slides `aula_07_apresentacao.pdf` (PDF só com imagens, lido visualmente: slides 4, 6–15), cap. 18 do *Hands-On ML (Scikit-Learn & PyTorch)* (seção GANs) e cap. 9 do *Deep Learning Crash Course* (Projects 9A/9B/9C).
**Não lido:** o vídeo da aula (`Visao_Computacional_...23-09-2026_20-00.mp4`, 188 MB).
Os PDFs dos livros na Aula-8 são **idênticos** (conferido com `cmp`) aos da Aula-7.

---

## 1. Padrões transversais do professor (Prof. Allan Spadini)

- **Roteiro pedagógico obrigatório**: *Situação-Problema do Mundo Real → Solução de Engenharia → Teoria Rigorosa* (cabeçalho de todos os notebooks e §2 do plano de aula). Isso casa com a convenção 0.2 do PLANO_PROJETO (Problema → Decisão → Código → Resultado → Análise).
- **Célula 0**: badge "Open in Colab", título, "Objetivos Deste Laboratório" numerados. Obs.: **nenhum notebook da aula declara RAM/VRAM e tempo**; o enunciado do projeto exige isso, então precisamos acrescentar.
- **Setup**: `SEED = 42` em `random`, `numpy`, `torch`, `cuda.manual_seed_all`; imprime o nome da GPU e a VRAM (`Tesla T4`, 14.56 GB). Não usa `cudnn.deterministic`.
- **Dados**: `kagglehub.dataset_download(...)` dentro de `try/except`, com **fallback sintético** gerado por numpy, para o notebook nunca quebrar.
- **Sanity check de shapes com `assert`** logo após instanciar G/D (DCGAN célula 8, CycleGAN célula 8).
- **Inicialização DCGAN**: `weights_init` com Conv ~ N(0, 0.02) e BatchNorm peso ~ N(1, 0.02), bias 0.
- **Normalização [-1, 1]** (`Normalize(0.5, 0.5)`) para casar com a `Tanh` do gerador, com a justificativa textual na DCGAN célula 3.
- **Hiperparâmetros canônicos**: Adam `lr=2e-4`, `betas=(0.5, 0.999)`; one-sided label smoothing `REAL=0.9`, `FAKE=0.0`.
- **Governança clínica**: o **Recall (sensibilidade) da classe minoritária** é a "métrica mandatória"; a avaliação é sempre num **teste 100% real** e o "paradoxo da acurácia" aparece de forma explícita. Meta citada no slide 14: **recall ≥ 90%**.
- **"5 problemas técnicos" de visão biomédica** (plano de aula §3 item 7 e §6): desbalanceamento, atalhos de aquisição (*shortcut learning*), vazamento por paciente/espécime, alucinações (do gerador) e acurácia enganosa. **É o vocabulário que o professor espera na rubrica 5.1.**
- **Tudo dimensionado para a T4**: 64 px, poucas épocas (12 na cGAN, 20 na DCGAN, 15 na CycleGAN). O plano de aula cita AMP (`torch.cuda.amp.autocast()`), mas **nenhum notebook usa AMP de fato** (grep sem ocorrências).
- **Estilo**: prints com emojis, markdown com LaTeX, `seaborn.heatmap` para as matrizes de confusão e gráfico de barras comparativo.

---

## 2. cGAN da Aula 7: `aula_07_gans_generative_adversarial_networks.ipynb` (21 células)

### 2.1 Mapa das células
| Célula | Conteúdo |
|---|---|
| 0 | Cabeçalho, 7 objetivos (auditoria de desbalanceamento, baseline, cGAN, treino na T4, augmentation, comparação de recall, CycleGAN) |
| 2–3 | `pip install -q kagglehub torchvision scikit-learn matplotlib seaborn tqdm`; imports e seed |
| 5 | `kagglehub.dataset_download("paultimothymooney/blood-cells")` + fallback sintético de "células" 64×64 |
| 7 | Transforms, `ImageFolder`, **simulação da escassez** e subsets |
| 8 | Visualização (`cmap="bone"`) |
| 10 | `build_classifier`, `train_classifier`, `evaluate_classifier` |
| 11 | Baseline (sem GAN) |
| 12–13 | Markdown e código de `ConditionalGenerator`, `ConditionalDiscriminator`, `weights_init` |
| 14–15 | Loop adversarial |
| 16 | Grade 2×4 de sintéticos da classe rara (`fixed_noise`, 16 vetores) |
| 18 | `SyntheticDataset` + geração de 80 amostras + `ConcatDataset` |
| 19 | Classificador aumentado |
| 20 | Duas matrizes de confusão lado a lado + barras de Recall/F1/Precisão |

A célula 0 lista a CycleGAN como objetivo 7, mas **não há CycleGAN neste notebook**; ela fica no caderno 3.

### 2.2 Dados e desbalanceamento (célula 7)
- `IMG_SIZE = 64`, `BATCH_SIZE = 32`. Treino: `Resize → Grayscale(1) → RandomHorizontalFlip(0.5) → ToTensor → Normalize([0.5],[0.5])`; teste sem o flip.
- O dataset real tem 4 classes (`EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL`). O notebook usa **só as classes 0 e 1** e trata a tarefa como **binária** ("Comum" = EOSINOPHIL, "Rara" = LYMPHOCYTE).
- Escassez simulada: 100% da classe 0 (2.497) e **20% da classe 1** (496) no treino, com os primeiros índices e sem sorteio. Teste: 1.243 imagens, **aproximadamente balanceado (623/620)**, portanto **sem a prevalência real**.
- **Não há conjunto de validação.**

### 2.3 Arquitetura (célula 13), com `nz=100`, `ngf=ndf=64`, 64×64×1
- **Gerador `ConditionalGenerator(nz=100, num_classes=2, embed_dim=16, ngf=64)`**: `nn.Embedding(2,16)` → `unsqueeze` para `[B,16,1,1]` → `cat` com z `[B,100,1,1]` → `[B,116,1,1]`. Em seguida 5 `ConvTranspose2d(k=4)`: 116→512 (s1,p0; 4×4) → 256 (8×8) → 128 (16×16) → 64 (32×32) → 1 (64×64), com BN+ReLU no meio e `Tanh` na saída; `bias=False`. **3.705.760 parâmetros.**
- **Discriminador `ConditionalDiscriminator(num_classes=2, img_size=64, ndf=64)`**: `nn.Embedding(2, 64*64)` → `view(-1,1,64,64)` (mapa espacial do rótulo) → `cat` com a imagem (2 canais). Depois `Conv2d(k4,s2,p1)` 2→64 (sem BN) → 128 → 256 → 512 (BN + LeakyReLU 0.2) → `Conv2d(512,1,k4,s1,p0)` → **`Sigmoid`**. **2.772.736 parâmetros.**
- No livro (DLCC Project 9A) a variante do D é `Embedding(10,100) → Linear → LeakyReLU → 4096 → reshape 64×64` como canal extra. O notebook pula o `Linear` e usa o embedding direto com dimensão H·W.

### 2.4 Loop de treino (célula 15)
- `criterion_bce = nn.BCELoss()`; Adam `lr_gan=2e-4`, `beta1=0.5` para **G e D (mesmo LR, sem TTUR)**; `EPOCHS_GAN = 12`.
- **A GAN é treinada no `imbalanced_train_set` inteiro**, condicionando nas 2 classes (y=0 e y=1), e não só na classe rara.
- Passo D: alvo real **0.9** (label smoothing), falso 0.0; `netD(fake_imgs.detach(), y)`; `loss_d = real + fake`; um único `backward`.
- Passo G: **reutiliza o mesmo `fake_imgs`** (sem `detach`) e `BCE(D(G(z,y),y), 1.0)`, ou seja, a heurística não saturante.
- Proporção 1:1 de passos D:G. Não guarda o histórico: **só imprime** a média de loss a cada 3 épocas. Não registra D(x)/D(G(z)), não salva grades por época e não calcula métrica de qualidade.
- Saída registrada: época 3 `D=0.947 G=3.673`; época 6 `1.210/1.569`; época 9 `1.212/1.586`; época 12 `1.274/1.467`.

### 2.5 Aumento e comparação de recall (células 10, 11, 18–20)
- Classificador: `resnet18(weights=ResNet18_Weights.DEFAULT)` com **`conv1` trocada para 1 canal e inicializada pela média dos pesos RGB** (`original_conv.weight.mean(dim=1, keepdim=True)`); `fc → Linear(512, 2)`. Entrada de **64×64** (a mesma resolução da GAN).
- Treino: `CrossEntropyLoss` sem pesos, `AdamW(lr=1e-4, wd=1e-2)`, **6 épocas**, sem scheduler, sem early stopping e **sem validação**.
- `SyntheticDataset(tensors, labels)`: guarda os tensores já em [-1, 1] (mesma normalização do real). `NUM_SYNTHETIC = 80` (sem justificativa), gerados com `netG.eval()` e `y=1`, depois `ConcatDataset([imbalanced_train_set, synthetic_dataset])`.
- Métricas: `recall_score / precision_score / f1_score(pos_label=1)` + `confusion_matrix` pelo sklearn.
- **Resultado (1 seed):**

| | Recall rara | Precisão rara | F1 | CM `[[TN,FP],[FN,TP]]` |
|---|---|---|---|---|
| Baseline | 75,81% | 97,51% | 0,853 | `[[611,12],[150,470]]` |
| + 80 sintéticos | 88,71% | 98,39% | 0,933 | `[[614,9],[70,550]]` |

### 2.6 Fragilidades do notebook (evitar ou apontar na análise)
1. **Uma única seed**, sem IC: a queda de FN de 150 para 70 com apenas +16% de amostras da classe rara pode ser variância do treino. O PLANO já prevê ≥3 seeds com média ± desvio.
2. O teste não tem a prevalência realista e não existe validação para seleção de modelo.
3. **Não há evidência de qualidade nem de diversidade da GAN** (sem curvas, FID ou checagem de memorização). A rubrica 5.3 pede exatamente isso.
4. `Sigmoid + BCELoss` é numericamente frágil e **incompatível com autocast/AMP**. Preferir logits + `BCEWithLogitsLoss`.
5. A geração em `netG.eval()` faz o BatchNorm usar estatísticas acumuladas; vale conferir visualmente se muda o resultado em relação ao modo `train()`.
6. Os sintéticos entram como tensores fixos, **sem a augmentation** aplicada às imagens reais.

### 2.7 O que adaptar para 3 classes de raio-X em tons de cinza
- `num_classes=3` nos dois `Embedding`; **manter os rótulos multiclasse** (sem binarização) e usar o mapeamento do `ImageFolder`/DataFrame (fixar `COVID`, `Normal`, `Pneumonia` = índices explícitos).
- **64 px** reaproveita a arquitetura sem mudanças. Para **128 px**, somar um bloco `ConvTranspose2d(k4,s2,p1)` no G (1→4→8→16→32→64→128) e um `Conv2d(k4,s2,p1)` no D, com `Embedding(3, 128*128)`. Na T4, 64 px é o caminho seguro; 128 px é viável com batch 32–64.
- **Treinar a GAN só no split de treino** (nunca em val ou teste). A classe COVID terá ~84–100 imagens de treino (de 120), um regime de pouquíssimos dados em que o D decora rápido. Isso justifica DiffAugment/ADA (fora do material; ver §9) e a checagem de memorização.
- Normalização: se o classificador usar a ResNet-18 em 224 px com normalização ImageNet, **os sintéticos precisam passar pelo mesmo pipeline**. O ideal é salvá-los como PNG uint8 e carregá-los com o mesmo `transform` do real. Caso contrário, o classificador pode aprender o atalho "imagem suavizada por upsampling = COVID" (ver §10, dúvida 2).
- Registrar `history` no estilo da DCGAN (célula 11): `loss_D`, `loss_G`, `D_x`, `D_G_z1`, `D_G_z2`, **por classe**, e uma grade de `fixed_noise × 3 classes` por época.

---

## 3. DCGAN: `aula_07_dcgan_cifar10_treinamento.ipynb` (17 células)

| Célula | Conteúdo |
|---|---|
| 3 | Por que normalizar para [-1,1] (Tanh) |
| 4 | CIFAR-10 32×32, `FILTER_CLASS = 6` (frog, 5.000 imagens), `BATCH_SIZE=128`, `num_workers=2`, `pin_memory` |
| 6 | Diretrizes de Radford: conv com stride no lugar de pooling, BN (exceto na 1ª camada do D e na saída do G), ReLU/Tanh no G, LeakyReLU 0.2 no D, init N(0, 0.02) |
| 7 | `Generator` (100→256→128→64→3; 4 ConvT) e `Discriminator` (3→64→128→256→1, Sigmoid); G 1,07 M e D 0,66 M de parâmetros |
| 8 | Sanity check com `assert` de shape |
| 9 | **Derivação da heurística não saturante**: para D=σ(a), `∂log(1−D)/∂a = −D` (≈0 quando D≈0), enquanto `∂(−log D)/∂a = −(1−D)` (≈1). Explicação do `.detach()`: sem ele, `loss_D.backward()` retropropaga por G (VRAM em dobro e atualizações espúrias) |
| 10 | `BCELoss`, `FIXED_NOISE` com 64 vetores, Adam 2e-4 (0.5, 0.999), `REAL_LABEL=0.9`, `FAKE_LABEL=0.0` |
| 11 | Loop com **telemetria** `history = {loss_D, loss_G, D_x, D_G_z1, D_G_z2}` e `img_list` (grade do ruído fixo por época); D com `backward` separado para real e fake; G com alvo 1.0 |
| 12 | **Critério de saúde do professor**: perdas oscilando em torno de um equilíbrio; `D(x) > 0.6` e `D(G(z))` subindo de ~0 para **0.3–0.5** |
| 13 | Dois painéis: curvas de loss D/G e D(x) vs D(G(z)) com uma linha em 0.5 |
| 14 | Comparação visual época 1 vs. época final |
| 15–16 | **Interpolação latente** `z(α) = (1−α)z_A + α z_B` em 10 passos, usada como "prova" de não memorização. O comentário do código diz "esférica", mas a interpolação é **linear** |

**Leitura das saídas (célula 11)**: na maior parte das épocas `D(G(z)) ≈ 0.00–0.05` e `Loss_G ≈ 3–7` (valores do último batch mostrados no tqdm), e na época 5 `D(x)` cai para 0.27. Pelo próprio critério da célula 12, isso é **D dominando o jogo**. O exemplo serve para ilustrar o diagnóstico de "D forte demais" na Run A.

**Reaproveitar na A4.1**: o `history` e o gráfico da célula 13, a grade por época com ruído fixo (célula 11/14) e a interpolação (célula 16) como evidência qualitativa de variedade contínua, que não prova ausência de memorização (ver §5).

---

## 4. Instabilidades e mitigações: o que o material diz e como evidenciar

### 4.1 Diagnóstico (fontes)
- **Slide 6 ("Amostragem Cega e Colapso de Modos")**: colapso **total** (G põe 100% da massa num modo), **parcial** e **"mode hopping"** (época 10: G colapsa no modo A; época 12: D passa a rejeitar A; época 15: G salta para B; o ciclo se repete). Causa teórica: comportamento *mode-seeking* da **KL reversa KL(p_g‖p_data)**. Sintoma visual: **"Batch Clone Syndrome"**, com amostras quase idênticas na grade. Detecção quantitativa: **o recall generativo despenca enquanto a precisão se mantém** (exemplo do slide: Precision 99%, Recall 12,5% = 1/8 modos).
- **HOML cap. 18, "The Difficulties of Training GANs"**: equilíbrio de Nash único (G perfeito e D=50%) sem garantia de ser atingido; **mode collapse** (exemplo dos "sapatos": G se especializa numa classe, D esquece as outras e os dois ciclam); **oscilação ou divergência súbita** dos parâmetros; forte sensibilidade a hiperparâmetros.
- **DLCC cap. 9 (9-1 e 9A)**: se a **loss do D fica baixa demais, o G não oferece desafio e o treino tende ao mode collapse**. Treinar mais épocas que o necessário também aumenta o risco (o livro para a cGAN MNIST em 10 épocas). Na cGAN, um sinal típico é **G gerar imagens parecidas para rótulos diferentes** (o G deixa de usar a condição).
- **Slide 4**: `D*(x) = p_data/(p_data+p_g)` e `V(D*,G) = −log 4 + 2·JSD`, que dão a base teórica do gradiente fraco quando os suportes são disjuntos.

### 4.2 Mitigações citadas
| Técnica | Onde aparece | Detalhe dado |
|---|---|---|
| **Heurística não saturante (−log D)** | Slide 4, DCGAN c9, cGAN c15, quiz do slide 15 | Obrigatória, já no baseline |
| **One-sided label smoothing** | Slide 9 (Salimans 2016), DCGAN c10, cGAN c15 | `y_real=0.9`, `y_fake=0.0`, **nunca suavizar o fake** (o D* passaria a reforçar amostras ruins) |
| **TTUR** | Slide 9 (Heusel 2017) | `η_D > η_G`, ex. **η_D = 4e-4, η_G = 1e-4** |
| **Instance noise** | Slide 9 (Sønderby 2016) | `x̃ = x + ε`, `ε~N(0,σ²)`, σ_t → 0 (annealing) |
| **Spectral Normalization** | Slide 9 (Miyato 2018) | `W/σ(W)` via power iteration, overhead < 2%, **`torch.nn.utils.parametrizations.spectral_norm(conv)`**, limita o D a 1-Lipschitz |
| **Experience replay / replay buffer** | HOML cap. 18; CycleGAN c10 (`ReplayBuffer(max_size=50)`) | D treina com fakes antigos, o que reduz o overfit ao G atual; exercício 12 do HOML pede "adicione experience replay" |
| **Minibatch discrimination** | HOML cap. 18 | D recebe a similaridade intra-batch e rejeita batches sem diversidade |
| **Ajustar LRs de G/D** | DLCC 9-1 e 9B | 9B usa **lr_G = 2e-4 > lr_D = 5e-5** (o sentido oposto do TTUR do slide) |
| **Ruído nos rótulos** | DLCC cap. 9 | "adding noise to the labels" |
| **LSGAN (MSE)** | CycleGAN c9–10, DLCC 9B | Evita a saturação da BCE |
| **PatchGAN** | CycleGAN c7, DLCC 9B | Mais relevante para image-to-image; pouco útil para cGAN z→imagem |
| **Wasserstein / 1-Lipschitz** | `Arquitetura GAN.md` ("distância de Wasserstein fornece gradientes mais suaves"), slide 9 (rótulo "1-Lipschitz") | **Sem fórmula de WGAN nem de GP, e nenhum código** |

### 4.3 Como evidenciar na Run A vs. Run B (proposta, com as mesmas métricas nas duas)
- **Curvas por iteração/época**: `loss_D`, `loss_G`, `D(x)`, `D(G(z))`. Divergência: `loss_D → 0`, `loss_G` crescendo ou explodindo, `D(G(z))` preso perto de 0. Colapso: `loss_G` cai enquanto a diversidade despenca.
- **Grade com ruído fixo por época e por classe** (padrão da DCGAN c11/c14): mostrar a "Batch Clone Syndrome" e o *mode hopping* entre épocas.
- **Diversidade intra-classe**: distância média par a par (LPIPS ou 1−MS-SSIM, ou L2 em features de uma rede pré-treinada) entre N sintéticos de COVID, comparada com o mesmo valor em N reais de COVID. Queda para uma fração do valor real indica colapso.
- **Distância de distribuição**: FID ou KID por classe (via `torchmetrics`). Com poucas centenas de imagens, **KID** é menos enviesado. O slide 13 alerta que o FID com < 10k amostras tem viés positivo, então **reportar só de forma comparativa (A vs. B, mesmo N)**.
- **Condicionamento**: um classificador real (o baseline corrigido) aplicado aos sintéticos de cada `y` deve acertar a classe pedida. Se não acertar, o G ignora a condição (o sinal citado no DLCC).
- **Memorização**: vizinho mais próximo de cada sintético no treino (L2/SSIM/LPIPS), comparado com a distância real→real.
- Sugestão para a **Run A**: sem label smoothing, LR iguais e altos, BN no D e mais épocas do que as 12 do notebook. A Run B aplica SN no D (normalmente **sem BN no D** quando se usa SN), label smoothing 0.9, TTUR 4e-4/1e-4 e, opcionalmente, DiffAugment.

---

## 5. Métricas de qualidade e diversidade citadas

- **Inception Score** (slide 13; Salimans 2016; também em `Aula-8/síntese de imagens.md`): `IS = exp(E_x[KL(p(y|x) ‖ p(y))])`, que mede nitidez (p(y|x) de baixa entropia) e diversidade (p(y) uniforme). **Falha crítica apontada no slide**: não compara com os dados reais, e memorizar 1 imagem por classe já dá um IS alto. Para raio-X, as classes ImageNet do Inception não têm significado, então **não usar**.
- **FID** (slide 13; Heusel 2017; Aula-8 .md): `‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2(Σ_rΣ_g)^{1/2})` sobre o pool3 do Inception-v3 (2048 dimensões). Protocolo de **50k imagens**, viés com < 10k e hipótese gaussiana rígida. O slide diz que, com mode collapse, "Σ_g colapsa e o traço explode".
- **Precision & Recall generativos** (slide 13, "solução moderna"; slide 6): separam a fidelidade (precision) da cobertura dos modos (recall). O slide não cita o autor (Sajjadi 2018 / Kynkäänniemi 2019 são referências externas).
- **Avaliação downstream** (slides 13–14; cGAN c20): o critério que o professor considera decisivo é o **ganho de recall da classe minoritária no teste 100% real**.
- **L1 de ciclo** (CycleGAN c15: forward 0,0727, backward 0,0814) e **LPIPS** como perda perceptual (DLCC 9B) valem só para tradução de imagens. O LPIPS, porém, serve bem como distância para medir diversidade (§4.3).
- **KID, MS-SSIM e LPIPS-diversidade não aparecem** nas aulas (ver §9).

---

## 6. Aula 8 e CycleGAN (síntese e tradução de domínios)

- `Aula-8/síntese de imagens.md` (único material próprio da Aula 8): síntese condicional, tradução entre domínios, **CycleGAN com consistência cíclica** para preservar a estrutura, comparação GAN vs. ViT vs. CNN, **IS** e **FID**. Não há notebook nem slides da Aula 8; o conteúdo prático é o da Aula 7.
- **`aula_07_cyclegan_holo2bright_traducao_dominios.ipynb`** (17 células): dados **sintéticos** (200 X + 200 Y gerados na célula 4, 64×64 RGB), `UnpairedDataset` (c5), `ResNetGenerator` com 6 blocos residuais e **InstanceNorm** e `PatchGANDiscriminator` com saída 7×7 em 64 px (c7). Perdas (c9–10): **LSGAN (`MSELoss`)**, ciclo L1 com `λ_cyc=10`, identidade L1 com `λ_idt=5`, `ReplayBuffer(50)`, Adam 2e-4 (0.5, 0.999). Treino de 15 épocas (c11); curvas (c13); avaliação dos dois ciclos (c15).
- O plano de aula cita PatchGAN **70×70** e AMP, mas o notebook usa uma PatchGAN com saída 7×7 em entradas de 64 px e **não usa AMP**.
- **Relevância para a A4.1**: baixa, porque não há pares ou domínios a traduzir. Serve de fonte para o **ReplayBuffer** (mitigação) e para a LSGAN como alternativa à BCE.
- **Relevância para a A4.2**: a CycleGAN como **adaptação de domínio no nível de pixel** (ex.: dia→noite, seco→chuva) aparece como ideia de mitigação de domain shift (§7).
- **DLCC Project 9B (livro)**: virtual staining é uma cGAN **condicionada por imagem** (pareada, bright-field→fluorescência), com **U-Net** + **PatchGAN**, perdas MSE (adversarial) + L1 + **LPIPS (VGG16)**, `batch=2` e LR assimétrico. Os **slides apresentam o 9B como G(z, y) com embedding de marcador** (DAPI/NeuN/GFP), o que não bate com o livro. Para a A4.1 vale o modelo do notebook (rótulo de classe → imagem).

---

## 7. Material aproveitável para a A4.2 (transfer learning ImageNet → fluxo de tráfego)

As aulas 7 e 8 **não tratam tráfego nem transfer learning diretamente**. O que dá para aproveitar:
- **Adaptação ImageNet → outro domínio** (cGAN c10): pesos ImageNet reaproveitados em imagens de canal único, com a média da `conv1` RGB. Ilustra que as features ImageNet vêm de **fotos RGB centradas em objetos**, e que usá-las fora desse domínio é uma aposta que precisa ser validada (argumento de *domain/task shift*).
- **Os "5 problemas técnicos" do professor** (plano de aula §6), transpostos para tráfego:
  - vazamento por paciente → **vazamento por câmera/sequência** (frames quase idênticos no treino e na validação); proposta: split *leave-camera-out*;
  - *shortcut learning* / atalhos de aquisição → o modelo aprende **a câmera, o horário ou o clima** em vez do fluxo;
  - acurácia enganosa → os 78% escondem o recall baixo de "congestionado"; métrica por classe e custo assimétrico;
  - desbalanceamento → poucas amostras de congestionamento e chuva/noite;
  - alucinações → risco ao usar dados sintéticos (CycleGAN dia→noite) sem validação.
- **Tradução de domínio (CycleGAN, Aula 7 caderno 3 / Aula 8 .md)** como mitigação da cobertura de condições (gerar noite/chuva a partir de dia), com a ressalva de alucinação e de validar sempre em dados reais das condições-alvo.
- **Avaliação estratificada num teste 100% real** (princípio central da aula): na A4.2, avaliar por condição (dia/noite/chuva) e por câmera nunca vista.
- Material conceitual de TL (fine-tuning vs. feature extraction, *catastrophic forgetting*) está nas aulas 1 e 6 (ver `A3_A2_cnn_clip.md`), não aqui.

---

## 8. Mapeamento rubrica → material

| Item | Exigência | Material de apoio (arquivo : célula/slide) |
|---|---|---|
| **5.1** | ≥5 problemas do projeto de raio-X com impacto clínico | Plano de aula §3.7 e §6 ("5 problemas": desbalanceamento, shortcut, vazamento por paciente, alucinações, acurácia enganosa); slide 14 (paradoxo da acurácia, FN, meta ≥90%); cGAN c9 (markdown "Paradoxo da Acurácia"); PLANO Etapa 3a (lista de 8 problemas) |
| **5.2** | cGAN com loop adversarial correto | cGAN c13 (G/D com embedding), c15 (loop com `.detach()`, NS, label smoothing); DCGAN c9 (justificativa do NS e do `.detach()`), c11 (loop instrumentado); slide 4 (derivação); slide 7 (objetivo minimax condicional); DLCC 9A |
| **5.3** | Instabilidade diagnosticada + mitigação com evidência | Slide 6 (tipos de colapso, detecção); slide 9 (TTUR, label smoothing, instance noise, SN); HOML cap. 18 (mode collapse, divergência, replay, minibatch discrimination); DLCC 9-1/9A (leitura de loss, LRs); DCGAN c12–13 (critério e gráficos); slide 13 (FID/PR) |
| **5.4** | Recall COVID com vs. sem sintéticos | cGAN c10, c11, c18–20 (protocolo baseline vs. aumentado, teste 100% real, CM + barras); slide 14 |
| **5.5** | ≥4 problemas do tráfego + riscos do TL ImageNet → fluxo | §7 acima (analogias); cGAN c10 (adaptação ImageNet); CycleGAN (domain shift); a maior parte **fora das aulas 7/8** |
| **5.6** | Plano integrado: modelo, métrica, sintéticos, critério clínico | Slide 13 ("o realismo estatístico não basta"; validação pelo ganho de recall); slide 14 (meta ≥ 90%); plano de aula §6 ("sem contaminação do teste 100% real") |

---

## 9. Lacunas (o que o material **não** cobre e exigirá fonte externa ou decisão)

1. **WGAN / WGAN-GP**: só aparecem como menção textual (`Arquitetura GAN.md`, "1-Lipschitz" no slide 9). Não há fórmula, código nem hiperparâmetros (n_critic, λ_GP). Referências externas: Arjovsky et al. 2017, Gulrajani et al. 2017.
2. **DiffAugment / ADA** (augmentation diferenciável para GANs com poucos dados): ausentes. São relevantes porque o COVID terá ~100 imagens de treino. Referências externas: Zhao et al. 2020, Karras et al. 2020.
3. **KID, MS-SSIM, LPIPS como métrica de diversidade, vizinho mais próximo**: ausentes. O material só cita IS, FID e P&R. Implementação externa: `torchmetrics` (FID/KID/LPIPS/MS-SSIM).
4. **Minibatch discrimination**: só descrita no HOML, sem código.
5. **Múltiplas seeds, IC e teste estatístico** do ganho de recall: o notebook usa uma única seed.
6. **Protocolo de split e prevalência**: o notebook não tem validação e usa um teste balanceado artificialmente. Para o projeto, seguir o PLANO (test set estratificado com a proporção real).
7. **Tradução para 3 classes**: todo o material da aula é binário ("comum vs. rara").
8. **AMP**: citado no plano de aula, não implementado em nenhum notebook.
9. **Conteúdo do vídeo da aula** (não transcrito): pode ter comentários do professor sobre o que ele espera no projeto.
10. **Transfer learning para tráfego / vídeo / temporalidade**: inexistente nas aulas 7/8.

---

## 10. Dúvidas para o Gilmar

1. **Resolução da cGAN: 64 px (idêntico ao notebook, mais estável e barato) ou 128 px?** Recomendação: 64 px na Run A/B; 128 px só se sobrar GPU.
2. **Em que resolução o classificador roda?** Se a ResNet-18 do pipeline corrigido rodar em 224 px e os sintéticos forem 64 px com upsampling, existe risco de atalho (o classificador aprende "borrado = COVID"). Opções: (a) classificador também em 64/128 px, como no notebook do professor; (b) 224 px com um teste de controle ("real vs. sintético" separável?). Recomendação: (a), a mesma resolução da GAN.
3. **Flip horizontal no raio-X**: o notebook do professor usa `RandomHorizontalFlip` (em células). Em radiografia de tórax isso inverte a lateralidade (coração à direita). Manter (é comum na literatura e barato) ou remover por plausibilidade clínica? O PLANO só proíbe o flip vertical.
4. **Treinar a cGAN nas 3 classes (como o notebook, que condiciona em todas) ou só usar a condição COVID na geração?** Recomendação: treinar nas 3 classes (mais dados para o G aprender a anatomia comum) e gerar só COVID; e, opcionalmente, Pneumonia.
5. **Quantidade de sintéticos N**: igualar a Pneumonia (~+120), igualar a Normal (~+720) ou fazer um pequeno *sweep* (0, 1×, 3× o COVID real)? O professor usou 80 sem justificativa. O slide 14 sugere a ideia de um "slider" de 0 a 3.000.
6. **Métrica GAN principal da 5.3**: aceitar **KID + diversidade LPIPS intra-classe** (mais adequadas a N pequeno), com o FID reportado de forma apenas comparativa?
7. **Mitigação da Run B**: pacote "SN no D (sem BN) + label smoothing 0.9 + TTUR 4e-4/1e-4" (tudo do slide 9) e DiffAugment como extra opcional (fora do material)? Ou restringir às técnicas citadas em aula?
8. **Troca de `Sigmoid+BCELoss` por logits + `BCEWithLogitsLoss`** (necessária se usarmos AMP e mais estável): ok desviar do código do professor, justificando no texto?
