# %% [markdown]
# # A1 — Vision Transformers para inspeção de defeitos em aço (NEU Surface Defects)
# **Visão Computacional com CNNs e Transformers** · Faculdade Infnet — Pós-Graduação · Gilmar Oliveira de Medeiros
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gilmarmedeirosgil/infnet-cv-projeto-disciplina/blob/main/notebooks/A1_vision_transformers.ipynb)
#
# **Objetivo.** Classificar os 6 tipos de defeito superficial de chapas de aço laminado a quente do NEU Surface Defect Database com três abordagens, **no mesmo split e no mesmo loop de treino**: (1) um **ViT implementado do zero** (atenção, multi-head, bloco encoder, patch embedding, CLS e positional embedding próprios, com testes); (2) um **ViT-B/16 pré-treinado no ImageNet-21k** com o head trocado (fine-tuning); (3) uma **CNN ResNet-18** pré-treinada como baseline. Depois, comparar desempenho e custo, visualizar a atenção (1 head e *attention rollout*) e discutir a escolha de arquitetura.
#
# **Requisitos de execução (Colab, runtime T4)** — estimativas, a atualizar após o "Executar tudo":
#
# | Recurso | Estimativa | Observação |
# |---|---|---|
# | RAM | ~4 GB | 1.800 imagens 200×200 em cinza cabem inteiras na memória (~72 MB em uint8) |
# | VRAM (pico) | ~5–7 GB | fine-tuning do ViT-B/16 (86 M parâmetros), batch 32, AMP fp16; o ViT do zero e a ResNet-18 ficam < 2 GB |
# | Disco | ~1 GB | dataset (28 MB) + pesos do ViT-B/16 (~350 MB) + checkpoints no Drive (~400 MB) |
# | Tempo total | ~25–40 min | download ~2 min; ViT do zero 150 épocas ~8–15 min; ViT-B/16 12 épocas ~5–8 min; ResNet-18 ~2 min; análise de atenção ~2 min |
#
# **Retomada.** Cada modelo salva o checkpoint final no Drive. Se o notebook for reexecutado (por exemplo, após uma desconexão do Colab), os modelos já treinados são carregados e o treino é pulado; `FORCE_RETRAIN = True` força o retreino. O ViT do zero salva também um checkpoint intermediário a cada 25 épocas.
#
# **Mapa da rubrica:** 2.1 → seção 4 · 2.2 → seção 10 · 2.3 → seção 4 · 2.4 → seção 5 · 2.5 → seção 11 · 3.1 → seção 4 · 3.2 → seções 6 e 10 · 3.3 → seções 7 e 9 · 3.4 → seção 11 · 3.5 → seções 9 e 11 · 3.6 → seções 9 e 11.
#
# Estrutura: Problema → Decisão técnica → Código → Resultado → Análise. Figuras usadas no relatório são salvas em PNG em `MyDrive/infnet_cv_projeto/outputs/A1/`.

# %% [markdown]
# ## 1. Setup
# Seed 42 em `random`, `numpy` e `torch`, checagem da GPU, configuração central (hiperparâmetros e flags), montagem do Drive e leitura dos Secrets do Kaggle.

# %%
!pip install -q kagglehub

# %%
import os, re, gc, math, json, time, random, hashlib, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
import transformers
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_recall_fscore_support

NOTEBOOK_T0 = time.time()
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__} | torchvision {torchvision.__version__} | transformers {transformers.__version__} | device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("ATENÇÃO: sem GPU. No Colab: Ambiente de execução > Alterar tipo > T4 GPU.")

C_BLUE, C_CYAN, C_RED, C_GREEN, C_ORANGE = "#0A345D", "#1BB5D8", "#DC2626", "#15803D", "#EA580C"
plt.rcParams.update({"axes.grid": True, "grid.linestyle": ":", "figure.dpi": 100})
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")

# %% [markdown]
# **Configuração.** Todos os hiperparâmetros ficam aqui e são gravados no JSON final.
# - **ViT do zero em 224 px, patch 16 → 14×14 = 196 tokens + CLS**, a mesma grade do ViT-B/16 pré-treinado, para que os mapas de atenção e a distância média de atenção dos dois modelos sejam comparáveis patch a patch. Tamanho "ViT-Tiny raso": dim 192, 6 blocos, 3 heads (d_k = 64), MLP 4× (~2,7 M parâmetros). Com ~1.260 imagens de treino, um modelo maior só aumentaria o overfitting. Se na T4 a época passar de ~15 s, a alternativa registrada é `img_size=128` (8×8 = 64 tokens, ~5× menos custo de atenção), ao preço de patches que cobrem uma área 3× maior da chapa.
# - **Canal único no ViT do zero** (`in_chans=1`): as imagens são em cinza e o modelo não herda pesos RGB, então replicar o canal só triplicaria os pesos do patch embedding. Os modelos pré-treinados recebem o cinza replicado em 3 canais (padrão das aulas, `convert("RGB")`).
# - **Otimização** (igual para os três modelos, mudando só LR, épocas e batch): AdamW, *warmup* linear + cosseno por passo, *label smoothing* 0,1, *gradient clipping* 1,0, AMP fp16 na T4, sem *weight decay* em bias, LayerNorm, CLS e positional embedding. O ViT do zero usa LR 5e-4 e 150 épocas (sem viés indutivo, precisa de muitas passadas); os pré-treinados usam LR baixa no backbone e 10× no head novo (que começa aleatório) e 12 épocas.
# - **Seleção do modelo**: a melhor época pelo **macro-F1 de validação** (desempate pela loss de validação), sem early stopping, para que o cosseno complete o ciclo. O teste é avaliado uma única vez.

# %%
NUM_CLASSES = 6
FORCE_RETRAIN = False     # True: ignora checkpoints do Drive e treina de novo
RUN_EXTRAS = False        # True: treina também o DeiT-small (extra, ~5 min na T4)
USE_AMP = torch.cuda.is_available()

SCRATCH_CFG = dict(img_size=224, patch_size=16, in_chans=1, embed_dim=192, depth=6, n_heads=3,
                   mlp_ratio=4.0, dropout=0.1, attn_dropout=0.0)
PRETRAINED_ID = "google/vit-base-patch16-224-in21k"
EXTRA_ID = "facebook/deit-small-patch16-224"
TRAIN_CFG = {
    "vit_scratch":    dict(epochs=150, warmup_epochs=10, batch_size=64, lr=5e-4, head_lr_mult=1.0,  weight_decay=0.05, ckpt_every=25),
    "vit_pretrained": dict(epochs=12,  warmup_epochs=1,  batch_size=32, lr=5e-5, head_lr_mult=10.0, weight_decay=0.05, ckpt_every=0),
    "resnet18":       dict(epochs=12,  warmup_epochs=1,  batch_size=32, lr=1e-4, head_lr_mult=10.0, weight_decay=0.05, ckpt_every=0),
    "deit_small":     dict(epochs=12,  warmup_epochs=1,  batch_size=32, lr=5e-5, head_lr_mult=10.0, weight_decay=0.05, ckpt_every=0),
}
LABEL_SMOOTHING, GRAD_CLIP, MIN_LR_RATIO = 0.1, 1.0, 0.01
AUG_JITTER = 0.10         # brilho e contraste em [0,9; 1,1]

# %%
try:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT_DIR = Path("/content/drive/MyDrive/infnet_cv_projeto/outputs/A1")
except Exception as e:
    print(f"Drive indisponível ({e!r}); usando armazenamento local do runtime.")
    OUT_DIR = Path("/content/outputs/A1") if Path("/content").exists() else Path("outputs/A1")
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
print("Artefatos em:", OUT_DIR)
FIGURES = []

def savefig(name):
    path = OUT_DIR / f"A1_{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    FIGURES.append(str(path))

# %%
def load_kaggle_credentials():
    try:
        from google.colab import userdata
    except ImportError:
        print("Fora do Colab: usando credenciais do ambiente (~/.kaggle ou variáveis).")
        return
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        try:
            os.environ[name] = userdata.get(name)
        except Exception:
            pass
    found = [n for n in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN") if os.environ.get(n)]
    print("Secrets do Kaggle carregados:", found or "nenhum (o dataset é público; o download deve funcionar mesmo assim)")

load_kaggle_credentials()
import kagglehub
DATASET_PATH = Path(kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database"))
print("Dataset em:", DATASET_PATH)

# %% [markdown]
# ## 2. Problema e dados
# **Problema.** Em uma linha de laminação a quente, a chapa de aço passa a vários metros por segundo; a inspeção visual humana é cansativa, inconsistente entre turnos e não acompanha a velocidade. Classificar automaticamente o tipo de defeito permite rastrear a causa no processo (cilindro danificado, carepa não removida, inclusões da aciaria) e decidir se a bobina é rebaixada ou sucateada. O NEU Surface Defect Database (Song & Yan, Northeastern University, 2013) é o benchmark clássico desse problema: **6 classes × 300 imagens 200×200 em tons de cinza** — *crazing* (rede de microfissuras), *inclusion* (partículas incrustadas), *patches* (manchas), *pitted_surface* (pites/corrosão), *rolled-in_scale* (carepa laminada na superfície) e *scratches* (riscos).
#
# **Por que este dataset é um bom teste para ViT.** (i) É **pequeno** (1.800 imagens): expõe a "fome de dados" do ViT treinado do zero. (ii) As imagens são **texturas**, não objetos centrados: o defeito ocupa a imagem toda (crazing, rolled-in scale) ou aparece em pontos/linhas espalhados (inclusion, scratches). (iii) Está **longe do ImageNet** (cinza, sem objetos), o que testa quanto a transferência de um pré-treino em fotos naturais ajuda.
#
# **Dados.** O pacote do Kaggle traz o NEU-DET já dividido em `train/images/<classe>` (240 por classe) e `validation/images/<classe>` (60 por classe). As duas partes são reunidas e redivididas (seção 3) com um split estratificado 70/15/15 próprio, para ter um conjunto de teste separado da validação usada na escolha da época. A varredura abaixo abre cada arquivo, levanta formato, modo de cor e tamanho, verifica se imagens RGB são de fato cinza (canais iguais) e detecta duplicatas exatas por MD5.

# %%
CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NATIVE_SIZE = 200

def infer_class(path):
    """Classe pela pasta (NEU-DET/<split>/images/<classe>/) ou, em último caso, pelo prefixo do nome do arquivo."""
    for part in reversed(path.parts[:-1]):
        if part in CLASS_TO_IDX:
            return part
    stem = re.sub(r"_\d+$", "", path.stem)
    return stem if stem in CLASS_TO_IDX else None

records, arrays, skipped = [], [], []
for p in sorted(DATASET_PATH.rglob("*")):
    if not p.is_file() or p.suffix.lower() not in IMG_EXT:
        continue
    cls = infer_class(p)
    if cls is None:
        skipped.append(str(p.relative_to(DATASET_PATH)))
        continue
    rec = {"path": str(p), "cls": cls, "label": CLASS_TO_IDX[cls], "ok": True, "error": "",
           "origin": next((s for s in ("train", "validation", "valid", "test") if s in [q.lower() for q in p.parts]), "?")}
    try:
        rec["md5"] = hashlib.md5(p.read_bytes()).hexdigest()
        with Image.open(p) as im:
            im.load()
            rec.update(format=im.format, mode=im.mode, width=im.size[0], height=im.size[1])
            if im.mode == "RGB":
                a = np.asarray(im, dtype=np.int16)
                rec["rgb_is_gray"] = bool((np.abs(a[..., 0] - a[..., 1]).max() == 0) and (np.abs(a[..., 1] - a[..., 2]).max() == 0))
            g = im.convert("L")
            if g.size != (NATIVE_SIZE, NATIVE_SIZE):
                g = g.resize((NATIVE_SIZE, NATIVE_SIZE), Image.BILINEAR)
            arr = np.asarray(g, dtype=np.uint8)
        rec.update(mean=float(arr.mean()), std=float(arr.std()))
        arrays.append(arr)
    except Exception as e:
        rec.update(ok=False, error=repr(e))
        arrays.append(None)
    records.append(rec)

df_all = pd.DataFrame(records)
df_all["arr_idx"] = range(len(df_all))
corrupted = df_all[~df_all.ok]
df_ok = df_all[df_all.ok].copy()
print(f"Arquivos de imagem com classe: {len(df_all)} | corrompidos: {len(corrupted)} | sem classe identificável: {len(skipped)}")
if skipped:
    print("Sem classe (amostra):", skipped[:5])
print("Formato:", df_ok.format.value_counts().to_dict(), "| modo de cor:", df_ok["mode"].value_counts().to_dict())
print("Tamanhos:", df_ok.groupby(["width", "height"]).size().to_dict())
if "rgb_is_gray" in df_ok:
    rgb = df_ok[df_ok["mode"] == "RGB"]
    print(f"Imagens RGB: {len(rgb)}, das quais com os 3 canais idênticos (cinza salvo como RGB): {int(rgb.rgb_is_gray.sum())}")
print("\nOrigem (pasta do Kaggle) × classe:\n", pd.crosstab(df_ok.cls, df_ok.origin).to_string())

# %%
md5_classes = df_ok.groupby("md5").cls.nunique()
conflicting = md5_classes[md5_classes > 1].index
n_dup = int(df_ok.duplicated("md5").sum())
df = df_ok[~df_ok.md5.isin(conflicting)].drop_duplicates("md5", keep="first").reset_index(drop=True)
print(f"Duplicatas exatas removidas: {n_dup} (hashes em mais de uma classe, removidos por completo: {len(conflicting)})")
counts = df.cls.value_counts().reindex(CLASS_NAMES)
print(f"Imagens válidas e únicas: {len(df)}\n")
print(counts.to_string())
assert counts.notna().all() and (counts > 0).all(), "Alguma classe ficou sem imagens"

X_np = np.stack([arrays[i] for i in df.arr_idx])          # [N, 200, 200] uint8
Y_np = df.label.to_numpy()
del arrays
print("Tensor de imagens:", X_np.shape, X_np.dtype, f"({X_np.nbytes / 2**20:.0f} MB)")

# %%
# fig: samples_grid
N_PER_CLASS = 6
rng = np.random.default_rng(SEED)
fig, axes = plt.subplots(NUM_CLASSES, N_PER_CLASS, figsize=(1.9 * N_PER_CLASS, 1.9 * NUM_CLASSES))
for r, cls in enumerate(CLASS_NAMES):
    rows = rng.choice(np.flatnonzero(Y_np == r), N_PER_CLASS, replace=False)
    for c, i in enumerate(rows):
        ax = axes[r, c]
        ax.imshow(X_np[i], cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if c == 0:
            ax.set_ylabel(cls, fontsize=11, fontweight="bold")
plt.suptitle("NEU Surface Defects: amostras aleatórias por classe (200×200, cinza)", fontweight="bold")
plt.tight_layout()
savefig("samples_grid")
plt.show()

# %%
# fig: intensity_stats
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, col, title in [(axes[0], "mean", "Brilho médio por imagem"), (axes[1], "std", "Contraste (desvio-padrão) por imagem")]:
    ax.boxplot([df[df.cls == c][col] for c in CLASS_NAMES], showfliers=True)
    ax.set_xticks(range(1, NUM_CLASSES + 1), CLASS_NAMES, rotation=20)
    ax.set_title(title, fontweight="bold"); ax.set_ylabel("nível de cinza (0–255)")
plt.suptitle("Estatísticas de intensidade por classe", fontweight="bold")
plt.tight_layout()
savefig("intensity_stats")
plt.show()
print(df.groupby("cls")[["mean", "std"]].agg(["mean", "std"]).round(1).to_string())

# %% [markdown]
# **Análise da EDA**
# <!-- ANALISE: (1) confirmar modo de cor/formato (JPEG L ou RGB com canais iguais) e tamanho 200×200 em todas; duplicatas/corrompidas encontradas (se houver, citar quantas e de que classe); (2) balanceamento perfeito (300/classe → macro-F1 ≈ accuracy); (3) observações visuais: texturas sem objeto centrado; crazing e rolled-in_scale cobrem a imagem toda; inclusion e pitted_surface são pontos escuros espalhados; scratches são linhas finas, quase sempre na vertical/horizontal (direção de laminação); patches são manchas grandes e claras/escuras; (4) estatísticas de intensidade: se o brilho médio separa classes (ex.: patches mais claras/escuras), o modelo pode usar o brilho como atalho — por isso o jitter de brilho/contraste é leve (±10%) e não forte; citar os números do boxplot. -->

# %% [markdown]
# ## 3. Split estratificado 70/15/15 e pipeline de entrada
# **Split.** `train_test_split` aplicado duas vezes com `stratify` e `random_state=42` (70% treino; os 30% restantes divididos ao meio em validação e teste), **o mesmo para os três modelos**. A divisão original train/validation do Kaggle é ignorada. O split é salvo em CSV.
#
# **Pipeline na GPU.** O dataset inteiro (~72 MB em uint8) vai para a GPU uma única vez; cada batch é redimensionado para 224 (bilinear, como o `ViTImageProcessor`), aumentado e normalizado na própria GPU. Isso elimina o gargalo de CPU do Colab (2 vCPUs) e deixa o tempo de treino medido refletir o modelo, não o carregamento. A normalização depende do modelo: estatísticas do próprio treino para o ViT do zero (1 canal), `image_mean/std` do processor do ViT pré-treinado e média/desvio do ImageNet para a ResNet-18 (3 canais).
#
# **Augmentations escolhidas (só no treino), com a justificativa para aço laminado:**
# 1. **Rotações de 90° e flips (grupo diedral D4: 8 orientações).** A imagem é uma vista de topo de uma superfície plana, sem "cima" natural: um risco rotacionado 90° continua sendo um risco, e uma inclusão espelhada continua sendo uma inclusão. As transformações são exatas na grade de pixels (sem interpolação nem bordas pretas), portanto não criam artefatos que o modelo possa aprender. Ressalva: *scratches* e *rolled-in_scale* tendem a se alinhar com a direção de laminação; ao rotacionar 90° o modelo passa a ver riscos transversais, que são raros na linha, mas continuam sendo riscos (o rótulo é preservado). O ganho é multiplicar por 8 a diversidade de orientação num dataset de 1.260 imagens.
# 2. **Jitter leve de brilho e contraste (±10%).** Simula variação de iluminação e de refletância da chapa entre bobinas e câmeras. É leve de propósito: o brilho médio difere entre classes (EDA), e um jitter forte apagaria uma pista legítima.
#
# **Descartadas, com o motivo:**
# - **Rotações arbitrárias** (ex.: 30°): exigem interpolação e preenchimento de cantos, criando bordas artificiais que não existem nas imagens reais.
# - **Recortes agressivos** (`RandomResizedCrop` com escala pequena): podem remover o único defeito da imagem em *inclusion* ou *pitted_surface* (pontos esparsos) e deixar só a textura de fundo com o rótulo do defeito.
# - **CutMix / Mixup** (usados no DeiT e no Crash Course): colariam a textura de um defeito sobre a de outro, criando chapas com dois defeitos e rótulo misto proporcional à área. Em defeitos pontuais, a área colada pode conter ou não o defeito, e o rótulo proporcional fica errado. Ficam registrados como alternativa de regularização para o ViT do zero, a testar em trabalho futuro (seção 11).

# %%
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df.label, random_state=SEED)
split_table = pd.DataFrame({name: d.cls.value_counts().reindex(CLASS_NAMES) for name, d in
                            [("train", train_df), ("val", val_df), ("test", test_df)]})
split_table.loc["TOTAL"] = split_table.sum()
print(split_table.to_string())
assert not (set(train_df.md5) & set(val_df.md5) or set(train_df.md5) & set(test_df.md5) or set(val_df.md5) & set(test_df.md5))
pd.concat([d.assign(split=s) for s, d in [("train", train_df), ("val", val_df), ("test", test_df)]])[
    ["path", "cls", "label", "origin", "md5", "split"]].to_csv(OUT_DIR / "A1_split.csv", index=False)

X_all = torch.from_numpy(X_np).unsqueeze(1).to(device)        # [N, 1, 200, 200] uint8 na GPU
Y_all = torch.tensor(Y_np, dtype=torch.long, device=device)
train_idx, val_idx, test_idx = (torch.tensor(d.index.to_numpy(), device=device) for d in (train_df, val_df, test_df))
TRAIN_MEAN = float(X_all[train_idx].float().mean() / 255)
TRAIN_STD = float(X_all[train_idx].float().std() / 255)
print(f"\nNormalização do ViT do zero (estatísticas do treino): média {TRAIN_MEAN:.4f} | desvio {TRAIN_STD:.4f}")

# %%
def augment(x):
    """x: [B, 1, S, S] em [0, 1]. Orientação aleatória do grupo D4 + jitter leve de brilho/contraste, por imagem."""
    B = x.size(0)
    k = torch.randint(0, 4, (B,), device=x.device)
    out = torch.empty_like(x)
    for kk in range(4):
        sel = k == kk
        if sel.any():
            out[sel] = torch.rot90(x[sel], kk, dims=(2, 3))
    flip = torch.rand(B, 1, 1, 1, device=x.device) < 0.5
    out = torch.where(flip, out.flip(3), out)                  # rot90^k (+ flip) cobre as 8 orientações
    c = 1 + (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * AUG_JITTER
    b = 1 + (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * AUG_JITTER
    m = out.mean(dim=(2, 3), keepdim=True)
    return (((out - m) * c + m) * b).clamp(0, 1)

def make_prep(size, mean, std, channels):
    """Devolve prep(x_uint8, train) -> tensor normalizado [B, channels, size, size] pronto para o modelo."""
    mean_t = torch.tensor(mean, device=device, dtype=torch.float32).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=torch.float32).view(1, -1, 1, 1)
    def prep(x_u8, train=False):
        x = x_u8.float() / 255
        if x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=size < x.shape[-1])
        if train:
            x = augment(x)
        return (x.expand(-1, channels, -1, -1) - mean_t) / std_t
    return prep

prep_scratch = make_prep(SCRATCH_CFG["img_size"], [TRAIN_MEAN], [TRAIN_STD], SCRATCH_CFG["in_chans"])
xb = prep_scratch(X_all[train_idx[:64]], train=True)
print("batch do ViT do zero:", tuple(xb.shape), f"| média {xb.mean():.3f} desvio {xb.std():.3f}")

# %%
# fig: augmentation_examples
def show01(ax, img, title=None):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold")

def cutmix_demo(a, b, frac=0.5):
    out, s = a.clone(), int(a.shape[-1] * math.sqrt(frac))
    out[..., :s, :s] = b[..., :s, :s]
    return out

demo_rows = [int(train_df.index[train_df.label == k][0]) for k in range(NUM_CLASSES)]
base = X_all[demo_rows].float() / 255
AUG_DEMO = {"original": lambda x, i: x, "rot 90°": lambda x, i: torch.rot90(x, 1, (1, 2)),
            "flip H": lambda x, i: x.flip(2), "flip V": lambda x, i: x.flip(1),
            "brilho/contraste −10%": lambda x, i: (((x - x.mean()) * 0.9 + x.mean()) * 0.9).clamp(0, 1),
            "brilho/contraste +10%": lambda x, i: (((x - x.mean()) * 1.1 + x.mean()) * 1.1).clamp(0, 1),
            "CutMix (descartado)": lambda x, i: cutmix_demo(x, base[(i + 3) % NUM_CLASSES])}
fig, axes = plt.subplots(NUM_CLASSES, len(AUG_DEMO), figsize=(1.9 * len(AUG_DEMO), 1.9 * NUM_CLASSES))
for r in range(NUM_CLASSES):
    for c, (name, fn) in enumerate(AUG_DEMO.items()):
        show01(axes[r, c], fn(base[r], r)[0].cpu(), name if r == 0 else None)
        if c == 0:
            axes[r, c].set_ylabel(CLASS_NAMES[r], fontsize=10, fontweight="bold")
plt.suptitle("Augmentations usadas no treino (colunas 2–6) e CutMix, descartado (col. 7: 1/4 da chapa vem de outra classe)", fontweight="bold")
plt.tight_layout()
savefig("augmentation_examples")
plt.show()

# %% [markdown]
# ## 4. Módulos do Transformer implementados do zero — **Rubricas 2.1, 2.3 e 3.1**
# **Decisão técnica.** Os módulos seguem a Aula 2 (SDPA, FFN, `EncoderLayer` Pre-LN) com três mudanças: (i) a **Multi-Head Attention tem projeções Q/K/V independentes por head** (`nn.ModuleList` de `AttentionHead`, cada uma com `Linear(d_model → d_k)` próprias), concatenadas com `torch.cat` e projetadas por `W_O`, que é a forma literal da definição; (ii) o bloco **devolve e guarda** os pesos de atenção (`last_attn`), necessários para os mapas da seção 10; (iii) o SDPA devolve os pesos **antes** do dropout, para que cada linha seja uma distribuição (soma 1) também no treino.
#
# $$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,\qquad \mathrm{head}_i=\mathrm{Attention}(XW_i^Q,\,XW_i^K,\,XW_i^V),\qquad \mathrm{MHA}(X)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,W^O$$
#
# **Convenção de máscara.** `mask` booleana (ou 0/1) com **`True`/1 = pode atender** e `False`/0 = bloqueado, a mesma de `F.scaled_dot_product_attention`. Posições bloqueadas recebem o menor valor finito do dtype antes do softmax (e não `-1e9`, que estoura em fp16 sob AMP). Atenção: `nn.MultiheadAttention` usa a convenção **oposta** (`True` = bloqueado); o teste abaixo passa `~mask` para ela. O ViT não usa máscara (todos os patches se veem), mas o suporte é testado com uma máscara causal.

# %%
class ScaledDotProductAttention(nn.Module):
    """softmax(Q K^T / sqrt(d_k) + máscara) V. Retorna (saída, pesos de atenção)."""
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None):
        # q: [..., T_q, d_k] | k: [..., T_k, d_k] | v: [..., T_k, d_v] | mask: broadcast para [..., T_q, T_k]
        d_k = q.size(-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)                   # [..., T_q, T_k]
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)                                   # [..., T_q, T_k], cada linha soma 1
        out = self.dropout(weights) @ v                                    # [..., T_q, d_v]
        return out, weights


class AttentionHead(nn.Module):
    """Uma head: projeções Q, K e V próprias (d_model -> d_k) + SDPA."""
    def __init__(self, d_model: int, d_k: int, attn_dropout: float = 0.0):
        super().__init__()
        self.q = nn.Linear(d_model, d_k)
        self.k = nn.Linear(d_model, d_k)
        self.v = nn.Linear(d_model, d_k)
        self.attention = ScaledDotProductAttention(attn_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # x: [B, T, d_model] -> saída [B, T, d_k], pesos [B, T, T]
        return self.attention(self.q(x), self.k(x), self.v(x), mask)


class MultiHeadAttention(nn.Module):
    """h heads independentes -> torch.cat -> W_O. Retorna (saída [B, T, d_model], pesos [B, h, T, T])."""
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model deve ser divisível por n_heads"
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        self.heads = nn.ModuleList([AttentionHead(d_model, self.d_k, attn_dropout) for _ in range(n_heads)])
        self.w_o = nn.Linear(d_model, d_model)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        outs, weights = zip(*(head(x, mask) for head in self.heads))     # h × [B, T, d_k], h × [B, T, T]
        concat = torch.cat(outs, dim=-1)                                   # [B, T, h·d_k] = [B, T, d_model]
        return self.proj_dropout(self.w_o(concat)), torch.stack(weights, dim=1)   # [B, T, d_model], [B, h, T, T]


class FusedMultiHeadAttention(nn.Module):
    """Versão fundida da Aula 2 (uma matriz d_model×d_model fatiada por reshape). Só serve de referência nos testes."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads, self.d_k = n_heads, d_model // n_heads
        self.w_q, self.w_k, self.w_v, self.w_o = (nn.Linear(d_model, d_model) for _ in range(4))
        self.attention = ScaledDotProductAttention(0.0)

    @torch.no_grad()
    def load_from_heads(self, mha: MultiHeadAttention):
        for name in ("q", "k", "v"):
            fused = getattr(self, f"w_{name}")
            fused.weight.copy_(torch.cat([getattr(h, name).weight for h in mha.heads], dim=0))
            fused.bias.copy_(torch.cat([getattr(h, name).bias for h in mha.heads], dim=0))
        self.w_o.load_state_dict(mha.w_o.state_dict())
        return self

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        B, T, D = x.shape
        split = lambda t: t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)   # [B, h, T, d_k]
        out, w = self.attention(split(self.w_q(x)), split(self.w_k(x)), split(self.w_v(x)), mask)
        return self.w_o(out.transpose(1, 2).reshape(B, T, D)), w


class MLP(nn.Module):
    """FFN de 2 camadas: Linear(d, 4d) -> GELU -> Dropout -> Linear(4d, d) -> Dropout."""
    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class TransformerEncoderBlock(nn.Module):
    """Bloco Pre-LN: x = x + MHA(LN(x)); x = x + MLP(LN(x)). Guarda os pesos de atenção em `last_attn`."""
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 attn_dropout: float = 0.0, eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=eps)
        self.attn = MultiHeadAttention(d_model, n_heads, attn_dropout, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=eps)
        self.mlp = MLP(d_model, int(d_model * mlp_ratio), dropout)
        self.last_attn = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        attn_out, weights = self.attn(self.norm1(x), mask)                 # [B, T, D], [B, h, T, T]
        x = x + attn_out                                                   # residual 1
        x = x + self.mlp(self.norm2(x))                                    # residual 2
        self.last_attn = weights.detach()
        return x, weights


class PatchEmbedding(nn.Module):
    """Conv2d com kernel = stride = patch: equivale a recortar patches P×P e aplicar a mesma projeção linear em cada um."""
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size deve ser múltiplo de patch_size"
        self.patch_size, self.grid = patch_size, img_size // patch_size
        self.n_patches = self.grid ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, D, H/P, W/P] -> [B, D, N] -> [B, N, D]
        return self.proj(x).flatten(2).transpose(1, 2)


class ViT(nn.Module):
    """Imagem -> patches -> [CLS; patches] + pos_embed -> N blocos -> LN -> head(CLS) -> logits."""
    def __init__(self, img_size=224, patch_size=16, in_chans=1, n_classes=6, embed_dim=192, depth=6,
                 n_heads=3, mlp_ratio=4.0, dropout=0.1, attn_dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.grid = self.patch_embed.grid
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))                          # CLS aprendível
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))  # PE 1D aprendível
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerEncoderBlock(embed_dim, n_heads, mlp_ratio, dropout, attn_dropout)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward_tokens(self, x: torch.Tensor, use_pe: bool = True):
        tokens = self.patch_embed(x)                                       # [B, N, D]
        cls = self.cls_token.expand(x.size(0), -1, -1)                     # [B, 1, D]
        z = torch.cat([cls, tokens], dim=1)                                # [B, 1+N, D]
        if use_pe:
            z = z + self.pos_embed                                         # soma posição a cada token
        z = self.pos_drop(z)
        attns = []
        for blk in self.blocks:
            z, a = blk(z)                                                  # a: [B, h, 1+N, 1+N]
            attns.append(a)
        return self.norm(z), attns

    def forward(self, x: torch.Tensor, return_attn: bool = False, use_pe: bool = True):
        z, attns = self.forward_tokens(x, use_pe)
        logits = self.head(z[:, 0])                                        # [B, n_classes], a partir do CLS
        return (logits, attns) if return_attn else logits

# %% [markdown]
# **Testes (asserts).** Rodam na CPU em float64, com tensores pequenos (e o ViT no tamanho real), em `eval()` (dropout desligado). Verificam: shapes; linhas da atenção somando 1; equivalência numérica do SDPA com `F.scaled_dot_product_attention` (sem máscara e com máscara causal); equivalência da MHA de heads independentes com a versão fundida da Aula 2 e com `nn.MultiheadAttention` (copiando os pesos); equivalência do bloco com `nn.TransformerEncoderLayer(norm_first=True)`; a propriedade residual (com `W_O` e `fc2` zerados o bloco vira a identidade); a igualdade entre o patch embedding por Conv2d e "recortar patches + Linear"; o forward do ViT com batch aleatório; e a contagem de parâmetros contra a fórmula fechada.

# %%
torch.manual_seed(SEED)
dt = torch.float64
B, T, D, H = 2, 7, 12, 3
x = torch.randn(B, T, D, dtype=dt)
causal = torch.tril(torch.ones(T, T, dtype=torch.bool))              # True = pode atender

# 1) SDPA
sdpa = ScaledDotProductAttention().eval()
q, k, v = torch.randn(3, B, H, T, D // H, dtype=dt).unbind(0)
for m in (None, causal):
    out, w = sdpa(q, k, v, m)
    assert out.shape == (B, H, T, D // H) and w.shape == (B, H, T, T)
    assert torch.allclose(w.sum(-1), torch.ones(B, H, T, dtype=dt)), "linhas da atenção não somam 1"
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v, attn_mask=m), atol=1e-10)
assert torch.all(w.masked_select(~causal) == 0), "máscara causal vazou"
print("OK SDPA: shapes, linhas somam 1, igual a F.scaled_dot_product_attention (sem e com máscara)")

# 2) MHA com heads independentes vs fundida vs nn.MultiheadAttention
mha = MultiHeadAttention(D, H).to(dt).eval()
fused = FusedMultiHeadAttention(D, H).to(dt).load_from_heads(mha).eval()
ref = nn.MultiheadAttention(D, H, batch_first=True).to(dt).eval()
with torch.no_grad():
    ref.in_proj_weight.copy_(torch.cat([fused.w_q.weight, fused.w_k.weight, fused.w_v.weight]))
    ref.in_proj_bias.copy_(torch.cat([fused.w_q.bias, fused.w_k.bias, fused.w_v.bias]))
    ref.out_proj.load_state_dict(mha.w_o.state_dict())
    for m in (None, causal):
        y, w = mha(x, m)
        y_f, w_f = fused(x, m)
        y_r, w_r = ref(x, x, x, attn_mask=None if m is None else ~m, need_weights=True, average_attn_weights=False)
        assert y.shape == (B, T, D) and w.shape == (B, H, T, T)
        assert torch.allclose(w.sum(-1), torch.ones(B, H, T, dtype=dt))
        assert torch.allclose(y, y_f, atol=1e-10) and torch.allclose(w, w_f, atol=1e-10)
        assert torch.allclose(y, y_r, atol=1e-10) and torch.allclose(w, w_r, atol=1e-10)
n_mha = sum(p.numel() for p in mha.parameters())
assert n_mha == 4 * (D * D + D) == sum(p.numel() for p in ref.parameters())
print(f"OK MHA: {H} heads independentes == versão fundida == nn.MultiheadAttention (saída e pesos); {n_mha} parâmetros")

# 3) Bloco encoder Pre-LN vs nn.TransformerEncoderLayer(norm_first=True)
blk = TransformerEncoderBlock(D, H, mlp_ratio=4.0, dropout=0.0).to(dt).eval()
ref_blk = nn.TransformerEncoderLayer(D, H, dim_feedforward=4 * D, dropout=0.0, activation="gelu",
                                     layer_norm_eps=1e-6, batch_first=True, norm_first=True).to(dt).eval()
with torch.no_grad():
    fused.load_from_heads(blk.attn)
    ref_blk.self_attn.in_proj_weight.copy_(torch.cat([fused.w_q.weight, fused.w_k.weight, fused.w_v.weight]))
    ref_blk.self_attn.in_proj_bias.copy_(torch.cat([fused.w_q.bias, fused.w_k.bias, fused.w_v.bias]))
    ref_blk.self_attn.out_proj.load_state_dict(blk.attn.w_o.state_dict())
    ref_blk.linear1.load_state_dict(blk.mlp.fc1.state_dict()); ref_blk.linear2.load_state_dict(blk.mlp.fc2.state_dict())
    ref_blk.norm1.load_state_dict(blk.norm1.state_dict()); ref_blk.norm2.load_state_dict(blk.norm2.state_dict())
    y, w = blk(x)
    assert y.shape == x.shape and w.shape == (B, H, T, T) and blk.last_attn is not None
    assert torch.allclose(y, ref_blk(x), atol=1e-10), "bloco difere de nn.TransformerEncoderLayer"
    blk_id = TransformerEncoderBlock(D, H).to(dt).eval()
    for lin in (blk_id.attn.w_o, blk_id.mlp.fc2):
        nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
    assert torch.allclose(blk_id(x)[0], x), "com os ramos zerados o bloco deveria ser a identidade (residual)"
n_blk = sum(p.numel() for p in blk.parameters())
assert n_blk == sum(p.numel() for p in ref_blk.parameters())
print(f"OK bloco: igual a nn.TransformerEncoderLayer(norm_first=True, gelu); residual verificado; {n_blk} parâmetros")

# 4) PatchEmbedding: Conv2d == recortar patches + Linear
pe = PatchEmbedding(32, 8, 1, D).to(dt)
img = torch.randn(B, 1, 32, 32, dtype=dt)
with torch.no_grad():
    patches = F.unfold(img, kernel_size=8, stride=8).transpose(1, 2)        # [B, N, C·P·P]
    manual = patches @ pe.proj.weight.view(D, -1).T + pe.proj.bias
    assert pe(img).shape == (B, 16, D) and torch.allclose(pe(img), manual, atol=1e-10)
print("OK PatchEmbedding: Conv2d(stride=P) == unfold + projeção linear, shape [B, N, D]")

# 5) ViT completo no tamanho real
def expected_vit_params(img_size, patch_size, in_chans, embed_dim, depth, n_heads, mlp_ratio, n_classes, **_):
    Dm, N, Hd = embed_dim, (img_size // patch_size) ** 2, int(embed_dim * mlp_ratio)
    block = 2 * Dm + 3 * (Dm * Dm + Dm) + (Dm * Dm + Dm) + 2 * Dm + (Dm * Hd + Hd) + (Hd * Dm + Dm)
    return (in_chans * patch_size ** 2 * Dm + Dm) + Dm + (1 + N) * Dm + depth * block + 2 * Dm + (Dm * n_classes + n_classes)

vit_test = ViT(**SCRATCH_CFG, n_classes=NUM_CLASSES).eval()
xin = torch.randn(4, SCRATCH_CFG["in_chans"], SCRATCH_CFG["img_size"], SCRATCH_CFG["img_size"])
with torch.no_grad():
    logits, attns = vit_test(xin, return_attn=True)
T_vit = 1 + vit_test.patch_embed.n_patches
assert logits.shape == (4, NUM_CLASSES) and torch.isfinite(logits).all()
assert len(attns) == SCRATCH_CFG["depth"] and all(a.shape == (4, SCRATCH_CFG["n_heads"], T_vit, T_vit) for a in attns)
assert all(torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5) for a in attns)
n_vit = sum(p.numel() for p in vit_test.parameters())
assert n_vit == expected_vit_params(**SCRATCH_CFG, n_classes=NUM_CLASSES)
print(f"OK ViT: {tuple(xin.shape)} -> logits {tuple(logits.shape)}; {len(attns)} mapas [B, h, {T_vit}, {T_vit}] com linhas somando 1; "
      f"{n_vit:,} parâmetros (= fórmula fechada)")
print("\nTodos os testes passaram.")

# %%
print(f"{'módulo':<28}{'parâmetros':>12}")
for name, mod in [("patch_embed", vit_test.patch_embed), ("cls_token + pos_embed", None), ("1 bloco encoder", vit_test.blocks[0]),
                  (f"{SCRATCH_CFG['depth']} blocos", vit_test.blocks), ("norm final + head", nn.ModuleList([vit_test.norm, vit_test.head]))]:
    n = (vit_test.cls_token.numel() + vit_test.pos_embed.numel()) if mod is None else sum(p.numel() for p in mod.parameters())
    print(f"{name:<28}{n:>12,}")
print(f"{'TOTAL':<28}{n_vit:>12,}")
del vit_test

# %% [markdown]
# ## 5. Positional encoding: por que a atenção sem PE não preserva a posição — **Rubrica 2.4**
# **Argumento.** Seja $X\in\mathbb{R}^{N\times d}$ a matriz dos tokens de patch e $P$ uma matriz de permutação. Na self-attention, $Q=XW^Q$, $K=XW^K$, $V=XW^V$; permutar as linhas da entrada dá $PQ$, $PK$, $PV$ e
# $$\mathrm{softmax}\!\left(\frac{PQ(PK)^\top}{\sqrt{d_k}}\right)PV=P\,\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)P^\top P\,V=P\,\mathrm{Attention}(Q,K,V),$$
# porque o softmax por linha comuta com a permutação de linhas e colunas e $P^\top P=I$. LayerNorm, MLP e as conexões residuais agem **token a token**, então o encoder inteiro é **equivariante a permutações**: embaralhar os patches só embaralha as saídas. O CLS fica fixo na posição 0 e atende a um **conjunto** de patches (a soma ponderada não depende da ordem), logo sua saída — e os logits — são **invariantes**. Sem PE, o ViT vê a imagem como um "saco de patches": uma chapa com um risco contínuo e a mesma chapa com os patches do risco espalhados são indistinguíveis. (As falas da aula dizem "invariante"; o termo preciso é equivariante para os tokens e invariante para o CLS.)
#
# **Solução no ViT.** $z_0=[x_{\text{class}};\,x_p^1E;\,\dots;\,x_p^NE]+E_{pos}$, com $E_{pos}\in\mathbb{R}^{(1+N)\times d}$ **aprendível** (1D, como no ViT original; o paper testou PE 2D sem ganho). Somar $E_{pos}[i]$ ao token $i$ quebra a simetria: o mesmo conteúdo em posições diferentes vira vetores diferentes, e os produtos $q\cdot k$ passam a depender da posição.
#
# **Demonstração.** Um ViT com pesos aleatórios (o argumento não depende do treino), em float64 na CPU, recebe uma imagem de teste e a mesma imagem com os 196 patches embaralhados. Sem PE, as saídas dos tokens são as originais permutadas e o CLS não muda; com PE, os dois mudam.

# %%
def shuffle_patches(x, perm, patch):
    """Reordena os patches P×P da imagem: o patch na posição i da saída é o patch perm[i] da entrada."""
    Bn, C, Hh, Ww = x.shape
    g = Hh // patch
    p = x.reshape(Bn, C, g, patch, g, patch).permute(0, 2, 4, 1, 3, 5).reshape(Bn, g * g, C, patch, patch)
    p = p[:, perm.to(x.device)]
    return p.reshape(Bn, g, g, C, patch, patch).permute(0, 3, 1, 4, 2, 5).reshape(Bn, C, Hh, Ww)

vit_demo = ViT(**SCRATCH_CFG, n_classes=NUM_CLASSES).double().eval()
x_img = prep_scratch(X_all[test_idx[:1]]).cpu().double()
perm = torch.randperm(vit_demo.patch_embed.n_patches, generator=torch.Generator().manual_seed(SEED))
x_shuf = shuffle_patches(x_img, perm, SCRATCH_CFG["patch_size"])
pe_demo = {}
with torch.no_grad():
    assert torch.allclose(vit_demo.patch_embed(x_shuf), vit_demo.patch_embed(x_img)[:, perm])
    for use_pe in (False, True):
        z, _ = vit_demo.forward_tokens(x_img, use_pe)
        zs, _ = vit_demo.forward_tokens(x_shuf, use_pe)
        pe_demo[use_pe] = {
            "tokens_equivariant": bool(torch.allclose(zs[:, 1:], z[:, 1:][:, perm], atol=1e-10)),
            "cls_invariant": bool(torch.allclose(zs[:, 0], z[:, 0], atol=1e-10)),
            "max_abs_diff_tokens": float((zs[:, 1:] - z[:, 1:][:, perm]).abs().max()),
            "max_abs_diff_cls": float((zs[:, 0] - z[:, 0]).abs().max()),
            "logits": vit_demo.head(z[:, 0])[0].tolist(), "logits_shuf": vit_demo.head(zs[:, 0])[0].tolist()}
print(pd.DataFrame(pe_demo).T.drop(columns=["logits", "logits_shuf"]).rename(index={False: "sem PE", True: "com PE"}).to_string())
assert pe_demo[False]["tokens_equivariant"] and pe_demo[False]["cls_invariant"]
assert not pe_demo[True]["tokens_equivariant"] and not pe_demo[True]["cls_invariant"]
print("OK: sem PE, permutar os patches só permuta as saídas e o CLS é idêntico; com PE, ambos mudam.")

# %%
# fig: pe_permutation_demo
fig, axes = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 1, 1.6]})
denorm = lambda t: (t[0, 0] * TRAIN_STD + TRAIN_MEAN).clamp(0, 1).numpy()
show01(axes[0], denorm(x_img), "imagem original (teste)")
show01(axes[1], denorm(x_shuf), "196 patches embaralhados")
labels = ["tokens de patch\n(saída permutada vs original)", "CLS\n(saída)"]
xs = np.arange(2)
for off, use_pe, color in [(-0.2, False, C_BLUE), (0.2, True, C_ORANGE)]:
    vals = [max(pe_demo[use_pe]["max_abs_diff_tokens"], 1e-17), max(pe_demo[use_pe]["max_abs_diff_cls"], 1e-17)]
    bars = axes[2].bar(xs + off, vals, 0.4, color=color, label="com PE" if use_pe else "sem PE")
    axes[2].bar_label(bars, labels=[f"{v:.1e}" for v in vals], fontsize=8)
axes[2].set_yscale("log"); axes[2].set_ylim(1e-17, 10)
axes[2].set_xticks(xs, labels); axes[2].set_ylabel("max |Δ| (float64)")
axes[2].axhline(1e-12, color=C_RED, linestyle=":", label="limiar numérico (1e-12)")
axes[2].legend(fontsize=8)
axes[2].set_title("Diferença após embaralhar os patches (ViT com pesos aleatórios)", fontweight="bold", fontsize=10)
plt.suptitle("Sem PE: tokens só permutados e CLS idêntico (erro de arredondamento); com PE: ambos mudam", fontweight="bold")
plt.tight_layout()
savefig("pe_permutation_demo")
plt.show()
del vit_demo

# %% [markdown]
# <!-- ANALISE: comentar a tabela (diferenças ~1e-15, i.e., erro de arredondamento, sem PE; com PE, ~1e-1 nos tokens e menor no CLS, porque o PE aleatório tem desvio 0,02 — pequeno, mas 12+ ordens de grandeza acima do arredondamento; no modelo treinado o PE é aprendido e o efeito é medido na seção 10.4) e ligar ao domínio: no NEU as texturas são quase estacionárias, então a posição absoluta pesa menos do que em fotos de objetos; o experimento de patches embaralhados com os modelos TREINADOS (seção 10.4) mede quanto cada modelo depende da organização espacial. -->

# %% [markdown]
# ## 6. Treino do ViT do zero — **Rubrica 3.2**
# **Loop comum aos três modelos.** Batches sorteados por época com gerador semeado (`SEED·1000 + época`, o que torna a retomada reproduzível), AMP fp16 com `GradScaler`, *gradient clipping*, `LambdaLR` por passo (warmup linear + cosseno até 1% da LR), avaliação em validação a cada época (accuracy e macro-F1), melhor estado mantido em memória. **Tempo de treino** = soma das épocas (treino + validação, com `cuda.synchronize`). **VRAM de pico** = `max_memory_allocated` durante o treino menos o que já estava alocado antes de criar o modelo (dataset na GPU e modelos anteriores), ou seja, a memória atribuível ao modelo: pesos, gradientes, estados do AdamW e ativações.
#
# **Checkpoints no Drive.** `A1_<modelo>_final.pt` (melhores pesos + histórico + tempo + VRAM) é gravado ao fim; se existir, o treino é pulado. `A1_<modelo>_last.pt` (pesos, otimizador, scheduler, scaler, histórico) é gravado a cada `ckpt_every` épocas e permite continuar um treino interrompido.

# %%
def param_groups(model, lr, weight_decay, head_prefix, head_lr_mult):
    """Sem weight decay em bias/LayerNorm/CLS/PE; LR multiplicada no head novo."""
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = n.startswith(head_prefix)
        no_decay = p.ndim <= 1 or any(s in n for s in ("cls_token", "pos_embed", "position_embeddings", "distillation_token"))
        key = (is_head, no_decay)
        groups.setdefault(key, {"params": [], "lr": lr * (head_lr_mult if is_head else 1.0),
                                "weight_decay": 0.0 if no_decay else weight_decay})["params"].append(p)
    return list(groups.values())

def lr_lambda_factory(total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return f

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

@torch.no_grad()
def evaluate(model, prep, idx, batch_size=128, transform=None):
    """Accuracy, macro-F1 e loss (CE sem label smoothing) em um subconjunto; `transform` opcional pós-prep."""
    model.eval()
    logits = []
    for i in range(0, len(idx), batch_size):
        x = prep(X_all[idx[i:i + batch_size]], False)
        if transform is not None:
            x = transform(x)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
            logits.append(model(x).float())
    logits = torch.cat(logits)
    y = Y_all[idx]
    probs = logits.softmax(1)
    y_true, y_pred = y.cpu().numpy(), probs.argmax(1).cpu().numpy()
    return {"loss": float(F.cross_entropy(logits, y)), "acc": float(100 * (y_true == y_pred).mean()),
            "f1": float(f1_score(y_true, y_pred, average="macro")), "y_true": y_true, "y_pred": y_pred,
            "probs": probs.cpu().numpy()}

def cpu_state(model):
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

def train_model(name, build_fn, prep, cfg, head_prefix):
    """Treina (ou carrega do Drive) um modelo. Retorna (modelo com os melhores pesos, resultado)."""
    final_path, last_path = CKPT_DIR / f"A1_{name}_final.pt", CKPT_DIR / f"A1_{name}_last.pt"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); sync()
        base_mem = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    model = build_fn().to(device)
    if final_path.exists() and not FORCE_RETRAIN:
        ck = torch.load(final_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"[{name}] checkpoint final encontrado ({final_path.name}): treino pulado. "
              f"Melhor época {ck['result']['best_epoch']}, tempo original {ck['result']['train_time_s']:.0f}s.")
        return model, ck["result"]

    steps_per_epoch = math.ceil(len(train_idx) / cfg["batch_size"])
    total_steps = cfg["epochs"] * steps_per_epoch
    optimizer = torch.optim.AdamW(param_groups(model, cfg["lr"], cfg["weight_decay"], head_prefix, cfg["head_lr_mult"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(total_steps, cfg["warmup_epochs"] * steps_per_epoch))
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    history, start_epoch, elapsed, peak_prev = [], 1, 0.0, 0.0
    best = {"f1": -1.0, "loss": float("inf"), "epoch": 0, "state": None}
    if cfg["ckpt_every"] and last_path.exists() and not FORCE_RETRAIN:
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        history, best, elapsed, peak_prev = ck["history"], ck["best"], ck["elapsed"], ck["peak_vram_mb"]
        start_epoch = ck["epoch"] + 1
        print(f"[{name}] retomando da época {start_epoch} ({last_path.name})")

    n_train = len(train_idx)
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        sync(); t0 = time.perf_counter()
        model.train()
        order = torch.randperm(n_train, generator=torch.Generator().manual_seed(SEED * 1000 + epoch)).to(device)
        tr_loss, tr_correct = 0.0, 0
        for s in range(steps_per_epoch):
            idx = train_idx[order[s * cfg["batch_size"]:(s + 1) * cfg["batch_size"]]]
            x, y = prep(X_all[idx], True), Y_all[idx]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
                logits = model(x)
                loss = criterion(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            tr_loss += loss.item() * len(y)
            tr_correct += int((logits.argmax(1) == y).sum())
        va = evaluate(model, prep, val_idx)
        sync(); dt_ep = time.perf_counter() - t0
        elapsed += dt_ep
        improved = (va["f1"], -va["loss"]) > (best["f1"], -best["loss"])
        if improved:
            best = {"f1": va["f1"], "loss": va["loss"], "epoch": epoch, "state": cpu_state(model)}
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": tr_loss / n_train,
                        "train_acc": 100 * tr_correct / n_train, "val_loss": va["loss"], "val_acc": va["acc"],
                        "val_f1": va["f1"], "epoch_time_s": dt_ep})
        print(f"[{name}] ep {epoch:3d}/{cfg['epochs']} | train loss {tr_loss / n_train:.4f} acc {100 * tr_correct / n_train:6.2f}% | "
              f"val loss {va['loss']:.4f} acc {va['acc']:6.2f}% F1 {va['f1']:.4f} | {dt_ep:5.1f}s {'*' if improved else ''}")
        if cfg["ckpt_every"] and epoch % cfg["ckpt_every"] == 0 and epoch < cfg["epochs"]:
            peak_now = (torch.cuda.max_memory_allocated() - base_mem) / 2**20 if torch.cuda.is_available() else 0.0
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(), "history": history, "best": best, "elapsed": elapsed,
                        "peak_vram_mb": max(peak_prev, peak_now), "epoch": epoch}, last_path)

    peak = (torch.cuda.max_memory_allocated() - base_mem) / 2**20 if torch.cuda.is_available() else None
    model.load_state_dict(best["state"])
    result = {"history": history, "best_epoch": best["epoch"], "best_val_f1": best["f1"], "best_val_loss": best["loss"],
              "train_time_s": elapsed, "epochs": cfg["epochs"], "peak_vram_mb": None if peak is None else max(peak, peak_prev),
              "config": cfg}
    torch.save({"model": best["state"], "result": result}, final_path)
    last_path.unlink(missing_ok=True)
    print(f"[{name}] melhor época {best['epoch']} (val F1 {best['f1']:.4f}) | tempo {elapsed:.0f}s"
          + (f" | VRAM pico {result['peak_vram_mb']:.0f} MB" if peak is not None else ""))
    del optimizer, scheduler, scaler
    return model, result

MODELS = {}   # nome -> dict(model, prep, result, rótulo, ...)

# %%
vit_scratch, res = train_model("vit_scratch", lambda: ViT(**SCRATCH_CFG, n_classes=NUM_CLASSES), prep_scratch,
                               TRAIN_CFG["vit_scratch"], head_prefix="head.")
MODELS["vit_scratch"] = {"model": vit_scratch, "prep": prep_scratch, "result": res, "label": "ViT do zero",
                         "pretrain": "nenhum", "input": f"{SCRATCH_CFG['img_size']}px, 1 canal, {1 + vit_scratch.patch_embed.n_patches} tokens"}

# %% [markdown]
# **Análise do treino do ViT do zero**
# <!-- ANALISE: tempo por época na T4 (se > ~15 s, justificar/registrar a queda para 128); forma das curvas (seção 9.1): o warmup de 10 épocas, a velocidade de subida, se a train acc chega perto de 100% enquanto a val estaciona (overfitting típico de ViT sem viés indutivo); melhor época; comparar com a expectativa (ViT do zero em dataset pequeno fica bem abaixo do pré-treinado). -->

# %% [markdown]
# ## 7. Fine-tuning do ViT-B/16 pré-treinado no ImageNet-21k — **Rubrica 3.3**
# **Escolha do checkpoint: `google/vit-base-patch16-224-in21k`.** É o ViT original de Dosovitskiy et al., pré-treinado de forma **supervisionada** no ImageNet-21k (14 M imagens, 21.843 classes) e distribuído **sem head de classificação**, o que deixa a troca do head explícita e casa com a discussão da seção 11 (pré-treino supervisionado do ViT vs auto-supervisionado do BERT). É o checkpoint usado na Aula 5, no Géron e no Crash Course. ViT-B/16 (86 M parâmetros) cabe na T4 com batch 32 em fp16. Mesma grade 14×14 do ViT do zero.
#
# **Adaptações.** Head novo `Linear(768 → 6)` sobre o CLS (inicializado aleatoriamente, com LR 10× maior); cinza replicado em 3 canais; normalização com `image_mean/std` do `AutoImageProcessor` do checkpoint (lida, não reescrita à mão); `attn_implementation="eager"` para que `output_attentions=True` devolva as matrizes de atenção (as implementações SDPA/Flash não as expõem). **Loop manual** (o mesmo da seção 6) em vez do `Trainer`, para medir tempo e VRAM do mesmo jeito nos três modelos e usar a mesma augmentation. Fine-tuning **completo** (todas as camadas): o domínio (texturas de aço em cinza) está longe do ImageNet, então congelar o backbone tende a render menos.

# %%
class HFClassifier(nn.Module):
    """Adapta um modelo de classificação do Hugging Face à interface do ViT do zero: forward(x, return_attn)."""
    def __init__(self, hf_model):
        super().__init__()
        self.hf = hf_model

    def forward(self, x, return_attn=False):
        out = self.hf(pixel_values=x, output_attentions=return_attn)
        return (out.logits, list(out.attentions)) if return_attn else out.logits

def load_hf_classifier(model_id):
    hf = AutoModelForImageClassification.from_pretrained(
        model_id, num_labels=NUM_CLASSES, id2label=dict(enumerate(CLASS_NAMES)), label2id=CLASS_TO_IDX,
        ignore_mismatched_sizes=True, attn_implementation="eager")
    return HFClassifier(hf)

def hf_prep(model_id):
    proc = AutoImageProcessor.from_pretrained(model_id)
    size = proc.size.get("height") or proc.size.get("shortest_edge")
    crop = getattr(proc, "crop_size", None) or {}
    size = crop.get("height", size) if getattr(proc, "do_center_crop", False) else size
    print(f"{model_id}: resize {size} | image_mean {proc.image_mean} | image_std {proc.image_std}")
    return make_prep(size, proc.image_mean, proc.image_std, 3), size

prep_pretrained, size_pt = hf_prep(PRETRAINED_ID)
assert size_pt == 224

# %%
vit_pretrained, res = train_model("vit_pretrained", lambda: load_hf_classifier(PRETRAINED_ID), prep_pretrained,
                                  TRAIN_CFG["vit_pretrained"], head_prefix="hf.classifier")
MODELS["vit_pretrained"] = {"model": vit_pretrained, "prep": prep_pretrained, "result": res, "label": "ViT-B/16 pré-treinado",
                            "pretrain": "ImageNet-21k (supervisionado)", "input": "224px, 3 canais, 197 tokens"}
print(vit_pretrained.hf.classifier)

# %% [markdown]
# **Extra opcional: DeiT-small** (`RUN_EXTRAS = True`). `facebook/deit-small-patch16-224` (22 M parâmetros, ImageNet-1k com a receita de treino do DeiT, sem o token de destilação), mesmo loop. Serve para ancorar com dados a discussão da seção 11 sobre eficiência de dados. O processor do DeiT faz resize 256 + center crop 224; aqui as imagens vão direto para 224 (o crop cortaria 12% da chapa), com a média/desvio do processor.

# %%
if RUN_EXTRAS:
    proc_deit = AutoImageProcessor.from_pretrained(EXTRA_ID)
    prep_deit = make_prep(224, proc_deit.image_mean, proc_deit.image_std, 3)
    deit, res = train_model("deit_small", lambda: load_hf_classifier(EXTRA_ID), prep_deit, TRAIN_CFG["deit_small"],
                            head_prefix="hf.classifier")
    MODELS["deit_small"] = {"model": deit, "prep": prep_deit, "result": res, "label": "DeiT-small pré-treinado",
                            "pretrain": "ImageNet-1k (supervisionado, receita DeiT)", "input": "224px, 3 canais, 197 tokens"}
else:
    print("RUN_EXTRAS = False: DeiT-small não treinado.")

# %% [markdown]
# ## 8. Baseline CNN: ResNet-18 pré-treinada (fine-tuning)
# **Por que ResNet-18.** É a CNN de referência mais simples com pré-treino no ImageNet-1k (11,7 M parâmetros, 1,8 GFLOPs), leve na T4, e serve de contraponto de **viés indutivo**: convoluções 3×3 locais com pesos compartilhados (localidade e equivariância à translação) e agregação hierárquica. Mesmo split, mesma augmentation, mesmo loop e mesmas 12 épocas do ViT-B/16; `fc` trocada por `Linear(512 → 6)`; cinza replicado em 3 canais com a normalização do ImageNet dos pesos (`weights.transforms()`). A ResNet-50 foi descartada por custar ~2,3× mais FLOPs sem necessidade para 6 classes de textura.

# %%
RESNET_WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
rt = RESNET_WEIGHTS.transforms()
prep_resnet = make_prep(224, list(rt.mean), list(rt.std), 3)

def build_resnet():
    m = resnet18(weights=RESNET_WEIGHTS)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m

resnet, res = train_model("resnet18", build_resnet, prep_resnet, TRAIN_CFG["resnet18"], head_prefix="fc.")
MODELS["resnet18"] = {"model": resnet, "prep": prep_resnet, "result": res, "label": "ResNet-18 pré-treinada",
                      "pretrain": "ImageNet-1k (supervisionado)", "input": "224px, 3 canais"}

# %% [markdown]
# ## 9. Avaliação no teste e comparação — **Rubricas 3.3, 3.5 e 3.6**
# Cada modelo é avaliado **uma única vez** no teste (270 imagens, 45 por classe), com os pesos da melhor época de validação. Além de accuracy e macro-F1: parâmetros, tempo de treino, tempo por época, VRAM de pico no treino e latência de inferência (ms por imagem, batch 64, fp16).

# %%
@torch.no_grad()
def inference_ms_per_img(model, prep, batch_size=64, reps=10):
    model.eval()
    x = prep(X_all[test_idx[:batch_size]], False)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
        for _ in range(3):
            model(x)
        sync(); t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        sync()
    return 1000 * (time.perf_counter() - t0) / (reps * len(x))

rows = []
for name, m in MODELS.items():
    te = evaluate(m["model"], m["prep"], test_idx)
    m["test"] = te
    r = m["result"]
    n_params = sum(p.numel() for p in m["model"].parameters())
    hist = pd.DataFrame(r["history"])
    rows.append({"modelo": m["label"], "pré-treino": m["pretrain"], "entrada": m["input"],
                 "params (M)": round(n_params / 1e6, 2), "épocas": r["epochs"], "melhor época": r["best_epoch"],
                 "tempo treino (s)": round(r["train_time_s"], 1), "s/época": round(hist.epoch_time_s.mean(), 2),
                 "VRAM pico treino (MB)": None if r["peak_vram_mb"] is None else round(r["peak_vram_mb"]),
                 "inferência (ms/img)": round(inference_ms_per_img(m["model"], m["prep"]), 3),
                 "val macro-F1": round(r["best_val_f1"], 4), "test acc (%)": round(te["acc"], 2),
                 "test macro-F1": round(te["f1"], 4)})
comparison_df = pd.DataFrame(rows)
comparison_df.to_csv(OUT_DIR / "A1_comparison.csv", index=False)
print(comparison_df.to_string(index=False))

for name, m in MODELS.items():
    print(f"\n=== {m['label']} ===")
    print(classification_report(m["test"]["y_true"], m["test"]["y_pred"], target_names=CLASS_NAMES, digits=4, zero_division=0))
    pd.DataFrame({"path": test_df.path.values, "cls": test_df.cls.values,
                  "pred": [CLASS_NAMES[i] for i in m["test"]["y_pred"]],
                  "conf": m["test"]["probs"].max(1).round(4)}).to_csv(OUT_DIR / f"A1_test_predictions_{name}.csv", index=False)

# %% [markdown]
# ### 9.1 Curvas de treino

# %%
# fig: training_curves
names = list(MODELS)
fig, axes = plt.subplots(2, len(names), figsize=(5.2 * len(names), 7.5), squeeze=False)
for c, name in enumerate(names):
    h = pd.DataFrame(MODELS[name]["result"]["history"])
    best_ep = MODELS[name]["result"]["best_epoch"]
    ax = axes[0, c]
    ax.plot(h.epoch, h.train_loss, color=C_BLUE, label="train (CE + label smoothing)")
    ax.plot(h.epoch, h.val_loss, color=C_CYAN, label="val (CE)")
    ax.set_title(f"{MODELS[name]['label']}: loss", fontweight="bold"); ax.set_xlabel("época"); ax.legend(fontsize=8)
    ax = axes[1, c]
    ax.plot(h.epoch, h.train_acc, color=C_BLUE, label="train acc")
    ax.plot(h.epoch, h.val_acc, color=C_CYAN, label="val acc")
    ax.plot(h.epoch, 100 * h.val_f1, "--", color=C_ORANGE, label="val macro-F1 ×100")
    ax.set_title(f"{MODELS[name]['label']}: accuracy", fontweight="bold"); ax.set_xlabel("época"); ax.legend(fontsize=8)
    for ax in axes[:, c]:
        ax.axvline(best_ep, color=C_RED, linestyle=":", label="melhor época")
plt.suptitle("Curvas de treino (mesmo split, mesmo loop)", fontweight="bold")
plt.tight_layout()
savefig("training_curves")
plt.show()

# %% [markdown]
# ### 9.2 Matrizes de confusão e F1 por classe

# %%
# fig: confusion_matrices
fig, axes = plt.subplots(1, len(names), figsize=(5.3 * len(names), 4.8), squeeze=False)
for ax, name in zip(axes[0], names):
    te = MODELS[name]["test"]
    cm = confusion_matrix(te["y_true"], te["y_pred"], labels=range(NUM_CLASSES))
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES, fontsize=8)
    ax.set_xlabel("predito"); ax.set_ylabel("real"); ax.grid(False)
    ax.set_title(f"{MODELS[name]['label']}\nacc {te['acc']:.1f}% | F1 {te['f1']:.3f}", fontweight="bold", fontsize=10)
plt.suptitle("Matrizes de confusão no teste (cor = recall da linha; números = contagens)", fontweight="bold")
plt.tight_layout()
savefig("confusion_matrices")
plt.show()

for name in names:
    te = MODELS[name]["test"]
    cm = confusion_matrix(te["y_true"], te["y_pred"], labels=range(NUM_CLASSES)); np.fill_diagonal(cm, 0)
    pairs = sorted(((cm[i, j], CLASS_NAMES[i], CLASS_NAMES[j]) for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if cm[i, j]), reverse=True)
    print(f"{MODELS[name]['label']}: confusões mais frequentes (real -> predito):", [(f"{a}->{b}", int(n)) for n, a, b in pairs[:4]])

# %%
# fig: per_class_f1
f1_table = pd.DataFrame({MODELS[n]["label"]: precision_recall_fscore_support(
    MODELS[n]["test"]["y_true"], MODELS[n]["test"]["y_pred"], labels=range(NUM_CLASSES), zero_division=0)[2] for n in names},
    index=CLASS_NAMES)
f1_table.round(4).to_csv(OUT_DIR / "A1_per_class_f1.csv")
print(f1_table.round(4).to_string())
ax = f1_table.plot.bar(figsize=(11, 4.2), color=[C_ORANGE, C_BLUE, C_GREEN, C_CYAN][:len(names)], width=0.8)
ax.set_ylim(max(0, f1_table.values.min() - 0.1), 1.01); ax.set_ylabel("F1 no teste")
ax.set_xticklabels(CLASS_NAMES, rotation=0); ax.legend(fontsize=9, loc="lower right")
ax.set_title("F1 por classe no teste", fontweight="bold")
plt.tight_layout()
savefig("per_class_f1")
plt.show()

# %% [markdown]
# **Análise da comparação**
# <!-- ANALISE: tabela: gap de accuracy/F1 entre ViT do zero e ViT-B/16 pré-treinado (esperado: vários pontos percentuais); posição da ResNet-18 (esperado: próxima do ViT-B/16 com ~7× menos parâmetros, menos VRAM e menor latência); custo por ponto de F1; em quais classes o ViT do zero erra mais (hipótese: pares de textura difusa crazing↔rolled-in_scale e inclusion↔pitted_surface, que se diferenciam por detalhes finos e locais) e se os modelos pré-treinados eliminam esses erros; tempo e VRAM (o ViT do zero é o mais barato por época, mas precisa de 150 épocas; o ViT-B/16 é o mais caro em VRAM). Lembrar que o teste tem 45 imagens/classe: 1 erro ≈ 2,2 p.p. na accuracy da classe; diferenças < ~1 p.p. na accuracy global (≈ 3 imagens) não são conclusivas. -->

# %% [markdown]
# ## 10. Mapas de atenção — **Rubricas 2.2 e 3.2**
# Dois métodos, nos dois ViTs:
# - **Uma head, última camada:** a linha do CLS na matriz de atenção, `attn[-1][:, head, 0, 1:]`, com 196 valores → grade 14×14 → interpolação bicúbica para 224 e sobreposição à imagem (padrão da Aula 5/DINO). A head exibida é escolhida **na validação** (não no teste): a de menor entropia média da distribuição CLS→patches, ou seja, a mais focada.
# - **Attention rollout** (Abnar & Zuidema, 2020; Aula 4): em cada camada, média das heads, $\tilde A=\tfrac12(A+I)$ renormalizada por linha (a identidade representa a conexão residual) e produto $R=\tilde A_L\cdots\tilde A_1$; a linha do CLS de $R$ estima quanto cada patch de entrada contribuiu para o CLS final, atravessando todas as camadas.
#
# Além dos mapas, a **distância média de atenção** (Dosovitskiy et al., Fig. 7) mede, para cada head, a distância em pixels entre a query e os patches que ela atende, ponderada pelos pesos: um valor pequeno indica comportamento "convolucional" (local) e um grande indica agregação global.

# %%
@torch.no_grad()
def get_attentions(model, prep, idx):
    model.eval()
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
        logits, attns = model(prep(X_all[idx], False), return_attn=True)
    return logits.float(), [a.float() for a in attns]

def attention_rollout(attns):
    """attns: lista de L tensores [B, h, T, T] -> [B, T-1] (CLS -> patches acumulado em todas as camadas)."""
    result = None
    for a in attns:
        a = a.mean(1)                                                      # [B, T, T]: média das heads
        a = 0.5 * a + 0.5 * torch.eye(a.size(-1), device=a.device)         # conexão residual
        a = a / a.sum(-1, keepdim=True)
        result = a if result is None else torch.bmm(a, result)
    return result[:, 0, 1:]

def patch_distance_matrix(grid, px):
    ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    coords = torch.stack([ys.flatten(), xs.flatten()], 1).float() * px
    return torch.cdist(coords, coords).to(device)                          # [N, N] em pixels

@torch.no_grad()
def attention_stats(model, prep, idx, batch_size=32):
    """Por camada e head: distância média de atenção (patch->patch, px na escala 224) e entropia do CLS->patches."""
    dist_sum, ent_sum, n = None, None, 0
    for i in range(0, len(idx), batch_size):
        _, attns = get_attentions(model, prep, idx[i:i + batch_size])
        T = attns[0].size(-1); grid = int(round(math.sqrt(T - 1)))
        Dm = patch_distance_matrix(grid, 224 / grid)
        d_l, e_l = [], []
        for a in attns:
            pp = a[:, :, 1:, 1:]; pp = pp / pp.sum(-1, keepdim=True)       # [B, h, N, N]
            d_l.append((pp * Dm).sum(-1).mean(-1).sum(0))                  # [h]
            c = a[:, :, 0, 1:]; c = c / c.sum(-1, keepdim=True)            # [B, h, N]
            e_l.append(-(c * (c + 1e-12).log()).sum(-1).sum(0))            # [h]
        d_l, e_l = torch.stack(d_l), torch.stack(e_l)                      # [L, h]
        dist_sum = d_l if dist_sum is None else dist_sum + d_l
        ent_sum = e_l if ent_sum is None else ent_sum + e_l
        n += attns[0].size(0)
    return (dist_sum / n).cpu().numpy(), (ent_sum / n).cpu().numpy(), grid

VIT_NAMES = [n for n in MODELS if n != "resnet18"]
ATTN = {}
for name in VIT_NAMES:
    dist, ent, grid = attention_stats(MODELS[name]["model"], MODELS[name]["prep"], val_idx)
    head = int(np.argmin(ent[-1]))
    ATTN[name] = {"dist": dist, "ent": ent, "grid": grid, "head": head}
    print(f"{MODELS[name]['label']}: {dist.shape[0]} camadas × {dist.shape[1]} heads | head escolhida na última camada: {head} "
          f"(entropia CLS→patches {ent[-1, head]:.2f} nats; uniforme = {math.log(grid * grid):.2f})")
    print("   distância média de atenção por camada (px, média das heads):", np.round(dist.mean(1), 1).tolist())

# %%
def to_map(v, grid, size=224):
    m = F.interpolate(v.view(1, 1, grid, grid), size=(size, size), mode="bicubic", align_corners=False)[0, 0]
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m.cpu().numpy()

def display_images(idx):
    return (F.interpolate(X_all[idx].float(), size=(224, 224), mode="bilinear", align_corners=False)[:, 0] / 255).cpu().numpy()

def overlay(ax, img, amap, cmap, title=None, color="black"):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.imshow(amap, cmap=cmap, alpha=0.55)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    if title:
        ax.set_title(title, fontsize=8, fontweight="bold", color=color)

N_EX = 2   # exemplos por classe: os primeiros do teste de cada classe (escolha fixa, não filtrada por acerto)
test_pos = {k: np.flatnonzero(test_df.label.to_numpy() == k)[:N_EX] for k in range(NUM_CLASSES)}
ex_pos = np.concatenate([test_pos[k] for k in range(NUM_CLASSES)])
ex_idx = test_idx[torch.tensor(ex_pos, device=device)]
ex_imgs = display_images(ex_idx)

MAPS = {}
for name in VIT_NAMES:
    logits, attns = get_attentions(MODELS[name]["model"], MODELS[name]["prep"], ex_idx)
    g, h = ATTN[name]["grid"], ATTN[name]["head"]
    MAPS[name] = {"probs": logits.softmax(1).cpu().numpy(),
                  "head_raw": attns[-1][:, h, 0, 1:].view(-1, g, g).cpu().numpy(),
                  "head": [to_map(attns[-1][i, h, 0, 1:], g) for i in range(len(ex_pos))],
                  "all_heads": [[to_map(attns[-1][i, hh, 0, 1:], g) for hh in range(attns[-1].size(1))] for i in range(len(ex_pos))],
                  "rollout": [to_map(r, g) for r in attention_rollout(attns)]}

# %% [markdown]
# ### 10.1 Heatmap de atenção em um exemplo (1 head, valores brutos)
# O primeiro exemplo de *scratches* do teste: imagem, heatmap bruto 14×14 da head escolhida (pesos CLS→patch, com escala de cores), sobreposição da head e sobreposição do rollout, para os dois ViTs.

# %%
# fig: attention_single_example
k_scr = CLASS_TO_IDX["scratches"] * N_EX
fig, axes = plt.subplots(len(VIT_NAMES), 4, figsize=(15, 3.9 * len(VIT_NAMES)), squeeze=False)
for r, name in enumerate(VIT_NAMES):
    M = MAPS[name]
    pred = int(M["probs"][k_scr].argmax())
    show01(axes[r, 0], ex_imgs[k_scr], f"{MODELS[name]['label']}\npred: {CLASS_NAMES[pred]} ({100 * M['probs'][k_scr].max():.0f}%)")
    im = axes[r, 1].imshow(M["head_raw"][k_scr], cmap="magma")
    axes[r, 1].set_title(f"head {ATTN[name]['head']} (última camada): pesos CLS→patch", fontsize=9, fontweight="bold")
    axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([]); axes[r, 1].grid(False)
    fig.colorbar(im, ax=axes[r, 1], fraction=0.046)
    overlay(axes[r, 2], ex_imgs[k_scr], M["head"][k_scr], "inferno", "overlay: 1 head")
    overlay(axes[r, 3], ex_imgs[k_scr], M["rollout"][k_scr], "jet", "overlay: attention rollout")
plt.suptitle("Atenção do CLS em um exemplo de scratches (teste)", fontweight="bold")
plt.tight_layout()
savefig("attention_single_example")
plt.show()

# %% [markdown]
# ### 10.2 Mapas por classe: 1 head e rollout, nos dois ViTs
# Dois exemplos fixos por classe (os primeiros do teste, sem filtrar por acerto). Título verde = acerto, vermelho = erro.

# %%
def attention_grid(name, fig_name):
    M = MAPS[name]
    fig, axes = plt.subplots(NUM_CLASSES, 3 * N_EX, figsize=(2.15 * 3 * N_EX, 2.25 * NUM_CLASSES))
    for k in range(NUM_CLASSES):
        for e in range(N_EX):
            i = k * N_EX + e
            pred, conf = int(M["probs"][i].argmax()), 100 * M["probs"][i].max()
            color = C_GREEN if pred == k else C_RED
            show01(axes[k, 3 * e], ex_imgs[i])
            axes[k, 3 * e].set_title(f"{CLASS_NAMES[k]}\npred: {CLASS_NAMES[pred]} ({conf:.0f}%)", fontsize=8, fontweight="bold", color=color)
            overlay(axes[k, 3 * e + 1], ex_imgs[i], M["head"][i], "inferno", f"head {ATTN[name]['head']}" if k == 0 else None)
            overlay(axes[k, 3 * e + 2], ex_imgs[i], M["rollout"][i], "jet", "rollout" if k == 0 else None)
    plt.suptitle(f"{MODELS[name]['label']}: atenção CLS→patches (1 head da última camada e rollout)", fontweight="bold")
    plt.tight_layout()
    savefig(fig_name)
    plt.show()

# %%
# fig: attention_maps_vit_scratch
attention_grid("vit_scratch", "attention_maps_vit_scratch")

# %%
# fig: attention_maps_vit_pretrained
attention_grid("vit_pretrained", "attention_maps_vit_pretrained")

# %% [markdown]
# ### 10.3 Especialização das heads (última camada) e distância de atenção por camada

# %%
def heads_grid(name, fig_name):
    M, n_heads = MAPS[name], len(MAPS[name]["all_heads"][0])
    fig, axes = plt.subplots(NUM_CLASSES, 1 + n_heads, figsize=(1.45 * (1 + n_heads), 1.55 * NUM_CLASSES))
    for k in range(NUM_CLASSES):
        i = k * N_EX
        show01(axes[k, 0], ex_imgs[i])
        axes[k, 0].set_ylabel(CLASS_NAMES[k], fontsize=8, fontweight="bold")
        for hh in range(n_heads):
            overlay(axes[k, 1 + hh], ex_imgs[i], M["all_heads"][i][hh], "inferno", f"head {hh}" if k == 0 else None)
    plt.suptitle(f"{MODELS[name]['label']}: CLS→patches por head (última camada)", fontweight="bold")
    plt.tight_layout()
    savefig(fig_name)
    plt.show()

# %%
# fig: attention_heads_vit_scratch
heads_grid("vit_scratch", "attention_heads_vit_scratch")

# %%
# fig: attention_heads_vit_pretrained
heads_grid("vit_pretrained", "attention_heads_vit_pretrained")

# %%
# fig: attention_distance
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for name, color in zip(VIT_NAMES, [C_ORANGE, C_BLUE, C_CYAN]):
    dist, ent = ATTN[name]["dist"], ATTN[name]["ent"]
    L = np.arange(1, dist.shape[0] + 1)
    for l in range(dist.shape[0]):
        axes[0].scatter(np.full(dist.shape[1], L[l]), dist[l], s=14, color=color, alpha=0.6)
        axes[1].scatter(np.full(ent.shape[1], L[l]), ent[l], s=14, color=color, alpha=0.6)
    axes[0].plot(L, dist.mean(1), "-", color=color, label=MODELS[name]["label"])
    axes[1].plot(L, ent.mean(1), "-", color=color, label=MODELS[name]["label"])
axes[0].set_title("Distância média de atenção (px, escala 224)", fontweight="bold"); axes[0].set_xlabel("camada")
axes[1].axhline(math.log(196), color=C_RED, linestyle="--", label="uniforme (ln 196)")
axes[1].set_title("Entropia da atenção CLS→patches (nats)", fontweight="bold"); axes[1].set_xlabel("camada")
for ax in axes:
    ax.legend(fontsize=8)
plt.suptitle("Cada ponto é uma head (média na validação); linha = média das heads", fontweight="bold")
plt.tight_layout()
savefig("attention_distance")
plt.show()

# %% [markdown]
# ### 10.4 Dependência da organização espacial: teste com patches embaralhados
# Os três modelos treinados recebem as imagens de teste com os 196 patches 16×16 embaralhados (a mesma permutação fixa para todas). Se a classe fosse decidida só pela "estatística de textura" dos patches, a accuracy quase não cairia; a queda mede quanto cada modelo usa a posição relativa dos patches (PE nos ViTs, campos receptivos que atravessam as bordas dos patches na CNN).

# %%
SHUFFLE_PATCH = 16
shuffle_rows = []
for name, m in MODELS.items():
    size = SCRATCH_CFG["img_size"] if name == "vit_scratch" else 224
    n_p = (size // SHUFFLE_PATCH) ** 2          # 196 em 224 px; mesma semente -> mesma permutação para todos os modelos
    perm_m = torch.randperm(n_p, generator=torch.Generator().manual_seed(SEED + 1))
    te_s = evaluate(m["model"], m["prep"], test_idx, transform=lambda x, p=perm_m: shuffle_patches(x, p, SHUFFLE_PATCH))
    m["test_shuffled"] = te_s
    shuffle_rows.append({"modelo": m["label"], "test acc (%)": round(m["test"]["acc"], 2),
                         "acc patches embaralhados (%)": round(te_s["acc"], 2),
                         "queda (p.p.)": round(m["test"]["acc"] - te_s["acc"], 2),
                         "macro-F1 embaralhado": round(te_s["f1"], 4)})
shuffle_df = pd.DataFrame(shuffle_rows)
shuffle_df.to_csv(OUT_DIR / "A1_patch_shuffle.csv", index=False)
print(shuffle_df.to_string(index=False))
comparison_df["acc patches embaralhados (%)"] = shuffle_df["acc patches embaralhados (%)"].values
comparison_df.to_csv(OUT_DIR / "A1_comparison.csv", index=False)

# %%
# fig: patch_shuffle_robustness
fig, ax = plt.subplots(figsize=(8, 3.8))
xs = np.arange(len(shuffle_df))
ax.bar(xs - 0.2, shuffle_df["test acc (%)"], 0.4, color=C_BLUE, label="teste original")
ax.bar(xs + 0.2, shuffle_df["acc patches embaralhados (%)"], 0.4, color=C_ORANGE, label="patches embaralhados")
for x_, a, b in zip(xs, shuffle_df["test acc (%)"], shuffle_df["acc patches embaralhados (%)"]):
    ax.text(x_ - 0.2, a + 1, f"{a:.1f}", ha="center", fontsize=8); ax.text(x_ + 0.2, b + 1, f"{b:.1f}", ha="center", fontsize=8)
ax.set_xticks(xs, shuffle_df.modelo); ax.set_ylim(0, 108); ax.set_ylabel("accuracy no teste (%)")
ax.axhline(100 / NUM_CLASSES, color=C_RED, linestyle=":", label="acaso (16,7%)"); ax.legend(fontsize=8, loc="lower right")
ax.set_title("Accuracy com os patches 16×16 embaralhados", fontweight="bold")
plt.tight_layout()
savefig("patch_shuffle_robustness")
plt.show()

# %% [markdown]
# **Interpretação da atenção (rascunho técnico, a refinar com as figuras)** — **Rubricas 2.2 e 3.2**
#
# O que se espera ver e como ler as figuras:
# - **scratches:** a atenção deve se alinhar à linha do risco (faixa estreita e contínua de patches), porque o risco é o único elemento que diferencia a chapa do fundo; um mapa que acompanha a linha indica que o modelo encontrou o defeito, e não uma correlação espúria de iluminação.
# - **inclusion e pitted_surface:** pontos escuros esparsos; o esperado é atenção em "manchas" sobre os pontos, com o rollout mais difuso. A confusão entre as duas classes, quando existe, deve aparecer em imagens com poucos pites grandes.
# - **patches:** regiões grandes de brilho diferente; a atenção tende a marcar as **bordas** da mancha (transição de contraste), não o seu interior homogêneo.
# - **crazing e rolled-in_scale:** o defeito cobre a imagem toda (textura estacionária). Não há "objeto" a localizar e o mapa esperado é difuso ou em padrão espalhado. Isso contraria a narrativa da Aula 4 (atenção concentrada no objeto) e é coerente com o domínio: para uma textura, qualquer amostra de patches carrega a mesma informação.
# - **Do zero vs pré-treinado:** o ViT do zero, com poucos dados, tende a ter heads de última camada pouco especializadas (entropia perto da uniforme) e mapas ruidosos; o pré-treinado costuma ter heads mais focadas e diversificadas (figura 10.3). Na distância de atenção, o ViT pré-treinado costuma mostrar heads locais e globais nas primeiras camadas e atenção global no fim (Dosovitskiy et al., Fig. 7); se o ViT do zero não desenvolver heads locais nas primeiras camadas, é a evidência direta da falta de viés de localidade, que a CNN tem por construção.
# - **Limites da leitura:** atenção não é explicação causal (o rollout ignora o MLP e os valores V); mapas de uma head mostram *para onde o CLS olha*, não *por que* a classe foi escolhida. As observações devem ser confirmadas por padrões recorrentes em vários exemplos por classe, como nas grades 10.2, e não por uma imagem isolada.
#
# <!-- ANALISE: descrever, classe a classe, o que as figuras 10.1–10.3 mostram de fato para os dois ViTs (onde a head escolhida e o rollout concentram peso; se acompanham o risco em scratches; pontos em inclusion/pitted; bordas em patches; difuso em crazing/rolled-in); comparar a entropia e a distância de atenção do zero vs pré-treinado com os números da célula de estatísticas; citar erros (títulos vermelhos) e se a atenção explica o erro; usar a queda no teste embaralhado (10.4) para dizer quanto cada modelo depende da organização espacial vs estatística de textura — queda pequena confirma que o NEU é quase um problema de textura "bag of patches". -->

# %% [markdown]
# ## 11. Discussão
#
# ### 11.1 Pré-treino do BERT vs pré-treino do ViT: o que cada um maximiza — **Rubrica 2.5**
# **BERT (auto-supervisionado, texto).** Encoder bidirecional treinado em BookCorpus + Wikipedia (~3,3 B palavras) sem rótulos humanos, com dois objetivos: (i) **MLM**: 15% dos tokens são selecionados (80% viram `[MASK]`, 10% um token aleatório, 10% ficam iguais) e a perda é a entropia cruzada só nessas posições; o modelo **maximiza a verossimilhança do token original dado o contexto bidirecional**, $\max \sum_{i\in\mathcal{M}}\log p(x_i\mid x_{\setminus\mathcal{M}})$, e por isso aprende representações contextuais de cada token; (ii) **NSP**: classificação binária no `[CLS]` (a sentença B segue a A?), que maximiza a coerência entre sentenças no CLS; $\mathcal{L}=\mathcal{L}_{MLM}+\mathcal{L}_{NSP}$. O RoBERTa mostrou depois que o NSP pouco ajuda. A supervisão vem **dos próprios dados**, o que permite usar corpora enormes.
#
# **ViT original (supervisionado, imagens).** Pré-treino por **classificação** no ImageNet-21k (14 M imagens, 21.843 classes) ou no JFT-300M, e depois fine-tuning com head novo — exatamente o checkpoint usado na seção 7. O objetivo **maximiza a verossimilhança do rótulo da imagem inteira**, $\max\log p(y\mid x)$, lida no CLS: as features ficam discriminativas e alinhadas às categorias do pré-treino, e as informações irrelevantes para o rótulo (textura de fundo, posição exata) podem ser descartadas. Depende de rótulos humanos em grande escala.
#
# **Diferenças que importam.** (1) Fonte do sinal: o BERT tira o sinal da estrutura do próprio texto; o ViT original depende de rótulos. (2) Granularidade: o MLM obriga cada token a carregar informação local reconstruível; a classificação só exige que o CLS separe classes. (3) Redundância: uma palavra mascarada é difícil de adivinhar, enquanto um patch mascarado é fácil de interpolar pelos vizinhos, por isso o análogo visual direto do MLM, o **MAE**, mascara **75%** dos patches e reconstrói pixels (MSE) com um decoder leve. O **DINO**, outra alternativa auto-supervisionada, não reconstrói nada: maximiza a **concordância** entre visões (recortes locais e globais) da mesma imagem via auto-destilação aluno/professor EMA, e suas heads aprendem a segmentar objetos sem rótulos.
#
# **Relevância para o NEU.** O ViT-B/16 in21k traz features de objetos naturais; parte delas (bordas, texturas, contraste) transfere para aço. Com muitas imagens de chapas **sem rótulo** (o que é comum na indústria: a câmera grava tudo, o rótulo é caro), um pré-treino auto-supervisionado **no domínio** (MAE ou DINO) seria o caminho natural para um ViT, reduzindo a dependência do ImageNet.
#
# <!-- ANALISE: amarrar com o resultado: o ganho do pré-treinado sobre o do zero (tabela 9) quantifica o valor de pré-treino, mesmo vindo de um domínio distante. -->
#
# ### 11.2 DeiT e Swin: o que resolvem que o ViT original não resolve — **Rubrica 3.4**
# **DeiT (Touvron et al., 2021): eficiência de dados.** O ViT original só supera CNNs quando pré-treinado em JFT-300M; treinado só no ImageNet-1k, fica abaixo de ResNets comparáveis. O DeiT torna o ViT competitivo **só com ImageNet-1k** e 8 GPUs em ~3 dias, com duas contribuições: (i) uma **receita de treino** com regularização e augmentation fortes (RandAugment, Mixup, CutMix, Random Erasing, repeated augmentation, stochastic depth, AdamW com warmup e cosseno) que compensa a falta de viés indutivo; (ii) o **token de destilação**: um token aprendível extra, ao lado do CLS, com head própria, treinado para imitar a predição de um **professor CNN** congelado (RegNetY-16GF). Na destilação *hard*, $\mathcal{L}=\tfrac12\mathrm{CE}(z_{cls},y)+\tfrac12\mathrm{CE}(z_{dist},\arg\max z_{teacher})$; na inferência, média das duas heads, sem a CNN. O professor convolucional transfere, via rótulos, o **viés de localidade** que o ViT não tem. Resolve: dependência de datasets gigantes e custo do pré-treino.
#
# **Swin Transformer (Liu et al., 2021): escala e custo.** O ViT tem dois problemas estruturais: (a) atenção global com custo **quadrático** no número de tokens, $\Omega(\text{MSA})=4hwC^2+2(hw)^2C$, inviável em alta resolução; (b) arquitetura **isotrópica**, com um único mapa de 1/16 da resolução, ruim para tarefas densas (detecção, segmentação) que precisam de múltiplas escalas. O Swin (i) calcula a atenção **dentro de janelas locais** M×M (M = 7), $\Omega(\text{W-MSA})=4hwC^2+2M^2hwC$, **linear** em $hw$; (ii) alterna janelas regulares e **deslocadas** (*shifted windows*, deslocamento de ⌊M/2⌋, implementado com *cyclic shift* e máscara) para que a informação atravesse as fronteiras das janelas; (iii) monta uma **hierarquia** com *patch merging* (patches 4×4 → mapas de 1/4, 1/8, 1/16, 1/32), como uma CNN, compatível com FPN. Resolve: custo em alta resolução e ausência de representação multiescala; de quebra, reintroduz localidade como viés.
#
# **No NEU.** O DeiT ataca exatamente o problema observado aqui (ViT do zero com poucos dados). O Swin interessaria se o problema passasse a ser **detecção** dos defeitos (o NEU-DET tem caixas delimitadoras) ou imagens de linha em alta resolução, onde a atenção global do ViT fica cara.
#
# <!-- ANALISE: se RUN_EXTRAS rodou, citar o resultado do DeiT-small na tabela (params, F1, tempo) como evidência. -->
#
# ### 11.3 Quando o ViT supera a CNN e quando a CNN é preferível — **Rubrica 3.5**
# **Viés indutivo vs dados.** A CNN assume por construção **localidade** (kernels 3×3), **compartilhamento de pesos** e **equivariância à translação**, e agrega o contexto gradualmente. Esses pressupostos acertam para imagens naturais e reduzem a quantidade de dados necessária. O ViT quase não tem viés (só a divisão em patches), tem **campo receptivo global desde o primeiro bloco** e aprende quais relações espaciais importam; com dados suficientes, isso dá um teto mais alto e melhor escalabilidade (Dosovitskiy et al.: com ImageNet-1k a ResNet vence, com 21k empatam, com JFT-300M o ViT vence).
#
# **O ViT tende a superar a CNN quando:** há muitos dados ou um bom pré-treino transferível; a tarefa exige relações de **longo alcance** (partes distantes da imagem que precisam ser combinadas); há pré-treino multimodal disponível (CLIP) ou auto-supervisionado (DINO/MAE); a infraestrutura favorece matmuls grandes (GPUs modernas).
#
# **A CNN é preferível quando:** o dataset é pequeno e não há pré-treino adequado; os sinais discriminativos são **locais e de textura**; o orçamento de latência, memória ou energia é apertado (inspeção em linha, edge/NPU); a resolução de entrada é alta e variável.
#
# **Neste domínio.** O NEU é pequeno (1.260 imagens de treino) e de **textura**: os defeitos são padrões locais (linhas, pites, microfissuras) que se repetem ou aparecem em qualquer posição, exatamente o caso em que localidade e equivariância à translação ajudam. A expectativa é que o ViT do zero fique claramente abaixo, e que ViT pré-treinado e ResNet-18 fiquem próximos, com a CNN muito mais barata.
#
# <!-- ANALISE: confirmar ou refutar com a tabela 9 (acc/F1, params, VRAM, latência), com a distância de atenção (10.3: o ViT pré-treinado desenvolveu heads locais nas primeiras camadas, "reaprendendo" a localidade da CNN?) e com o teste de patches embaralhados (10.4). -->
#
# ### 11.4 Escolha de arquitetura para este domínio, com base nos dados — **Rubrica 3.6**
# **Rascunho do critério:** entre os modelos cuja macro-F1 no teste está dentro da incerteza do melhor (270 imagens: ~±1,5 p.p. de intervalo para accuracies perto de 97%), escolher o de **menor custo de inferência e de treino**, porque em uma linha de laminação o modelo roda continuamente, em hardware industrial, com requisitos de latência.
#
# <!-- ANALISE: aplicar o critério à tabela 9 com os números. Cenário mais provável: ResNet-18 ≈ ViT-B/16 em F1, com ~7× menos parâmetros, menos VRAM e menor latência → recomendar a CNN (ou ViT pré-treinado apenas se ganhar claramente em F1 ou em alguma classe crítica); ViT do zero descartado (F1 menor, precisa de muitas épocas). Citar F1 por classe nas classes críticas para a qualidade (ex.: inclusion e scratches, que são defeitos que comprometem a chapa). Se o ViT pré-treinado vencer com margem, justificar o custo extra. -->
#
# ### 11.5 O que eu mudaria (próximos passos)
# 1. **Pré-treino auto-supervisionado no domínio** (MAE ou DINO em imagens de aço sem rótulo, como o GC10-DET ou o Severstal) antes do fine-tuning, que é a resposta direta à fome de dados do ViT sem depender do ImageNet.
# 2. **Regularização do ViT do zero no estilo DeiT:** stochastic depth, RandAugment adaptado a textura, e **destilação** com a ResNet-18 treinada aqui como professora (token de destilação), testando também CutMix apenas entre classes de textura difusa.
# 3. **Arquiteturas híbridas / com viés de localidade:** stem convolucional no ViT (patch embedding com convoluções empilhadas), Swin-T ou ConvNeXt-T, que costumam fechar a distância para a CNN com poucos dados.
# 4. **Avaliação mais robusta:** validação cruzada estratificada (5 folds) com várias seeds, porque 45 imagens por classe no teste deixam intervalos de confiança de alguns p.p.
# 5. **Passar para detecção** (NEU-DET tem caixas): em produção interessa *onde* está o defeito e seu tamanho, não só a classe; mapas de atenção são um indício, não uma localização confiável.
#
# <!-- ANALISE: priorizar os itens com base nos resultados (ex.: se o ViT do zero ficou muito abaixo, o item 1/2 é prioritário; se a CNN empatou com o ViT pré-treinado, o item 3 é o mais barato). -->

# %% [markdown]
# ## 12. Métricas finais (JSON)

# %%
def model_summary(name):
    m, r = MODELS[name], MODELS[name]["result"]
    te, per_f1 = m["test"], precision_recall_fscore_support(
        m["test"]["y_true"], m["test"]["y_pred"], labels=range(NUM_CLASSES), zero_division=0)[2]
    row = comparison_df[comparison_df.modelo == m["label"]].iloc[0]
    return {"label": m["label"], "pretrain": m["pretrain"], "input": m["input"],
            "total_params": int(sum(p.numel() for p in m["model"].parameters())),
            "trainable_params": int(sum(p.numel() for p in m["model"].parameters() if p.requires_grad)),
            "test_accuracy": round(te["acc"], 2), "test_macro_f1": round(te["f1"], 4),
            "test_per_class_f1": {c: round(float(f), 4) for c, f in zip(CLASS_NAMES, per_f1)},
            "test_accuracy_patch_shuffled": round(m["test_shuffled"]["acc"], 2),
            "best_epoch": int(r["best_epoch"]), "epochs": int(r["epochs"]), "best_val_macro_f1": round(r["best_val_f1"], 4),
            "train_time_s": round(r["train_time_s"], 1), "s_per_epoch": float(row["s/época"]),
            "peak_vram_train_mb": None if r["peak_vram_mb"] is None else round(r["peak_vram_mb"]),
            "inference_ms_per_img": float(row["inferência (ms/img)"]), "train_config": r["config"]}

metrics = {
    "activity": "A1",
    "dataset": "kaustubhdikshit/neu-surface-defect-database (NEU-DET train+validation reunidos)",
    "n_images_clean": int(len(df)), "n_corrupted": int(len(corrupted)), "n_duplicates_removed": int(len(df_ok) - len(df)),
    "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    "classes": CLASS_NAMES,
    "models": {name: model_summary(name) for name in MODELS},
    "attention": {name: {"head_shown": ATTN[name]["head"],
                         "mean_attention_distance_px_per_layer": np.round(ATTN[name]["dist"].mean(1), 2).tolist(),
                         "cls_entropy_last_layer_per_head": np.round(ATTN[name]["ent"][-1], 3).tolist()} for name in ATTN},
    "pe_demo": {("with_pe" if k else "without_pe"): {kk: vv for kk, vv in v.items() if not kk.startswith("logits")} for k, v in pe_demo.items()},
    "config": {"scratch_vit": SCRATCH_CFG, "pretrained_id": PRETRAINED_ID, "extras": RUN_EXTRAS,
               "label_smoothing": LABEL_SMOOTHING, "grad_clip": GRAD_CLIP, "min_lr_ratio": MIN_LR_RATIO,
               "aug": f"D4 (rot90 + flip) + brilho/contraste ±{AUG_JITTER:.0%}", "amp_fp16": USE_AMP, "seed": SEED},
    "figures": FIGURES,
    "notebook_time_s": round(time.time() - NOTEBOOK_T0, 1),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
}
(OUT_DIR / "A1_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
print(json.dumps(metrics, indent=2, ensure_ascii=False))
