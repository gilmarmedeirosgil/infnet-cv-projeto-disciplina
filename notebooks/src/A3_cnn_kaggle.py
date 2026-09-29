# %% [markdown]
# # A3 — Classificador de imagens com CNN pré-treinada (feature extraction)
# **Visão Computacional com CNNs e Transformers** · Faculdade Infnet — MBA em Engenharia de IA, Machine Learning e Deep Learning · Gilmar Oliveira de Medeiros
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gilmarmedeirosgil/infnet-cv-projeto-disciplina/blob/main/notebooks/A3_cnn_kaggle.ipynb)
#
# **Objetivo.** Classificar as 7 classes do dataset Kaggle `pavansanagapati/images-dataset` (bike, cars, cats, dogs, flowers, horses, human) com uma CNN pré-treinada no ImageNet usada como **extratora de features**: backbone congelado e apenas um novo head linear treinado.
#
# **Requisitos de execução (Colab, runtime T4)** — valores medidos no "Executar tudo" (Tesla T4, 14,56 GB de VRAM); todos são registrados no JSON da seção 10:
#
# | Recurso | Valor | Observação |
# |---|---|---|
# | RAM | **~1,9 GB** no processo principal (pico `maxrss` 1.947 MB) + **~1,3 GB** no maior worker do DataLoader (1.326 MB) | medido com `resource.getrusage`; as imagens são decodificadas sob demanda. O `maxrss` dos filhos é o do maior worker, e os 2 workers compartilham páginas com o processo principal (*copy-on-write*), então o total real fica entre ~2 e ~4,6 GB (soma no pior caso), folgado para os 12,7 GB do Colab |
# | VRAM (treino real) | **0,7 GB** alocados (723 MB) / **1,3 GB** reservados pelo cache do PyTorch (1.264 MB) | backbone congelado: o autograd não guarda ativações do backbone |
# | VRAM (pico do notebook) | **~5,6 GB** alocados (5.611 MB) / **~6,1 GB** reservados (6.118 MB) | benchmark sintético de *fine-tuning* completo (seção 4), batch 64, fp32; o `nvidia-smi` mostra ainda o contexto CUDA (~0,3–0,5 GB) |
# | Disco | ~1,5 GB (dataset baixado 1,01 GB + extraído) | estimativa; pesos da EfficientNet-B0: 20,5 MB |
# | Tempo total | **~5,7 min** (341 s) | medido da primeira célula (pip) até o JSON final: pip 7 s, download + extração do dataset 76 s, treino de 15 épocas **143 s** (~9,5 s/época); o resto é montagem do Drive, varredura/EDA, pHash, benchmark e avaliação. Com o dataset já em cache, ~4,5 min |

#
# **Mapa da rubrica:** 1.1 → seção 5 · 1.2 → seções 6–7 · 1.3 → seção 8 · 1.4 → seção 9 · 1.5 → seção 4.
#
# Estrutura: Problema → Decisão técnica → Código → Resultado → Análise. Figuras usadas no relatório são salvas em PNG em `MyDrive/infnet_cv_projeto/outputs/A3/`.

# %% [markdown]
# ## 1. Setup
# Seed 42 em `random`, `numpy` e `torch` (padrão da Aula 1), checagem da GPU, montagem do Drive para persistir os artefatos e leitura dos Secrets do Kaggle.

# %%
import time
NOTEBOOK_T0 = time.time()
!pip install -q kagglehub imagehash
PIP_TIME_S = time.time() - NOTEBOOK_T0
print(f"pip: {PIP_TIME_S:.1f}s")

# %%
import os, gc, copy, json, time, random, hashlib, resource
from pathlib import Path

import psutil
import imagehash

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, resnet50, ResNet50_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support, f1_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__} | torchvision {torchvision.__version__} | device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("ATENÇÃO: sem GPU. No Colab: Ambiente de execução > Alterar tipo > T4 GPU.")

C_BLUE, C_CYAN, C_RED, C_GREEN = "#0A345D", "#1BB5D8", "#DC2626", "#15803D"
plt.rcParams.update({"axes.grid": True, "grid.linestyle": ":", "figure.dpi": 110})

def ram_stats():
    """RSS atual e pico (ru_maxrss em KB no Linux) do processo e dos filhos (workers do DataLoader), em MB."""
    return {"ram_rss_mb": round(psutil.Process().memory_info().rss / 2**20),
            "ram_maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
            "ram_maxrss_children_mb": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024)}
print("RAM:", ram_stats())

# %%
try:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT_DIR = Path("/content/drive/MyDrive/infnet_cv_projeto/outputs/A3")
except Exception as e:
    print(f"Drive indisponível ({e!r}); usando armazenamento local do runtime.")
    OUT_DIR = Path("/content/outputs/A3") if Path("/content").exists() else Path("outputs/A3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("Artefatos em:", OUT_DIR)

def savefig(name, dpi=150, max_kb=1000):
    """Salva o PNG; se passar de max_kb, re-salva com dpi menor (determinístico)."""
    path = OUT_DIR / f"A3_{name}.png"
    for d in sorted({dpi, 100, 80, 60} - {x for x in (100, 80, 60) if x > dpi}, reverse=True):
        plt.savefig(path, dpi=d, bbox_inches="tight")
        if path.stat().st_size / 1024 <= max_kb:
            break
    print(f"{path.name}: {path.stat().st_size / 1024:.0f} KB (dpi {d})")

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
t0 = time.time()
DATASET_PATH = Path(kagglehub.dataset_download("pavansanagapati/images-dataset"))
DOWNLOAD_TIME_S = time.time() - t0
print(f"Dataset em: {DATASET_PATH} | download/cache: {DOWNLOAD_TIME_S:.1f}s")

# %% [markdown]
# ## 2. Dados e EDA
# **Problema.** O dataset tem as classes em `data/<classe>` e uma **cópia duplicada em `data/data/`**, que é ignorada (usar as duas geraria vazamento entre treino e teste). Antes do split, cada arquivo é aberto e decodificado por completo para: (i) descartar arquivos corrompidos; (ii) levantar formato, modo de cor (RGB, L, P, RGBA…) e tamanho; (iii) detectar **duplicatas exatas** (hash MD5), que também causariam vazamento. Todas as imagens são convertidas para RGB; as que têm transparência são compostas sobre fundo branco, porque um `convert("RGB")` direto pode deixar o fundo preto.

# %%
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".jfif"}

def find_data_root(base):
    if (base / "data").is_dir():
        return base / "data"
    for d in sorted(base.rglob("*")):
        if d.is_dir() and {"cats", "dogs"} <= {c.name for c in d.iterdir() if c.is_dir()}:
            return d
    raise FileNotFoundError(f"Pastas de classe não encontradas em {base}")

def load_rgb(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("I", "I;16", "F"):
        arr = np.asarray(img, dtype=np.float32)
        img = Image.fromarray((255 * (arr - arr.min()) / max(np.ptp(arr), 1e-6)).astype(np.uint8))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.getchannel("A"))
        return background
    return img.convert("RGB")

DATA_ROOT = find_data_root(DATASET_PATH)
CLASS_NAMES = sorted(d.name for d in DATA_ROOT.iterdir() if d.is_dir() and d.name != "data")
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
print("Raiz:", DATA_ROOT, "| classes:", CLASS_NAMES)
assert NUM_CLASSES == 7, f"Esperadas 7 classes, encontradas {NUM_CLASSES}: {CLASS_NAMES}"

# %%
records, non_images = [], []
for cls in CLASS_NAMES:
    for p in sorted((DATA_ROOT / cls).rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXT:
            non_images.append(str(p.relative_to(DATA_ROOT)))
            continue
        rec = {"path": str(p), "cls": cls, "label": CLASS_TO_IDX[cls], "ext": p.suffix.lower(), "ok": True, "error": ""}
        try:
            rec["md5"] = hashlib.md5(p.read_bytes()).hexdigest()
            with Image.open(p) as im:
                rec.update(format=im.format, mode=im.mode, width=im.size[0], height=im.size[1])
                im.load()
        except Exception as e:
            rec.update(ok=False, error=repr(e))
        records.append(rec)

df_all = pd.DataFrame(records)
corrupted = df_all[~df_all.ok]
df_ok = df_all[df_all.ok].copy()
print(f"Arquivos de imagem: {len(df_all)} | corrompidos: {len(corrupted)} | não-imagem ignorados: {len(non_images)}")
if len(corrupted):
    print(corrupted[["path", "error"]].to_string())
if non_images:
    print("Não-imagem (amostra):", non_images[:10])
print("\nFormato:", df_ok.format.value_counts().to_dict())
print("Modo de cor:", df_ok["mode"].value_counts().to_dict())
print("Extensão:", df_ok.ext.value_counts().to_dict())
print("\nModo de cor por classe:\n", pd.crosstab(df_ok.cls, df_ok["mode"]).to_string())

# %% [markdown]
# **Canal alfa.** A composição sobre fundo branco em `load_rgb` só altera pixels com alfa < 255. Para cada imagem com canal alfa (ou paleta com transparência) mede-se a fração de pixels não opacos; se for zero em todas, a composição não muda nenhum pixel.

# %%
def alpha_fraction(path):
    with Image.open(path) as im:
        if not (im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)):
            return np.nan
        return float((np.asarray(im.convert("RGBA").getchannel("A")) < 255).mean())

df_ok["alpha_frac"] = [alpha_fraction(p) for p in df_ok.path]
with_alpha = df_ok.dropna(subset=["alpha_frac"])
alpha_summary = {cls: {"n_images_with_alpha": int(len(g)), "mean_frac_alpha_lt_255": round(float(g.alpha_frac.mean()), 6),
                       "max_frac_alpha_lt_255": round(float(g.alpha_frac.max()), 6),
                       "n_images_any_transparent": int((g.alpha_frac > 0).sum())}
                 for cls, g in with_alpha.groupby("cls")}
print(f"Imagens com canal alfa: {len(with_alpha)}")
print(pd.DataFrame(alpha_summary).T.to_string() if alpha_summary else "Nenhuma imagem com canal alfa.")

# %% [markdown]
# **Resultado.** As 210 imagens com canal alfa são todas de flowers, e **todas são 100% opacas**: 0 pixels com A < 255 em qualquer uma delas (média e máximo da fração = 0). O canal alfa é só um artefato do formato de gravação (PNG RGBA) e não carrega informação. A composição sobre fundo branco em `load_rgb` não altera nenhum pixel, então a hipótese de atalho "fundo branco/borda de recorte → flowers" está **descartada**. O que continua em aberto é o atalho de **resolução/compressão**: flowers são as únicas imagens 128×128 sem perda (PNG), ampliadas 2× pelo pré-processamento (seção 7).

# %%
md5_classes = df_ok.groupby("md5").cls.nunique()
conflicting = md5_classes[md5_classes > 1].index
n_dup = int(df_ok.duplicated("md5").sum())
df = df_ok[~df_ok.md5.isin(conflicting)].drop_duplicates("md5", keep="first").reset_index(drop=True)
print(f"Duplicatas exatas removidas: {n_dup} (hashes em mais de uma classe, removidos por completo: {len(conflicting)})")
print(f"Imagens válidas e únicas: {len(df)}")

counts = df.cls.value_counts().reindex(CLASS_NAMES)
imbalance_ratio = counts.max() / counts.min()
print(pd.DataFrame({"n": counts, "%": (100 * counts / counts.sum()).round(1)}).to_string())
print(f"Razão de desbalanceamento (maior/menor): {imbalance_ratio:.2f}")

def value_counts_dict(s):
    return {str(k): int(v) for k, v in s.value_counts().items()}

eda_summary = {
    "formats": value_counts_dict(df.format),
    "modes": value_counts_dict(df["mode"]),
    "per_class": {cls: {"n": int(len(g)), "formats": value_counts_dict(g.format), "modes": value_counts_dict(g["mode"]),
                        "median_width": float(g.width.median()), "median_height": float(g.height.median())}
                  for cls, g in df.groupby("cls")},
}
print(json.dumps(eda_summary["per_class"], indent=1))

# %%
# fig: class_counts
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(CLASS_NAMES, counts.values, color=C_BLUE)
ax.bar_label(bars)
ax.axhline(counts.mean(), color=C_RED, linestyle="--", label=f"média = {counts.mean():.0f}")
ax.set_ylabel("nº de imagens")
ax.set_title("Imagens por classe (após limpeza)", fontweight="bold")
ax.legend()
savefig("class_counts")
plt.show()

# %%
# fig: samples_grid
N_PER_CLASS = 6
rng = np.random.default_rng(SEED)
fig, axes = plt.subplots(NUM_CLASSES, N_PER_CLASS, figsize=(1.6 * N_PER_CLASS, 1.6 * NUM_CLASSES), dpi=80)
for r, cls in enumerate(CLASS_NAMES):
    paths = df[df.cls == cls].path.values
    for c, p in enumerate(rng.choice(paths, N_PER_CLASS, replace=False)):
        ax = axes[r, c]
        ax.imshow(load_rgb(p))
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if c == 0:
            ax.set_ylabel(cls, fontsize=12, fontweight="bold")
plt.suptitle("Amostras aleatórias por classe", fontweight="bold")
plt.tight_layout()
savefig("samples_grid", dpi=100)
plt.show()

# %%
# fig: image_sizes
df["aspect"] = df.width / df.height
print(df.groupby("cls")[["width", "height", "aspect"]].median().round(2).to_string())
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for cls in CLASS_NAMES:
    sub = df[df.cls == cls]
    axes[0].scatter(sub.width, sub.height, s=8, alpha=0.6, label=cls)
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_xlabel("largura (px)"); axes[0].set_ylabel("altura (px)")
axes[0].set_title("Tamanho original das imagens", fontweight="bold")
axes[0].legend(fontsize=8, markerscale=2)
axes[1].hist(df.aspect.clip(0.3, 3), bins=40, color=C_CYAN)
axes[1].axvline(1, color=C_RED, linestyle="--")
axes[1].set_xlabel("razão de aspecto (largura/altura)")
axes[1].set_title("Razão de aspecto", fontweight="bold")
savefig("image_sizes")
plt.show()

# %% [markdown]
# **Análise da EDA**
#
# 1. **Integridade.** 1.803 arquivos, **0 corrompidos** (todos decodificados por completo com `im.load()`), nenhum arquivo não-imagem. Foram removidas **39 duplicatas exatas** (MD5), nenhuma com rótulos conflitantes; as perdas ficaram em horses (−19), human (−19) e bike (−1). Restam **1.764 imagens únicas**. Sem essa etapa, cópias da mesma imagem poderiam cair em treino e teste e inflar a accuracy.
# 2. **Formato, cor e tamanho estão amarrados à classe.** Essa é a observação mais importante da EDA:
#
#    | Grupo de classes | Formato | Modo | Tamanho típico (mediana) |
#    |---|---|---|---|
#    | bike, cars | BMP (785 = bike 365 + cars 420, antes da deduplicação) | RGB | 640×480, todas iguais |
#    | flowers | **PNG, 210/210** | **RGBA, 210/210** | **128×128, todas iguais** |
#    | cats, dogs, horses, human | JPEG (808) | RGB | variável (≈190–460 px; human em retrato, razão 0,73) |
#
#    Nenhuma imagem está em tons de cinza no modo `L` e nenhuma outra classe tem canal alfa, então só as 210 flowers passaram pela composição sobre fundo branco em `load_rgb`. Na grade de amostras as flowers aparecem com fundo natural (folhas, céu), e a célula "Canal alfa" acima confirma: **0% de pixels com A < 255** nas 210 imagens, então a composição não muda nenhum pixel. O formato em si não chega ao modelo, porque tudo vira tensor RGB, mas os **traços** dele chegam: flowers são as únicas imagens ampliadas 2× (128 → 256 no resize), portanto mais borradas e sem artefatos de compressão; bike/cars vêm de uma mesma fonte de fotos de rua 640×480; as outras quatro classes têm blocos JPEG. Isso abre espaço para *shortcut learning* (Geirhos et al., 2020): o head pode separar classes por nitidez ou resolução, e não por conteúdo. O risco é discutido na seção 7 com os resultados.
# 3. **Conteúdo.** A grade mostra que bike e cars são cenas de rua com o objeto muitas vezes pequeno e fora do centro; human é quase só de **cavaleiros em roupa de equitação**, às vezes com o cavalo na foto (rótulo único para uma cena com dois objetos); horses mistura fotos com desenhos, pinturas e imagens em tons de cinza guardadas como RGB. Não há erro de rótulo evidente nas amostras.
# 4. **Tamanho e razão de aspecto.** O pico em 1,33 (4:3) vem de bike/cars; o pico em 1,0 é flowers; human fica abaixo de 1. O pré-processamento (resize do lado menor para 256 + center crop 224) descarta ~17% da largura de cada lado numa imagem 4:3, o que pode cortar bicicletas e carros encostados na borda; em human (retrato) corta-se em cima e embaixo, onde ficam cabeça e pés. As imagens pequenas (flowers com 128 px, várias human/horses com < 200 px) são ampliadas e perdem nitidez em relação às demais.
# 5. **Desbalanceamento.** Razão maior/menor de **2,30** (cars 420 vs horses/human 183). É moderado: o split estratificado preserva as proporções e a avaliação usa accuracy por classe e macro-F1, que não escondem as classes menores. Não foram usados pesos de classe, porque todas as classes têm ≥ 128 imagens de treino; se as classes menores tivessem recall pior, pesos de classe seriam o próximo passo (a seção 7 mostra que não foi o caso).

# %% [markdown]
# ## 3. Split estratificado 70/15/15
# `train_test_split` do scikit-learn aplicado duas vezes com `stratify` e `random_state=42`: 70% treino, depois os 30% restantes divididos ao meio em validação e teste. A validação escolhe a melhor época (early stopping); o teste só é usado uma vez, no final. O split é salvo em CSV para reprodutibilidade.

# %%
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df.label, random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df.label, random_state=SEED)
train_df, val_df, test_df = (d.reset_index(drop=True) for d in (train_df, val_df, test_df))

split_table = pd.DataFrame({name: d.cls.value_counts().reindex(CLASS_NAMES) for name, d in
                            [("train", train_df), ("val", val_df), ("test", test_df)]})
split_table.loc["TOTAL"] = split_table.sum()
print(split_table.to_string())
assert not (set(train_df.md5) & set(val_df.md5) or set(train_df.md5) & set(test_df.md5) or set(val_df.md5) & set(test_df.md5))

pd.concat([d.assign(split=s) for s, d in [("train", train_df), ("val", val_df), ("test", test_df)]])[
    ["path", "cls", "label", "split"]].to_csv(OUT_DIR / "A3_split.csv", index=False)

# %% [markdown]
# ### 3.1 Quase-duplicatas entre partições (auditoria de vazamento)
# O MD5 só pega cópias byte a byte. A mesma foto reescalada, recomprimida ou com pequena edição tem outro MD5 e pode cair em treino e teste. Para auditar isso, calcula-se o **pHash** (`imagehash.phash`, 64 bits, a partir da DCT 32×32 da imagem em cinza) de todas as imagens e a distância de Hamming entre todos os pares.
#
# **Limiar: distância ≤ 4 (de 64 bits).** Imagens não relacionadas ficam em torno de 32 bits de distância (hashes ~independentes); cópias reescaladas ou recomprimidas da mesma foto ficam tipicamente em 0–4, porque essas operações quase não mudam as componentes de baixa frequência da DCT. Um limiar mais frouxo (8–10) começa a juntar fotos diferentes com a mesma composição (p.ex. carros de frente em fundo de rua), e 4 é conservador (poucos falsos positivos). Os pares são mostrados na figura para conferência visual.
#
# **O split não é alterado** (o modelo já foi treinado e avaliado com ele). Em vez disso, a seção 7 também avalia o teste **excluindo** as imagens de teste com quase-duplicata no treino ("teste limpo").

# %%
PHASH_THRESHOLD = 4
split_df = pd.concat([d.assign(split=s) for s, d in [("train", train_df), ("val", val_df), ("test", test_df)]], ignore_index=True)
t0 = time.time()
split_df["phash"] = [str(imagehash.phash(load_rgb(p))) for p in split_df.path]
bits = np.array([np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8)) for h in split_df.phash], dtype=np.float32)
signs = 2 * bits - 1
hamming = np.rint((bits.shape[1] - signs @ signs.T) / 2).astype(int)
ia, ib = np.where(np.triu(hamming <= PHASH_THRESHOLD, k=1))
print(f"pHash de {len(split_df)} imagens em {time.time() - t0:.1f}s")

near_dup_df = pd.DataFrame({
    "path_a": split_df.path.values[ia], "split_a": split_df.split.values[ia], "cls_a": split_df.cls.values[ia],
    "path_b": split_df.path.values[ib], "split_b": split_df.split.values[ib], "cls_b": split_df.cls.values[ib],
    "hamming": hamming[ia, ib]})
near_dup_df["pair"] = ["×".join(sorted((a, b), key=["train", "val", "test"].index)) for a, b in zip(near_dup_df.split_a, near_dup_df.split_b)]
near_dup_df["same_class"] = near_dup_df.cls_a == near_dup_df.cls_b
near_dup_df = near_dup_df.sort_values(["hamming", "pair", "path_a", "path_b"]).reset_index(drop=True)
near_dup_df.to_csv(OUT_DIR / "A3_near_duplicates.csv", index=False)

PAIR_TYPES = ["train×train", "val×val", "test×test", "train×val", "train×test", "val×test"]
near_dup_counts = {t: int((near_dup_df.pair == t).sum()) for t in PAIR_TYPES}
print(f"Pares com Hamming ≤ {PHASH_THRESHOLD}: {len(near_dup_df)} (rótulos diferentes: {int((~near_dup_df.same_class).sum())})")
print(pd.Series(near_dup_counts, name="pares").to_string())
if len(near_dup_df):
    print("\nPor classe (classe da imagem a):\n", pd.crosstab(near_dup_df.cls_a, near_dup_df.pair).to_string())

test_paths = set(test_df.path)
cross_tt = near_dup_df[near_dup_df.pair == "train×test"]
test_leaky_paths = sorted({a if a in test_paths else b for a, b in zip(cross_tt.path_a, cross_tt.path_b)})
print(f"\nImagens de teste com quase-duplicata no treino: {len(test_leaky_paths)} de {len(test_df)}")

# %%
# fig: near_duplicates
cross = near_dup_df[near_dup_df.pair.isin(["train×val", "train×test", "val×test"])]
show_pairs = (cross if len(cross) else near_dup_df).head(8)
if len(show_pairs):
    n_rows = int(np.ceil(len(show_pairs) / 2))
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3.2 * n_rows), squeeze=False, dpi=80)
    for ax in axes.flat:
        ax.axis("off")
    for k, (_, r) in enumerate(show_pairs.iterrows()):
        for j, side in enumerate("ab"):
            ax = axes[k // 2, 2 * (k % 2) + j]
            ax.imshow(load_rgb(r[f"path_{side}"]))
            ax.set_title(f"{r[f'split_{side}']}/{r[f'cls_{side}']} (d={r.hamming})", fontsize=9,
                         color=C_BLUE if r.same_class else C_RED, fontweight="bold")
    kind = "entre partições" if len(cross) else "dentro das partições (não há pares entre partições)"
    plt.suptitle(f"Quase-duplicatas por pHash, Hamming ≤ {PHASH_THRESHOLD}: {kind}", fontweight="bold")
    plt.tight_layout()
    savefig("near_duplicates", dpi=100)
    plt.show()
else:
    print("Nenhum par de quase-duplicatas com o limiar escolhido.")

# %% [markdown]
# **Análise das quase-duplicatas**
#
# - **22 pares** com Hamming ≤ 4, **nenhum entre classes diferentes** (não há conflito de rótulo). Por partição: train×train 12, test×test 1, **train×val 6, train×test 3**, val×val 0, val×test 0.
# - **Onde:** human 11 pares, horses 6, bike 4, dogs 1 (pares são sempre da mesma classe). São as classes que parecem montadas a partir de buscas na web (catálogos de roupa de equitação, fotos de banco de imagens), as mesmas em que a deduplicação por MD5 já tinha removido 19 + 19 cópias exatas. Em cars, cats e flowers não há nenhum par.
# - **Natureza:** todos os pares entre partições da figura têm **d = 0** e são, a olho, **a mesma foto** (mesmo enquadramento, mesma pose, mesma marca d'água), não só uma composição parecida. Como o MD5 é diferente, são a mesma imagem salva com outra resolução ou compressão. O limiar 4 não produziu falsos positivos visíveis.
# - **Vazamento:** **3 das 265 imagens de teste** (1 bike, 2 human) têm cópia no treino, e há 6 pares train×val (até 6 imagens de validação com cópia no treino). É pouco (1,1% do teste), e o efeito é medido na seção 7 com o "teste limpo".

# %% [markdown]
# ## 4. Escolha do modelo — **Rubrica 1.5**
# **Candidatos:** EfficientNet-B0 (slide 14 da Aula 1, Tabela 12-3 do TorchVision) e ResNet-50 (padrão da indústria, mesma família da ResNet-34 usada em aula).
#
# **Decisão: EfficientNet-B0 (`IMAGENET1K_V1`).** Argumentos:
# 1. **Custo:** ~5,3 M parâmetros e ~0,39 GFLOPs por imagem contra ~25,6 M e ~4,1 GFLOPs da ResNet-50 (~10× menos computação). A tabela abaixo lê esses números de `weights.meta`, sem baixar os pesos.
# 2. **VRAM na T4 (15 GB):** o slide 15 mostra que, no treino da ResNet-50 com batch 64, as **ativações ocupam ~84% da VRAM** (~7,2 de ~8,5 GB). Com o backbone congelado, o autograd não guarda as ativações do backbone e esse termo praticamente desaparece. O benchmark abaixo mede o pico de VRAM e o tempo por passo das duas redes, congeladas (*feature extraction*) e em *fine-tuning* completo, para mostrar a folga em números.
# 3. **Nº de classes e tamanho do dataset:** com 7 classes, o head da EfficientNet-B0 tem 1280·7 + 7 = **8.967** parâmetros treináveis (o da ResNet-50 teria 2048·7 + 7 = 14.343). Com 1.234 imagens de treino, um head pequeno sobre features de 1280 dimensões basta; um backbone maior não traz ganho proporcional e aumenta custo e latência.
# 4. **Qualidade das features:** 77,7% top-1 no ImageNet (V1), acima da ResNet-34 do código da aula (73,3%). A ResNet-50 V2 chega a ~80,9%, mas com receita de treino mais pesada e 10× o custo. Para classes de objetos do cotidiano bem representadas no ImageNet, a diferença tende a sumir no linear probe.
#
# Detalhe de implementação: o head da EfficientNet é `classifier = Sequential(Dropout(0.2), Linear(1280, 1000))`; troca-se só `classifier[1]`. O backbone tem BatchNorm e **Stochastic Depth**, que se comportam de forma diferente em `train()` e `eval()`. Por isso o backbone fica sempre em `eval()` (estatísticas do BN e profundidade fixas) e só o head entra em `train()` (dropout ativo).

# %%
def meta_row(name, weights):
    m = weights.meta
    return {"modelo": name, "pesos": weights.name, "params (M)": round(m["num_params"] / 1e6, 2),
            "GFLOPs": m.get("_ops"), "ImageNet top-1 (%)": m.get("_metrics", {}).get("ImageNet-1K", {}).get("acc@1"),
            "tamanho arquivo (MB)": m.get("_file_size")}

meta_df = pd.DataFrame([meta_row("EfficientNet-B0", EfficientNet_B0_Weights.IMAGENET1K_V1),
                        meta_row("ResNet-50", ResNet50_Weights.IMAGENET1K_V1),
                        meta_row("ResNet-50", ResNet50_Weights.IMAGENET1K_V2)])
print(meta_df.to_string(index=False))

# %%
ARCHS = {"efficientnet_b0": efficientnet_b0, "resnet50": resnet50}
HEAD_NAME = {"efficientnet_b0": "classifier", "resnet50": "fc"}

def replace_head(model, arch, n_classes):
    if arch == "efficientnet_b0":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    else:
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model

def set_head_train_mode(model, arch="efficientnet_b0"):
    model.eval()
    getattr(model, HEAD_NAME[arch]).train()

def benchmark_step(arch, finetune, batch_size=64, steps=10, warmup=3):
    """Pico de VRAM e tempo por passo de treino com dados sintéticos (não é um treino do modelo)."""
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = replace_head(ARCHS[arch](weights=None), arch, NUM_CLASSES)
    for name, p in model.named_parameters():
        p.requires_grad = finetune or name.startswith(HEAD_NAME[arch] + ".")
    model.to(device)
    model.train() if finetune else set_head_train_mode(model, arch)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    y = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize(); t0 = time.time()
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
        opt.step()
    torch.cuda.synchronize()
    row = {"modelo": arch, "modo": "fine-tuning completo" if finetune else "feature extraction",
           "params treináveis": sum(p.numel() for p in params),
           "ms/passo (batch 64)": round((time.time() - t0) / steps * 1000, 1),
           "VRAM pico (MB)": round(torch.cuda.max_memory_allocated() / 2**20),
           "VRAM reservada pico (MB)": round(torch.cuda.max_memory_reserved() / 2**20)}
    del model, opt, x, y, params
    gc.collect(); torch.cuda.empty_cache()
    return row

bench_rows = []
if torch.cuda.is_available():
    for arch in ARCHS:
        for finetune in (False, True):
            try:
                bench_rows.append(benchmark_step(arch, finetune))
            except torch.cuda.OutOfMemoryError:
                bench_rows.append({"modelo": arch, "modo": "fine-tuning completo" if finetune else "feature extraction", "VRAM pico (MB)": "OOM"})
                gc.collect(); torch.cuda.empty_cache()
    bench_df = pd.DataFrame(bench_rows)
    print(bench_df.to_string(index=False))
    bench_df.to_csv(OUT_DIR / "A3_benchmark_vram.csv", index=False)
else:
    print("Sem GPU: benchmark de VRAM ignorado.")

# %% [markdown]
# **Análise da escolha**
#
# | Modelo | Modo | Params treináveis | ms/passo (batch 64) | VRAM pico alocada | VRAM pico reservada |
# |---|---|---|---|---|---|
# | EfficientNet-B0 | feature extraction | 8.967 | **67,6** | **707 MB** | 956 MB |
# | EfficientNet-B0 | fine-tuning completo | 4.016.515 | 276,7 (4,1×) | 5.550 MB (7,9×) | 6.118 MB |
# | ResNet-50 | feature extraction | 14.343 | 162,7 | 783 MB | 1.190 MB |
# | ResNet-50 | fine-tuning completo | 23.522.375 | 585,0 (3,6×) | 5.611 MB (7,2×) | 6.104 MB |
#
# 1. **Congelar o backbone corta ~86–87% do pico de VRAM** nas duas redes (707 vs 5.550 MB; 783 vs 5.611 MB). É a mesma ordem de grandeza do slide 15 (ativações ≈ 84% da VRAM de treino da ResNet-50): em *feature extraction* as ativações do backbone são descartadas assim que a camada seguinte as consome, e sobram pesos, a entrada do batch e os buffers do cuDNN.
# 2. **Tudo cabe na T4 (14,56 GB).** Nenhuma das quatro configurações passou de 5,6 GB alocados (6,1 GB reservados pelo cache do PyTorch) com batch 64; nem o fine-tuning completo da ResNet-50 precisaria de *gradient checkpointing*. Uma extrapolação linear grosseira (pico ∝ batch, descontados os pesos) dá batch ~150–170 para o FT completo de qualquer das duas redes e batch na casa de mil para FE. A VRAM, portanto, **não** é o que decide entre as duas redes neste dataset; o que decide é o custo por passo e o risco de overfitting.
# 3. **A EfficientNet-B0 não é 10× mais barata na prática.** Ela tem ~10× menos GFLOPs (0,39 vs 4,09), mas em FT completo usa praticamente a mesma VRAM da ResNet-50 (5.550 vs 5.611 MB) e é só 2,1× mais rápida por passo. As convoluções *depthwise*, a expansão 6× dos blocos MBConv e os blocos *squeeze-and-excitation* geram muitos mapas de ativação grandes e têm baixa intensidade aritmética na GPU; a memória de treino depende das ativações, não do número de parâmetros. Em FE a vantagem de tempo é de 2,4× (67,6 vs 162,7 ms/passo).
# 4. **Conclusão para 1.5:** a EfficientNet-B0 congelada treina com **0,7 GB** (723 MB medidos no treino real, seção 6), 2,4× mais rápido que a ResNet-50 congelada, com top-1 no ImageNet (77,7%) próximo da ResNet-50 V1 (76,1%) e um head menor (8.967 vs 14.343 parâmetros) para 1.234 imagens de treino. A ResNet-50 seria uma escolha igualmente viável na T4; ela só se justificaria se as features da EfficientNet não bastassem, e a seção 7 mostra que bastaram.

# %% [markdown]
# ## 5. Feature extraction: backbone congelado + novo head — **Rubrica 1.1**
# **Pré-processamento.** `weights.transforms()` (Weights Enum API, slide 16, "Regra de Ouro") garante o mesmo tratamento do pré-treino: resize 256 (bicúbico) → center crop 224 → normalização com média e desvio do ImageNet. **Sem augmentation** no treino (decisão D8): o treino é único e a análise de augmentation fica na seção 8.
#
# **Decisão de implementação: forward completo com backbone em `eval()`, em vez de pré-computar as features.** Sem augmentation, as features de cada imagem são idênticas em todas as épocas, então pré-computá-las uma vez seria equivalente e mais rápido. Mantém-se o forward completo porque (i) o modelo treinado é a própria EfficientNet com o head trocado, que pode ser usada direto na inferência; (ii) a checagem de gradientes e a medida de VRAM refletem o cenário real de feature extraction; (iii) o custo é pequeno (1.234 imagens de treino, 0,4 GFLOPs cada). O risco dessa escolha é esquecer o `backbone.eval()`: em `train()` o BatchNorm atualizaria as médias móveis (mudando o backbone "congelado" sem gradiente) e o Stochastic Depth descartaria blocos aleatoriamente. As asserções abaixo e a comparação do `state_dict` do backbone antes/depois do treino verificam isso.

# %%
WEIGHTS = EfficientNet_B0_Weights.IMAGENET1K_V1
preprocess = WEIGHTS.transforms()
print(preprocess)

class ImageDataset(Dataset):
    def __init__(self, frame, transform):
        self.paths, self.labels, self.transform = frame.path.tolist(), frame.label.tolist(), transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        return self.transform(load_rgb(self.paths[i])), self.labels[i]

BATCH_SIZE = 64
g = torch.Generator().manual_seed(SEED)
loader_kw = dict(batch_size=BATCH_SIZE, num_workers=2, pin_memory=torch.cuda.is_available())
train_loader = DataLoader(ImageDataset(train_df, preprocess), shuffle=True, generator=g, **loader_kw)
val_loader = DataLoader(ImageDataset(val_df, preprocess), shuffle=False, **loader_kw)
test_loader = DataLoader(ImageDataset(test_df, preprocess), shuffle=False, **loader_kw)
xb, yb = next(iter(train_loader))
print("batch:", tuple(xb.shape), xb.dtype, f"| média {xb.mean():.3f} desvio {xb.std():.3f} (normalizado)")

# %%
model = efficientnet_b0(weights=WEIGHTS)
for p in model.parameters():
    p.requires_grad = False
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
model = model.to(device)
print(model.classifier)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
print(f"Parâmetros totais: {total_params:,} | treináveis: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
print("Parâmetros treináveis:", trainable_names)
assert in_features == 1280
assert trainable_names == ["classifier.1.weight", "classifier.1.bias"]
assert trainable_params == in_features * NUM_CLASSES + NUM_CLASSES

# %%
set_head_train_mode(model)
assert not model.features.training and model.classifier.training
loss = F.cross_entropy(model(xb.to(device)), yb.to(device))
loss.backward()
with_grad = [n for n, p in model.named_parameters() if p.grad is not None]
print("Parâmetros que receberam gradiente:", with_grad)
assert with_grad == ["classifier.1.weight", "classifier.1.bias"], "Gradiente vazou para o backbone!"
model.zero_grad(set_to_none=True)
backbone_ref = {k: v.detach().clone() for k, v in model.features.state_dict().items()}
print("OK: só o head tem gradiente; snapshot do backbone (pesos + estatísticas do BN) guardado.")

# %% [markdown]
# ## 6. Treino único do head
# AdamW (lr 1e-3, weight decay 1e-4) **só nos parâmetros do head** e `CosineAnnealingLR`, como na Aula 1 (células 12 e 16); até 15 épocas com early stopping pela loss de validação (paciência 4). O melhor head é salvo no Drive. O treino leva poucos minutos, então não há retomada de checkpoint: rodar de novo reproduz o resultado.

# %%
EPOCHS, LR, WD, PATIENCE = 15, 1e-3, 1e-4, 4
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def run_epoch(model, loader, optimizer=None):
    """Uma passada pelo loader; treina se receber otimizador (adaptado de Aula-1 nb[10])."""
    training = optimizer is not None
    set_head_train_mode(model) if training else model.eval()
    total_loss, ys, probs = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            ys.append(y.cpu())
            probs.append(logits.detach().softmax(1).cpu())
    y_true, P = torch.cat(ys).numpy(), torch.cat(probs).numpy()
    y_pred = P.argmax(1)
    return {"loss": total_loss / len(y_true), "acc": 100.0 * (y_pred == y_true).mean(),
            "y_true": y_true, "y_pred": y_pred, "probs": P}

# %%
history, best_val_loss, best_epoch, bad_epochs = [], float("inf"), 0, 0
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
t_train = time.time()
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    lr = optimizer.param_groups[0]["lr"]
    tr = run_epoch(model, train_loader, optimizer)
    va = run_epoch(model, val_loader)
    scheduler.step()
    history.append({"epoch": epoch, "lr": lr, "train_loss": tr["loss"], "train_acc": tr["acc"],
                    "val_loss": va["loss"], "val_acc": va["acc"], "epoch_time_s": time.time() - t0})
    improved = va["loss"] < best_val_loss
    if improved:
        best_val_loss, best_epoch, bad_epochs = va["loss"], epoch, 0
        best_head = copy.deepcopy(model.classifier.state_dict())
        torch.save(best_head, OUT_DIR / "A3_best_head.pt")
    else:
        bad_epochs += 1
    pd.DataFrame(history).to_csv(OUT_DIR / "A3_history.csv", index=False)
    print(f"ep {epoch:2d} | lr {lr:.2e} | train loss {tr['loss']:.4f} acc {tr['acc']:6.2f}% | "
          f"val loss {va['loss']:.4f} acc {va['acc']:6.2f}% | {time.time() - t0:5.1f}s {'*' if improved else ''}")
    if bad_epochs >= PATIENCE:
        print(f"Early stopping: {PATIENCE} épocas sem melhora na val loss.")
        break
train_time_s = time.time() - t_train
peak_vram_train_mb = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
peak_vram_train_reserved_mb = torch.cuda.max_memory_reserved() / 2**20 if torch.cuda.is_available() else None
ram_after_train = ram_stats()
hist_df = pd.DataFrame(history)
print(f"\nMelhor época: {best_epoch} (val loss {best_val_loss:.4f}) | tempo de treino: {train_time_s:.1f}s")
if peak_vram_train_mb:
    print(f"VRAM pico no treino: {peak_vram_train_mb:.0f} MB alocados | {peak_vram_train_reserved_mb:.0f} MB reservados pelo cache do PyTorch")
print("RAM após o treino:", ram_after_train)

model.classifier.load_state_dict(best_head)
backbone_now = model.features.state_dict()
assert all(torch.equal(v, backbone_now[k]) for k, v in backbone_ref.items()), "O backbone mudou durante o treino!"
print("OK: backbone idêntico ao pré-treinado após o treino (pesos e running stats do BN).")

# %%
# fig: training_curves
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, key, label in [(axes[0], "loss", "Loss (cross-entropy)"), (axes[1], "acc", "Accuracy (%)")]:
    ax.plot(hist_df.epoch, hist_df[f"train_{key}"], "o-", color=C_BLUE, label="train")
    ax.plot(hist_df.epoch, hist_df[f"val_{key}"], "s-", color=C_CYAN, label="val")
    ax.axvline(best_epoch, color=C_RED, linestyle="--", label=f"melhor época ({best_epoch})")
    ax.set_xlabel("época"); ax.set_title(label, fontweight="bold"); ax.legend()
plt.suptitle("EfficientNet-B0 congelada + head linear: curvas de treino", fontweight="bold")
plt.tight_layout()
savefig("training_curves")
plt.show()

# %% [markdown]
# **Análise das curvas**
#
# 1. **Convergência muito rápida.** Depois de **uma** época a val acc já é 98,11% (260/265); da época 2 à 8 fica em 99,25% (263/265) e da 9 em diante em 99,62% (264/265). Em accuracy, o treino inteiro mudou o destino de **uma única imagem de validação** depois da época 2. A loss continua caindo (val 0,400 → 0,161 → 0,076 na época 5 → 0,050), o que indica que o head está ficando mais confiante nas imagens que já acertava, não acertando imagens novas.
# 2. **Sem overfitting.** Nas épocas 1–4 a loss de treino fica **acima** da de validação (1,058 vs 0,400 na época 1). Há dois motivos: a loss de treino é a média ao longo da época, enquanto o head ainda está aprendendo, e o dropout de 0,2 do head fica ativo no treino e desligado na validação. As curvas se cruzam na época 5 (empate) e daí em diante terminam com um gap pequeno e estável (train 0,038, val 0,050 na época 15; train acc 99,68% vs val 99,62%). Com 8.967 parâmetros (~7 por imagem de treino) sobre features fixas, o head é praticamente uma regressão logística e tem pouca capacidade de memorizar.
# 3. **A melhor época foi a última (15), e o early stopping não disparou.** A val loss melhorou em **todas** as 15 épocas, mas nas últimas cinco o ganho foi de só 0,0018 (0,0517 → 0,0499), porque o *cosine* levou a LR de 1e-3 para 1,1e-5. Duas leituras: (i) o critério é `<` estrito, sem `min_delta`, então qualquer melhora de 1e-4 zera a paciência; mesmo com `min_delta=1e-3` a melhor época seria a 12 ou a 13, dependendo do arredondamento, e a paciência de 4 não se esgotaria antes do limite de 15 épocas, ou seja, o early stopping nunca teve espaço para agir neste orçamento; (ii) a loss **ainda não convergiu** de fato: foi o calendário da LR que a congelou. Mais épocas ou LR inicial maior (p.ex. 3e-3) reduziriam mais a loss, mas como treino e validação já estão ~separáveis linearmente, isso aumentaria sobretudo o tamanho dos pesos e a confiança das predições, com ganho esperado de no máximo 1 imagem de validação e risco de sobreconfiança. O que o cosine fez de útil foi estabilizar o fim: as curvas ficaram planas, sem oscilação.
# 4. **Custo.** ~9,5 s por época (**142,7 s** no total). VRAM: **723 MB alocados** no pico e **1.264 MB reservados** pelo alocador do PyTorch (o cache guarda blocos liberados para reuso; é o valor mais próximo do que o `nvidia-smi` mostraria, sem o contexto CUDA). RAM depois do treino: 1.864 MB residentes no processo principal (pico 1.901 MB) e 1.301 MB no maior worker do DataLoader; as imagens não ficam em memória, então esse valor não cresce com o número de épocas. Como o backbone é fixo e não há augmentation, quase todo o tempo é decodificação/redimensionamento das imagens e o forward do backbone, repetidos 15 vezes para produzir as mesmas features. Pré-computar as features uma vez deixaria o treino do head em menos de 1 s por época (seção 5).

# %% [markdown]
# ## 7. Avaliação no conjunto de teste — **Rubrica 1.2**
# Accuracy global, accuracy por classe (igual ao recall da classe), precision/F1 por classe, macro-F1, matriz de confusão e exemplos de erros. O teste é avaliado uma única vez, com o head da melhor época.

# %%
te = run_epoch(model, test_loader)
y_true, y_pred, P = te["y_true"], te["y_pred"], te["probs"]
cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
per_class_acc = cm.diagonal() / cm.sum(1)
prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0)
test_acc = 100.0 * (y_true == y_pred).mean()
macro_f1 = f1_score(y_true, y_pred, average="macro")

per_class_df = pd.DataFrame({"classe": CLASS_NAMES, "n_test": support, "accuracy (%)": 100 * per_class_acc,
                             "precision": prec, "recall": rec, "f1": f1}).round(4)
per_class_df.to_csv(OUT_DIR / "A3_per_class_metrics.csv", index=False)
print(f"Accuracy global (test): {test_acc:.2f}% | macro-F1: {macro_f1:.4f} | test loss: {te['loss']:.4f}\n")
print(per_class_df.to_string(index=False))
print("\n" + classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

test_pred_df = test_df[["path", "cls"]].assign(pred=[CLASS_NAMES[i] for i in y_pred], conf=P.max(1).round(4),
                                               near_dup_in_train=test_df.path.isin(test_leaky_paths))
test_pred_df.to_csv(OUT_DIR / "A3_test_predictions.csv", index=False)

clean = ~test_pred_df.near_dup_in_train.values
test_acc_clean = 100.0 * (y_true[clean] == y_pred[clean]).mean()
macro_f1_clean = f1_score(y_true[clean], y_pred[clean], average="macro", labels=range(NUM_CLASSES), zero_division=0)
print(f"\nTeste limpo (sem as {int((~clean).sum())} imagens com quase-duplicata no treino): "
      f"{int(clean.sum())} imagens | accuracy {test_acc_clean:.2f}% | macro-F1 {macro_f1_clean:.4f}")
print(f"Accuracy nas imagens de teste COM quase-duplicata no treino: "
      + (f"{100.0 * (y_true[~clean] == y_pred[~clean]).mean():.2f}%" if (~clean).any() else "n/a (nenhuma)"))

# %%
# fig: per_class_accuracy
fig, ax = plt.subplots(figsize=(8, 4))
colors = [C_GREEN if a >= test_acc / 100 else C_RED for a in per_class_acc]
bars = ax.bar(CLASS_NAMES, 100 * per_class_acc, color=colors)
ax.bar_label(bars, fmt="%.1f")
ax.axhline(test_acc, color=C_BLUE, linestyle="--", label=f"global = {test_acc:.2f}%")
ax.set_ylim(min(80, 100 * per_class_acc.min() - 5), 101)
ax.set_ylabel("accuracy (%)")
ax.set_title("Accuracy por classe no teste", fontweight="bold")
ax.legend(loc="lower right")
savefig("per_class_accuracy")
plt.show()

# %%
# fig: confusion_matrix
cm_norm = cm / cm.sum(1, keepdims=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, mat, title, fmt in [(axes[0], cm, "Contagens", "d"), (axes[1], cm_norm, "Normalizada por linha (recall)", ".2f")]:
    im = ax.imshow(mat, cmap="Blues")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=9)
    ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
    ax.set_xlabel("predito"); ax.set_ylabel("real"); ax.set_title(title, fontweight="bold"); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
plt.suptitle("Matriz de confusão — teste", fontweight="bold")
plt.tight_layout()
savefig("confusion_matrix")
plt.show()

off = cm.copy(); np.fill_diagonal(off, 0)
pairs = sorted(((off[i, j], CLASS_NAMES[i], CLASS_NAMES[j]) for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if off[i, j]), reverse=True)
print("Confusões mais frequentes (real -> predito):", [(f"{a}->{b}", int(n)) for n, a, b in pairs[:5]])

# %%
# fig: error_examples
wrong = test_pred_df[test_pred_df.cls != test_pred_df.pred]
show = wrong.sort_values("conf", ascending=False) if len(wrong) else test_pred_df.sort_values("conf").head(8)
title = f"Erros no teste ({len(wrong)} de {len(test_pred_df)})" if len(wrong) else "Sem erros: 8 acertos de menor confiança"
show = show.head(8)
IMNET_MEAN, IMNET_STD = np.array(preprocess.mean), np.array(preprocess.std)

def model_input_image(path):
    """Exatamente o tensor que entra no modelo (resize 256 + center crop 224), desnormalizado para exibição."""
    x = preprocess(load_rgb(path)).numpy().transpose(1, 2, 0)
    return np.clip(x * IMNET_STD + IMNET_MEAN, 0, 1)

n_cols = min(4, len(show))
n_rows = max(1, int(np.ceil(len(show) / n_cols)))
fig, axes = plt.subplots(n_rows, 2 * n_cols, figsize=(3.2 * 2 * n_cols, 3.6 * n_rows), squeeze=False)
for ax in axes.flat:
    ax.axis("off")
for k, (_, r) in enumerate(show.iterrows()):
    ax_orig, ax_in = axes[k // n_cols, 2 * (k % n_cols)], axes[k // n_cols, 2 * (k % n_cols) + 1]
    orig = load_rgb(r.path)
    ok = r.cls == r.pred
    ax_orig.imshow(orig)
    ax_orig.set_title(f"real: {r.cls} | pred: {r.pred} ({100 * r.conf:.0f}%)\noriginal {orig.size[0]}×{orig.size[1]}",
                      color=C_GREEN if ok else C_RED, fontweight="bold", fontsize=9)
    ax_in.imshow(model_input_image(r.path))
    ax_in.set_title("entrada do modelo\n(resize 256 + crop 224)", fontsize=9)
plt.suptitle(title, fontweight="bold")
plt.tight_layout()
savefig("error_examples")
plt.show()

# %% [markdown]
# **Análise dos resultados no teste**
#
# **Números.** Accuracy global **99,25%** (263/265), **macro-F1 0,9929**, test loss 0,063. Só 2 erros: 1 bike → cars e 1 cats → dogs. Cinco classes ficaram em 100%. A precision de cars (0,984) e de dogs (0,969) cai pelos mesmos dois erros.
#
# **Teste limpo (sem quase-duplicatas do treino).** Tirando as 3 imagens de teste que têm cópia no treino (seção 3.1), sobram 262: accuracy **99,24%** (260/262) e macro-F1 **0,9928**, contra 99,25% e 0,9929 no teste completo. As 3 imagens vazadas foram acertadas, mas os 2 erros não estão entre elas, então o vazamento por quase-duplicata **não infla o resultado de forma mensurável** (−0,01 p.p., muito abaixo de 1 imagem). O 99% não vem de memorização de cópias.
#
# **Incerteza: 265 imagens medem pouco.** Com 2 erros em 265, o intervalo de Wilson de 95% para a accuracy global é **[97,3%; 99,8%]**. Por classe é bem mais largo, porque 1 erro vale 1,9 p.p. em bike (54 imagens) e 3,3 p.p. em cats (30):
#
# | Classe | Acertos | Accuracy | IC 95% (Wilson) |
# |---|---|---|---|
# | bike | 53/54 | 98,1% | [90,2%; 99,7%] |
# | cats | 29/30 | 96,7% | [83,3%; 99,4%] |
# | cars | 63/63 | 100% | [94,3%; 100%] |
# | flowers | 32/32 | 100% | [89,3%; 100%] |
# | dogs / horses / human | 31/31, 28/28, 27/27 | 100% | limite inferior 89,0% / 87,9% / 87,5% |
#
# Na prática, "96,7% em cats" e "100% em horses" não são distinguíveis com esse teste. Os 100% por classe dizem que a accuracy real provavelmente passa de ~88–94%, não que é perfeita. A validação conta a mesma história (264/265), então as duas partições juntas somam 3 erros em 530 imagens (99,4%).
#
# **Os dois erros (figura `A3_error_examples.png`).**
# - **cats → dogs (78%)**: gato preto, em retrato (original 282×499), sobre fundo branco estourado, com uma mão humana fazendo carinho e uma coleira verde. O painel "entrada do modelo" **confirma o efeito do crop**: o resize para 256×453 seguido do center crop 224 mantém só a faixa de ~25% a ~75% da altura, e o que o modelo recebe começa na boca aberta. **Olhos e orelhas ficaram de fora**; sobraram a boca, a mão, a coleira e o corpo preto, ampliados e sem textura. Que o corte do rosto seja a **causa** do erro continua hipótese (não foi medido, p.ex. reclassificando a imagem inteira com *padding*), assim como o papel da coleira e da mão como pistas de "cachorro".
# - **bike → cars (68%)**: cena de rua em que a bicicleta ocupa uma fração pequena da imagem, encostada no muro à esquerda, e a cena é dominada por contêineres de lixo, calçada e asfalto. Não há carro na foto. Aqui o crop **não** é o problema: o painel "entrada do modelo" mostra a bicicleta inteira, perto da borda esquerda. Como bike e cars vêm da mesma fonte de cenas urbanas 640×480, o erro **sugere** que o head usa em parte o **contexto** "rua com asfalto" como pista de cars e que, quando o objeto é pequeno, o contexto vence. É uma hipótese de *shortcut* de contexto, a verificar com Grad-CAM (abaixo).
# - Os dois erros têm **confiança abaixo de 80%**. Para comparar: uma loss de teste de 0,063 equivale a uma probabilidade média geométrica de ~94% na classe correta (e^−0,063), e os dois erros ficam bem abaixo disso. Isso é compatível com uma opção de rejeição (mandar para revisão as predições com confiança < 0,8); o custo dessa regra em acertos rejeitados precisa ser medido em `A3_test_predictions.csv`.
# - **Desbalanceamento:** não há relação entre tamanho da classe e erro. cars (a maior) acertou tudo, bike (a segunda maior) errou 1, e horses/human (as menores, 183 imagens) acertaram tudo. Os erros parecem vir do conteúdo da imagem (objeto pequeno; rosto fora do crop), não da frequência da classe.
# - As confusões seguem os pares esperados (felino ↔ canino, veículo ↔ veículo). **Não houve** confusão horses ↔ human, apesar de human ser quase só de cavaleiros, às vezes com o cavalo na foto.
#
# **Risco de *shortcut*: o que os 100% em flowers podem significar.** Pela EDA, flowers é a única classe em PNG RGBA, toda em 128×128 (ampliada 2× pelo pré-processamento); bike/cars são todas BMP 640×480; as outras quatro são JPEG de tamanhos variados. Há duas explicações para o 100% em flowers, e **este teste não separa uma da outra**:
# 1. *Semântica*: flowers é a única classe vegetal, as flores são o objeto central da foto e o ImageNet tem classes de flores (*daisy* etc.). O 100% é o esperado.
# 2. *Atalho de aquisição*: o head pode estar usando a assinatura da imagem ampliada (borrão, pouca energia de alta frequência, ausência de blocos JPEG). O canal alfa ficou fora dessa lista: a seção 2 mediu que ele é 100% opaco em todas as flowers. Features congeladas de ImageNet codificam nitidez e textura, e um head linear pode explorá-las.
#
# Como o teste vem da **mesma fonte** do treino, ele carrega os mesmos artefatos, e as duas hipóteses preveem 100%. O fato de os dois erros ficarem dentro de um mesmo grupo de formato (bike/cars em BMP; cats/dogs em JPEG) é compatível com as duas hipóteses, porque esses grupos também são semanticamente próximos. Testes que eu faria para separar:
# - ~~Medir o alfa~~ **feito** (seção 2): 0% de pixels com A < 255 nas 210 flowers. A composição em branco não muda nada, e o risco que sobra é de resolução e compressão.
# - **Troca contrafactual de aquisição**: reduzir imagens de teste de outras classes para 128×128, salvar em PNG e reclassificar; e, no sentido inverso, recomprimir as flowers em JPEG (q≈75). Se cats reduzidos passarem a ser "flowers", ou se as flowers em JPEG caírem, o atalho está confirmado.
# - **Baseline de metadados**: um classificador só com (largura, altura, formato) já acerta flowers e separa {bike, cars} das demais. Isso prova que o vazamento **existe nos dados**; os testes acima mostram se o modelo **o usa**.
# - **Grad-CAM** no último bloco convolucional (`features[8]`) para flowers e para os dois erros: a ativação deve cair nas pétalas, no rosto do gato e na bicicleta, não em bordas ou fundo.
# - **Teste externo**: 30–50 imagens por classe de outra fonte (p.ex. flores em JPEG de alta resolução; bicicletas e carros fora das ruas dessa coleção). Esse é o único teste que mede generalização de verdade.
#
# Em resumo: 99,25% é um bom resultado **para esta distribuição**, mas não prova que o modelo aprendeu "flor", "bicicleta" e "carro" de forma transferível. Os erros sugerem (hipótese a verificar com Grad-CAM) que ele usa contexto de cena, e a EDA mostra que a fonte e o formato das imagens estão amarrados às classes.

# %% [markdown]
# ## 8. Análise de data augmentation para este domínio — **Rubrica 1.3**
# O treino acima **não** usou augmentation (decisão D8: treino único com `weights.transforms()`, como no código da aula). Esta seção analisa, para cada família de transformação, se ajudaria neste dataset e **para quais classes pode distorcer ou prejudicar**. A figura abaixo só ilustra as transformações em imagens do dataset; nenhuma delas entra no treino.
#
# Para cada família: se ajuda neste dataset e para quais classes pode distorcer, à luz da EDA (seção 2) e dos dois erros do teste (seção 7).
#
# **(a) Geométrica.**
# - *Flip horizontal* (`RandomHorizontalFlip`): **ajuda** em todas as 7 classes. Bicicletas, carros, animais, flores e pessoas continuam plausíveis espelhados, e nenhuma classe depende de assimetria esquerda/direita. As placas de rua em bike/cars ficam com o texto espelhado, mas o texto não define a classe. É a augmentation mais segura aqui.
# - *Rotação pequena* (±10–15°): ajuda a lidar com fotos levemente inclinadas. Rotações grandes, *flip vertical* e rotações de 90°/180° **prejudicam** bike, cars, horses e human, que têm orientação canônica pela gravidade (carro de cabeça para baixo não aparece no teste e ensina invariâncias inúteis). Flowers é a exceção: fotos de flores vistas de cima são quase invariantes à rotação.
#
# **(b) Cor.**
# - *Brilho/contraste leves* (`ColorJitter(0.2, 0.2)`): ajudam em todas as classes, simulando iluminação e câmeras diferentes.
# - *Saturação/matiz fortes e grayscale* (`RandomGrayscale`): **prejudicam flowers**, cuja cor saturada é uma das pistas mais fortes contra as demais classes (e contra fundos verdes), e podem prejudicar horses/dogs/cats, em que a cor e o padrão da pelagem ajudam, ainda que pouco, a separar as espécies. Em cars a cor não define a classe, então jitter de matiz ajuda o modelo a não associar "vermelho" a "carro". A EDA não achou imagens no modo `L`, mas horses e human têm desenhos e fotos antigas em cinza/sépia guardados como RGB; um `RandomGrayscale` com p baixo (≈0,05) reduziria o atalho "imagem sem cor → horses/human", ao custo de apagar a pista de cor em flowers nessas poucas amostras.
#
# **(c) Escala / recorte.**
# - `RandomResizedCrop(224, scale=(0.6, 1.0))` **ajuda**: os objetos aparecem em tamanhos muito diferentes (EDA de tamanhos) e o center crop fixo corta as bordas. O erro cats → dogs parece ser isso: pela geometria, o crop central de uma imagem em retrato provavelmente tirou o rosto do gato (a confirmar na figura com a entrada real do modelo). Treinar com recortes em posições variadas expõe o head a vistas parciais; na inferência, a alternativa é redimensionar sem cortar (*padding*) ou usar TTA com vários recortes.
# - Recortes agressivos (o padrão `scale=(0.08, 1.0)`) **prejudicam**: em bike e cars o recorte pode ficar só com uma roda ou só com asfalto, e em cenas como a do erro bike → cars (bicicleta pequena num canto) a maioria dos recortes **não contém bicicleta nenhuma**, o que vira ruído de rótulo e reforçaria o possível atalho "rua → cars". Em cats/dogs/horses, um recorte só de pelo remove a forma da cabeça, que é o que separa as três classes (a figura abaixo mostra um recorte só com o olho do gato e outro só com a pata do cavalo); em human, sobra só a roupa. O rótulo deixa de ser verdadeiro para o recorte.
# - A razão de aspecto do crop também deve ficar próxima de 1 (padrão 3/4–4/3), para não achatar animais e veículos.
#
# **(d) Normalização.** Não é opcional: com o backbone congelado, a entrada **tem** de seguir a média/desvio do ImageNet (`weights.transforms()`), porque os pesos e as *running stats* do BN (fixas em `eval()`) foram calibrados nessa distribuição. Estatísticas do próprio dataset deslocariam a entrada de todas as camadas e degradariam as features de todas as classes. Nenhuma classe é "distorcida" pela normalização correta; o risco está em usar a errada.
#
# **(e) Resolução e compressão (específica deste dataset).** A EDA mostrou que resolução e formato estão amarrados à classe (flowers = 128×128 PNG; bike/cars = 640×480 BMP; demais = JPEG). Aplicar a **todas** as classes, com probabilidade ~0,3, uma redução aleatória para 96–160 px seguida de ampliação, e uma recompressão JPEG com qualidade 60–95 (`torchvision.transforms.v2.JPEG`), iguala as assinaturas de aquisição entre as classes e tira do head a pista "imagem borrada → flowers". Não distorce nenhuma classe, porque um humano reconhece todas elas em 128 px. É a augmentation que ataca o risco de *shortcut* discutido na seção 7.
#
# **Observação sobre feature extraction.** Com o backbone congelado, a augmentation só age através do head: o backbone não aprende invariâncias novas, então o ganho esperado é menor do que em fine-tuning, e ela impede pré-computar as features (cada época vê features diferentes). O pipeline recomendado, se houvesse um segundo treino, reúne as **5 estratégias recomendadas** da tabela abaixo (numeradas nos comentários), mais a normalização (d); requer `torchvision` ≥ 0.19 por causa de `v2.JPEG`:
# ```python
# import torchvision.transforms.v2 as T2
# train_tf = T2.Compose([
#     T2.ToImage(),                                                          # PIL -> tensor uint8
#     T2.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(3/4, 4/3),
#                          interpolation=T2.InterpolationMode.BICUBIC),      # 2. crop moderado
#     T2.RandomHorizontalFlip(p=0.5),                                        # 1. flip horizontal
#     T2.RandomRotation(10),                                                 # 5. rotação pequena
#     T2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0), # 3. fotométrica leve
#     T2.RandomApply([T2.RandomResize(96, 161), T2.Resize((224, 224))], p=0.3),  # 4a. resolução 96-160 px
#     T2.RandomApply([T2.JPEG(quality=(60, 95))], p=0.3),                    # 4b. recompressão JPEG q 60-95
#     T2.ToDtype(torch.float32, scale=True),
#     T2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),            # (d) normalização ImageNet
# ])
# ```

# %%
# fig: augmentation_examples
base_tf = T.Compose([T.Resize(256), T.CenterCrop(224)])
AUGS = {"original": T.Lambda(lambda x: x),
        "flip horizontal": T.RandomHorizontalFlip(p=1.0),
        "flip vertical": T.RandomVerticalFlip(p=1.0),
        "rotação 30°": T.RandomRotation((30, 30)),
        "ColorJitter forte": T.ColorJitter(0.4, 0.4, 0.6, 0.3),
        "grayscale": T.RandomGrayscale(p=1.0),
        "crop moderado": T.RandomResizedCrop(224, scale=(0.6, 1.0)),
        "crop agressivo": T.RandomResizedCrop(224, scale=(0.08, 0.15))}
demo_classes = [c for c in ["flowers", "bike", "cats", "horses"] if c in CLASS_NAMES]
torch.manual_seed(SEED)
fig, axes = plt.subplots(len(demo_classes), len(AUGS), figsize=(1.7 * len(AUGS), 1.9 * len(demo_classes)), squeeze=False, dpi=80)
for r, cls in enumerate(demo_classes):
    img = base_tf(load_rgb(df[df.cls == cls].path.iloc[0]))
    for c, (name, aug) in enumerate(AUGS.items()):
        ax = axes[r, c]
        ax.imshow(aug(img)); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if r == 0:
            ax.set_title(name, fontsize=10, fontweight="bold")
        if c == 0:
            ax.set_ylabel(cls, fontsize=11, fontweight="bold")
plt.suptitle("Augmentations candidatas (só ilustração; não usadas no treino)", fontweight="bold")
plt.tight_layout()
savefig("augmentation_examples", dpi=100)
plt.show()

# %% [markdown]
# **Análise: estratégias recomendadas e descartadas**
#
# A figura acima mostra, em imagens reais do dataset, por que algumas transformações ficam de fora: o cavalo de cabeça para baixo no *flip* vertical, a flor rosa que vira roxa no *ColorJitter* forte, a flor que perde a cor no *grayscale* e o recorte agressivo que deixa só o olho do gato ou a pata do cavalo.
#
# | # | Estratégia | Parâmetros | Por que ajuda aqui | Classes em que pode prejudicar |
# |---|---|---|---|---|
# | 1 | Flip horizontal | p = 0,5 | Dobra as poses de animais, pessoas e veículos; nenhuma classe depende de esquerda/direita | Nenhuma |
# | 2 | Crop moderado | `RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(3/4, 4/3))` | Objetos em escalas muito diferentes; o erro cats → dogs provavelmente veio de um crop central que tirou o rosto | Com `scale` < ~0,5: bike/cars (recorte sem o objeto) e cats/dogs/horses (recorte sem a cabeça) |
# | 3 | Fotométrica leve | `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0)` | Exposição varia muito (o gato do erro está sobre fundo estourado) | flowers se houver matiz/saturação fortes; cats/dogs/horses (cor da pelagem) com jitter forte |
# | 4 | Resolução e compressão | redução para 96–160 px + JPEG q 60–95, p ≈ 0,3, em todas as classes | Iguala as assinaturas de aquisição que estão amarradas às classes (flowers 128 px PNG; bike/cars BMP) | Nenhuma nessa intensidade |
# | 5 | Rotação pequena | ±10° | Fotos levemente inclinadas | Nenhuma nessa faixa |
#
# **Descartadas:** *flip* vertical e rotações de 90°/180° (bike, cars, horses, human, cats e dogs têm orientação dada pela gravidade; só flowers vistas de cima toleram); *hue* forte e *grayscale* frequente (flowers, cuja cor é pista forte); recorte agressivo `scale=(0.08, …)` (todas as classes com objeto pequeno ou definido pela cabeça); normalização com estatísticas do próprio dataset (quebra a calibração do backbone congelado, item d).
#
# **Quanto se esperaria ganhar?** No teste atual, **no máximo 2 imagens** (0,75 p.p.), um ganho que cabe inteiro no IC de 95% [97,3%; 99,8%]. Com 265 imagens de teste não dá para medir o efeito da augmentation por accuracy: seria preciso repetir com várias seeds ou validação cruzada, comparar a loss/calibração em vez da accuracy, ou, melhor, avaliar num **teste de estresse** (imagens de teste reduzidas, recomprimidas, com crop descentralizado, ou de outra fonte). É nesse tipo de teste que as estratégias 2 e 4 devem fazer diferença, e não nos 99,25% da mesma distribuição.

# %% [markdown]
# ## 9. Feature extraction vs fine-tuning — **Rubrica 1.4**
# A escolha depende de dois eixos: **tamanho do dataset** e **distância entre o domínio e o ImageNet**.
#
# | | Domínio próximo do ImageNet | Domínio distante (médico, satélite, texturas industriais) |
# |---|---|---|
# | **Dataset pequeno** | **Feature extraction** (este caso) | Fine-tuning parcial das últimas camadas, LR baixa, augmentation forte; ou backbone pré-treinado no domínio (MONAI, TorchGeo) |
# | **Dataset grande** | Fine-tuning parcial ou completo (ganho marginal sobre FE) | Fine-tuning completo |
#
# **Neste dataset, feature extraction é a escolha certa:**
# 1. **Domínio:** as 7 classes são objetos do cotidiano fotografados, amplamente cobertos pelo ImageNet (dezenas de raças de cães e gatos, *sorrel* para cavalos, *mountain bike*, *sports car*, *daisy*; pessoas aparecem em milhares de imagens, embora "pessoa" não seja classe do ImageNet-1k). As camadas finais do backbone (semântica de alto nível, slide 17) já separam essas categorias, e um head linear basta.
# 2. **Tamanho:** 1.234 imagens de treino (128–294 por classe) para ~4,0 M de parâmetros no backbone: descongelar tudo daria ~3.200 parâmetros por imagem de treino, com alto risco de overfitting e de *catastrophic forgetting* das features genéricas. O head tem 8.967 parâmetros, ~7 por imagem de treino.
# 3. **Custo:** pelo benchmark da seção 4, o FT completo da EfficientNet-B0 custaria **7,9× a VRAM** (5.550 vs 707 MB) e **4,1× o tempo por passo** (277 vs 67,6 ms). Em FE o treino real usou 723 MB alocados (1.264 MB reservados) e levou 143 s.
#
# **Quando o fine-tuning compensaria:** se a accuracy de validação estacionar abaixo do desejado com erros concentrados em classes finas (p.ex. cats↔dogs), o passo seguinte é o **fine-tuning parcial** do último estágio (`features[7:]` na EfficientNet-B0, o equivalente ao `layer4` da ResNet), com LR diferencial (backbone ~1e-5, head ~1e-3, como no desafio da Aula 1, nb[21]) e augmentation da seção 8, sempre partindo do head já treinado (linear probe antes, como recomenda a Aula 6, slide 12). Em domínios distantes, como os defeitos de aço da A1, o FE sozinho tende a falhar, porque as features de objetos do ImageNet não descrevem bem texturas.

# %% [markdown]
# **Análise com os resultados**
#
# - **O teto já foi atingido com o head.** Só treinando 8.967 parâmetros (0,22% do modelo), o teste chegou a **99,25%** (IC 95% [97,3%; 99,8%]) e a validação a 99,62%. O fine-tuning pode ganhar no máximo as **2 imagens** erradas (+0,75 p.p.), um ganho dentro do intervalo de confiança e que o teste não tem resolução para confirmar. Pagar 7,9× a VRAM e 4,1× o tempo por passo por um ganho que não dá para medir não se justifica.
# - **O FT poderia piorar.** Com ~3.200 parâmetros por imagem de treino, o FT completo tem capacidade para decorar as 1.234 imagens e, pior, para **adaptar o backbone às assinaturas de aquisição** discutidas na seção 7 (resolução de flowers, cenas de rua de bike/cars). O backbone congelado limita o modelo às features genéricas do ImageNet, e isso reduz (não elimina) o espaço para atalhos. A Aula 6 (slide 12) resume o mesmo ponto para o CLIP: linear probe tem "zero esquecimento"; FT completo arrisca a robustez fora da distribuição.
# - **Os dois erros não pedem FT.** Num deles o center crop tirou olhos e orelhas do gato (visto na figura de erros; que isso causou o erro é hipótese) e o outro de uma bicicleta pequena numa cena de rua. Os dois se resolvem melhor no pré-processamento (sem crop ou TTA com vários recortes) e no dado (recorte moderado na augmentation) do que ajustando 4 M de pesos.
# - **Quando eu faria FT aqui:** se o teste externo ou o teste contrafactual de aquisição (seção 7) mostrasse queda forte, a ordem seria: (1) augmentation de resolução/compressão com o backbone ainda congelado; (2) FT parcial de `features[7:]` com LR diferencial (1e-5 no backbone, 1e-3 no head, como no desafio da Aula 1, nb[21]), partindo do head já treinado; (3) avaliação no teste externo, não no teste atual, que está saturado.
# - **Contraste com a A1.** No NEU Surface Defects (texturas de aço em cinza, fora do ImageNet), o FE deve ficar bem abaixo do que se viu aqui. É lá que o eixo "domínio" da tabela acima deve pesar.

# %% [markdown]
# ## 10. Métricas finais (JSON)

# %%
metrics = {
    "activity": "A3",
    "dataset": "pavansanagapati/images-dataset",
    "model": "efficientnet_b0 IMAGENET1K_V1, backbone congelado, head Linear(1280, 7)",
    "n_images_clean": int(len(df)),
    "n_corrupted": int(len(corrupted)),
    "n_duplicates_removed": int(len(df_ok) - len(df)),
    "split": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    "imbalance_ratio": round(float(imbalance_ratio), 3),
    "total_params": int(total_params),
    "trainable_params": int(trainable_params),
    "test_accuracy": round(float(test_acc), 2),
    "test_macro_f1": round(float(macro_f1), 4),
    "test_per_class_accuracy": {c: round(float(100 * a), 2) for c, a in zip(CLASS_NAMES, per_class_acc)},
    "best_epoch": int(best_epoch),
    "epochs_run": int(len(hist_df)),
    "best_val_loss": round(float(best_val_loss), 4),
    "best_val_acc": round(float(hist_df.loc[hist_df.epoch == best_epoch, "val_acc"].iloc[0]), 2),
    "near_duplicates": {"method": "imagehash.phash 64 bits", "hamming_threshold": PHASH_THRESHOLD,
                        "n_pairs": int(len(near_dup_df)), "n_pairs_diff_class": int((~near_dup_df.same_class).sum()),
                        "pairs_by_partition": near_dup_counts, "n_test_with_train_near_dup": len(test_leaky_paths)},
    "test_clean": {"n": int(clean.sum()), "accuracy": round(float(test_acc_clean), 2), "macro_f1": round(float(macro_f1_clean), 4)},
    "alpha": alpha_summary,
    "eda": eda_summary,
    "train_time_s": round(train_time_s, 1),
    "peak_vram_train_mb": round(peak_vram_train_mb) if peak_vram_train_mb else None,
    "peak_vram_train_reserved_mb": round(peak_vram_train_reserved_mb) if peak_vram_train_reserved_mb else None,
    "benchmark_vram": bench_rows,
    "ram_after_train": ram_after_train,
    **ram_stats(),
    "pip_time_s": round(PIP_TIME_S, 1),
    "download_time_s": round(DOWNLOAD_TIME_S, 1),
    "notebook_time_s": round(time.time() - NOTEBOOK_T0, 1),
    "notebook_time_scope": "da 1a célula de código (pip) até esta célula; inclui pip, montagem do Drive e download",
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
}
(OUT_DIR / "A3_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
print(json.dumps(metrics, indent=2, ensure_ascii=False))
