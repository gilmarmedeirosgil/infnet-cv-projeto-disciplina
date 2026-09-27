# %% [markdown]
# # A3 — Classificador de imagens com CNN pré-treinada (feature extraction)
# **Visão Computacional com CNNs e Transformers** · Faculdade Infnet — Pós-Graduação · Gilmar Oliveira de Medeiros
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gilmarmedeirosgil/infnet-cv-projeto-disciplina/blob/main/notebooks/A3_cnn_kaggle.ipynb)
#
# **Objetivo.** Classificar as 7 classes do dataset Kaggle `pavansanagapati/images-dataset` (bike, cars, cats, dogs, flowers, horses, human) com uma CNN pré-treinada no ImageNet usada como **extratora de features**: backbone congelado e apenas um novo head linear treinado.
#
# **Requisitos de execução (Colab, runtime T4)** — estimativas, a atualizar após o "Executar tudo":
#
# | Recurso | Estimativa | Observação |
# |---|---|---|
# | RAM | ~3 GB | ~1.800 imagens decodificadas sob demanda (não ficam em memória) |
# | VRAM (treino real) | < 1 GB | backbone congelado: o autograd não guarda ativações do backbone |
# | VRAM (pico do notebook) | ~8–11 GB | benchmark sintético de *fine-tuning* completo (seção 4), batch 64, fp32 |
# | Disco | ~0,5 GB | dataset + pesos da EfficientNet-B0 (~21 MB) |
# | Tempo total | ~6–10 min | download ~1 min, varredura ~0,5 min, benchmark ~1 min, treino (≤15 épocas) ~2–4 min |
#
# **Mapa da rubrica:** 1.1 → seção 5 · 1.2 → seções 6–7 · 1.3 → seção 8 · 1.4 → seção 9 · 1.5 → seção 4.
#
# Estrutura: Problema → Decisão técnica → Código → Resultado → Análise. Figuras usadas no relatório são salvas em PNG em `MyDrive/infnet_cv_projeto/outputs/A3/`.

# %% [markdown]
# ## 1. Setup
# Seed 42 em `random`, `numpy` e `torch` (padrão da Aula 1), checagem da GPU, montagem do Drive para persistir os artefatos e leitura dos Secrets do Kaggle.

# %%
!pip install -q kagglehub

# %%
import os, gc, copy, json, time, random, hashlib
from pathlib import Path

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
print(f"PyTorch {torch.__version__} | torchvision {torchvision.__version__} | device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("ATENÇÃO: sem GPU. No Colab: Ambiente de execução > Alterar tipo > T4 GPU.")

C_BLUE, C_CYAN, C_RED, C_GREEN = "#0A345D", "#1BB5D8", "#DC2626", "#15803D"
plt.rcParams.update({"axes.grid": True, "grid.linestyle": ":", "figure.dpi": 110})

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

def savefig(name):
    plt.savefig(OUT_DIR / f"A3_{name}.png", dpi=150, bbox_inches="tight")

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
DATASET_PATH = Path(kagglehub.dataset_download("pavansanagapati/images-dataset"))
print("Dataset em:", DATASET_PATH)

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
fig, axes = plt.subplots(NUM_CLASSES, N_PER_CLASS, figsize=(2 * N_PER_CLASS, 2 * NUM_CLASSES))
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
savefig("samples_grid")
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
# <!-- ANALISE: comentar (1) modos de cor encontrados e quantas imagens precisaram de conversão/composição de alpha; (2) corrompidas e duplicatas removidas; (3) variação de tamanho/razão de aspecto e o efeito do resize 256 + center crop 224 (objetos largos, como bike e cars, podem perder as extremidades); (4) desbalanceamento: razão ~2:1 (cars/bike vs cats/dogs/horses/human). É moderado: o split estratificado preserva a proporção em train/val/test e a avaliação usa accuracy por classe e macro-F1, que não escondem classes minoritárias. Não foram usados pesos de classe, porque a razão é pequena e todas as classes têm ≥ ~140 imagens de treino; se as classes menores tiverem recall pior, pesos de classe seriam o próximo passo. -->

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
# ## 4. Escolha do modelo — **Rubrica 1.5**
# **Candidatos:** EfficientNet-B0 (slide 14 da Aula 1, Tabela 12-3 do TorchVision) e ResNet-50 (padrão da indústria, mesma família da ResNet-34 usada em aula).
#
# **Decisão: EfficientNet-B0 (`IMAGENET1K_V1`).** Argumentos:
# 1. **Custo:** ~5,3 M parâmetros e ~0,39 GFLOPs por imagem contra ~25,6 M e ~4,1 GFLOPs da ResNet-50 (~10× menos computação). A tabela abaixo lê esses números de `weights.meta`, sem baixar os pesos.
# 2. **VRAM na T4 (15 GB):** o slide 15 mostra que, no treino da ResNet-50 com batch 64, as **ativações ocupam ~84% da VRAM** (~7,2 de ~8,5 GB). Com o backbone congelado, o autograd não guarda as ativações do backbone e esse termo praticamente desaparece. O benchmark abaixo mede o pico de VRAM e o tempo por passo das duas redes, congeladas (*feature extraction*) e em *fine-tuning* completo, para mostrar a folga em números.
# 3. **Nº de classes e tamanho do dataset:** com 7 classes, o head da EfficientNet-B0 tem 1280·7 + 7 = **8.967** parâmetros treináveis (o da ResNet-50 teria 2048·7 + 7 = 14.343). Com ~1.260 imagens de treino, um head pequeno sobre features de 1280 dimensões basta; um backbone maior não traz ganho proporcional e aumenta custo e latência.
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
           "VRAM pico (MB)": round(torch.cuda.max_memory_allocated() / 2**20)}
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
# <!-- ANALISE: usar a tabela de meta + benchmark: razão de VRAM FE vs FT em cada rede, ms/passo, e quanto de batch caberia na T4. Confirmar que a EfficientNet-B0 congelada treina com fração mínima da VRAM, e que mesmo o FT completo da ResNet-50 cabe na T4 com batch 64 (mas não é necessário para este dataset). -->

# %% [markdown]
# ## 5. Feature extraction: backbone congelado + novo head — **Rubrica 1.1**
# **Pré-processamento.** `weights.transforms()` (Weights Enum API, slide 16, "Regra de Ouro") garante o mesmo tratamento do pré-treino: resize 256 (bicúbico) → center crop 224 → normalização com média e desvio do ImageNet. **Sem augmentation** no treino (decisão D8): o treino é único e a análise de augmentation fica na seção 8.
#
# **Decisão de implementação: forward completo com backbone em `eval()`, em vez de pré-computar as features.** Sem augmentation, as features de cada imagem são idênticas em todas as épocas, então pré-computá-las uma vez seria equivalente e mais rápido. Mantém-se o forward completo porque (i) o modelo treinado é a própria EfficientNet com o head trocado, que pode ser usada direto na inferência; (ii) a checagem de gradientes e a medida de VRAM refletem o cenário real de feature extraction; (iii) o custo é pequeno (~1.260 imagens, 0,4 GFLOPs cada). O risco dessa escolha é esquecer o `backbone.eval()`: em `train()` o BatchNorm atualizaria as médias móveis (mudando o backbone "congelado" sem gradiente) e o Stochastic Depth descartaria blocos aleatoriamente. As asserções abaixo e a comparação do `state_dict` do backbone antes/depois do treino verificam isso.

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
hist_df = pd.DataFrame(history)
print(f"\nMelhor época: {best_epoch} (val loss {best_val_loss:.4f}) | tempo de treino: {train_time_s:.1f}s"
      + (f" | VRAM pico no treino: {peak_vram_train_mb:.0f} MB" if peak_vram_train_mb else ""))

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
# <!-- ANALISE: velocidade de convergência (quantas épocas até platô), gap train-val (overfitting ou não; lembrar que o dropout do head está ativo no train e desligado na val, o que pode deixar a loss de treino acima da de validação), efeito do cosine no fim, época escolhida pelo early stopping. -->

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

test_pred_df = test_df[["path", "cls"]].assign(pred=[CLASS_NAMES[i] for i in y_pred], conf=P.max(1).round(4))
test_pred_df.to_csv(OUT_DIR / "A3_test_predictions.csv", index=False)

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
show = show.head(16)
n_cols = 4
n_rows = max(1, int(np.ceil(len(show) / n_cols)))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
for ax in axes.flat:
    ax.axis("off")
for ax, (_, r) in zip(axes.flat, show.iterrows()):
    ax.imshow(load_rgb(r.path))
    ok = r.cls == r.pred
    ax.set_title(f"real: {r.cls} | pred: {r.pred} ({100 * r.conf:.0f}%)", color=C_GREEN if ok else C_RED, fontweight="bold", fontsize=10)
plt.suptitle(title, fontweight="bold")
plt.tight_layout()
savefig("error_examples")
plt.show()

# %% [markdown]
# **Análise dos resultados no teste**
# <!-- ANALISE: accuracy global e macro-F1; classes com pior accuracy e por quê (tamanho do teste por classe: ~30 imagens nas classes menores, então 1 erro ≈ 3,3 p.p.); pares confundidos (hipótese: cats↔dogs por semelhança visual; horses↔dogs; human em fotos com animais/bicicletas; conteúdo com múltiplos objetos, rótulo de uma única classe); padrões nos erros (imagens com objeto pequeno, cortado pelo center crop, ilustrações/desenhos, erros de rótulo do dataset); erros com confiança alta vs baixa. Relacionar com o desbalanceamento: as classes maiores (cars, bike) têm desempenho melhor? -->

# %% [markdown]
# ## 8. Análise de data augmentation para este domínio — **Rubrica 1.3**
# O treino acima **não** usou augmentation (decisão D8: treino único com `weights.transforms()`, como no código da aula). Esta seção analisa, para cada família de transformação, se ajudaria neste dataset e **para quais classes pode distorcer ou prejudicar**. A figura abaixo só ilustra as transformações em imagens do dataset; nenhuma delas entra no treino.
#
# **Rascunho técnico** (a refinar com os resultados da seção 7):
#
# **(a) Geométrica.**
# - *Flip horizontal* (`RandomHorizontalFlip`): **ajuda** em todas as 7 classes. Bicicletas, carros, animais, flores e pessoas continuam plausíveis espelhados, e não há texto nem assimetria semântica esquerda/direita nas classes. É a augmentation mais segura aqui.
# - *Rotação pequena* (±10–15°): ajuda a lidar com fotos levemente inclinadas. Rotações grandes, *flip vertical* e rotações de 90°/180° **prejudicam** bike, cars, horses e human, que têm orientação canônica pela gravidade (carro de cabeça para baixo não aparece no teste e ensina invariâncias inúteis). Flowers é a exceção: fotos de flores vistas de cima são quase invariantes à rotação.
#
# **(b) Cor.**
# - *Brilho/contraste leves* (`ColorJitter(0.2, 0.2)`): ajudam em todas as classes, simulando iluminação e câmeras diferentes.
# - *Saturação/matiz fortes e grayscale* (`RandomGrayscale`): **prejudicam flowers**, cuja cor saturada é uma das pistas mais fortes contra as demais classes (e contra fundos verdes), e podem prejudicar horses/dogs/cats, em que a cor da pelagem ajuda a separar raças parecidas. Em cars a cor não define a classe, então jitter de matiz ajuda o modelo a não associar "vermelho" a "carro". Se a EDA mostrar imagens originalmente em tons de cinza (modo `L`), um `RandomGrayscale(p≈0.1)` reduz o atalho "cinza → classe X".
#
# **(c) Escala / recorte.**
# - `RandomResizedCrop(224, scale=(0.6, 1.0))` **ajuda**: objetos aparecem em tamanhos muito diferentes (EDA de tamanhos) e o center crop fixo corta objetos largos.
# - Recortes agressivos (o padrão `scale=(0.08, 1.0)`) **prejudicam**: em bike e cars o recorte pode ficar só com uma roda ou um para-choque; em cats/dogs/horses, um recorte só de pelo remove a forma da cabeça, que é o que separa as três classes (é onde se esperam confusões); em human, pode sobrar só a roupa. O rótulo deixa de ser verdadeiro para o recorte.
# - A razão de aspecto do crop também deve ficar próxima de 1 (padrão 3/4–4/3), para não achatar animais e veículos.
#
# **(d) Normalização.** Não é opcional: com o backbone congelado, a entrada **tem** de seguir a média/desvio do ImageNet (`weights.transforms()`), porque os pesos e as *running stats* do BN (fixas em `eval()`) foram calibrados nessa distribuição. Estatísticas do próprio dataset deslocariam a entrada de todas as camadas e degradariam as features de todas as classes. Nenhuma classe é "distorcida" pela normalização correta; o risco está em usar a errada.
#
# **Observação sobre feature extraction.** Com o backbone congelado, a augmentation só age através do head: o backbone não aprende invariâncias novas, então o ganho esperado é menor do que em fine-tuning, e ela impede pré-computar as features (cada época vê features diferentes). Mesmo assim, flip + crop moderado + jitter leve são o pipeline recomendado se houver overfitting do head:
# ```python
# train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
#                       T.RandomHorizontalFlip(), T.RandomRotation(10), T.ColorJitter(0.2, 0.2, 0.1, 0.0),
#                       T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
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
fig, axes = plt.subplots(len(demo_classes), len(AUGS), figsize=(2.2 * len(AUGS), 2.4 * len(demo_classes)), squeeze=False)
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
savefig("augmentation_examples")
plt.show()

# %% [markdown]
# <!-- ANALISE: ligar as confusões observadas na seção 7 às augmentations: p.ex., se cats↔dogs domina, crops agressivos piorariam; se os erros envolvem imagens pequenas/cortadas, RandomResizedCrop moderado ajudaria; se há imagens em modo L, citar RandomGrayscale leve. Fechar com as 3+ estratégias recomendadas e as que foram descartadas, com a classe afetada. -->

# %% [markdown]
# ## 9. Feature extraction vs fine-tuning — **Rubrica 1.4**
# **Rascunho técnico** (a refinar com os resultados):
#
# A escolha depende de dois eixos: **tamanho do dataset** e **distância entre o domínio e o ImageNet**.
#
# | | Domínio próximo do ImageNet | Domínio distante (médico, satélite, texturas industriais) |
# |---|---|---|
# | **Dataset pequeno** | **Feature extraction** (este caso) | Fine-tuning parcial das últimas camadas, LR baixa, augmentation forte; ou backbone pré-treinado no domínio (MONAI, TorchGeo) |
# | **Dataset grande** | Fine-tuning parcial ou completo (ganho marginal sobre FE) | Fine-tuning completo |
#
# **Neste dataset, feature extraction é a escolha certa:**
# 1. **Domínio:** as 7 classes são objetos do cotidiano fotografados, amplamente cobertos pelo ImageNet (dezenas de raças de cães e gatos, *sorrel* para cavalos, *mountain bike*, *sports car*, *daisy*; pessoas aparecem em milhares de imagens, embora "pessoa" não seja classe do ImageNet-1k). As camadas finais do backbone (semântica de alto nível, slide 17) já separam essas categorias, e um head linear basta.
# 2. **Tamanho:** ~1.260 imagens de treino (~140–290 por classe) para 4 M de parâmetros do backbone: descongelar tudo daria ~450× mais parâmetros do que exemplos e alto risco de overfitting e de *catastrophic forgetting* das features genéricas. O head tem 8.967 parâmetros, ~7 por imagem de treino.
# 3. **Custo:** o benchmark da seção 4 mostra a diferença de VRAM e tempo por passo entre FE e FT completo; em FE o treino inteiro cabe com folga na T4 e converge em poucas épocas.
#
# **Quando o fine-tuning compensaria:** se a accuracy de validação estacionar abaixo do desejado com erros concentrados em classes finas (p.ex. cats↔dogs), o passo seguinte é o **fine-tuning parcial** do último estágio (`features[7:]` na EfficientNet-B0, o equivalente ao `layer4` da ResNet), com LR diferencial (backbone ~1e-5, head ~1e-3, como no desafio da Aula 1, nb[21]) e augmentation da seção 8, sempre partindo do head já treinado (linear probe antes, como recomenda a Aula 6, slide 12). Em domínios distantes, como os defeitos de aço da A1, o FE sozinho tende a falhar, porque as features de objetos do ImageNet não descrevem bem texturas.

# %% [markdown]
# <!-- ANALISE: citar a accuracy obtida só com o head (se já está > 95%, o teto para ganhar com FT é pequeno e não justifica o custo) e os números do benchmark (VRAM FE vs FT). -->

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
    "train_time_s": round(train_time_s, 1),
    "peak_vram_train_mb": round(peak_vram_train_mb) if peak_vram_train_mb else None,
    "benchmark_vram": bench_rows,
    "notebook_time_s": round(time.time() - NOTEBOOK_T0, 1),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
}
(OUT_DIR / "A3_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
print(json.dumps(metrics, indent=2, ensure_ascii=False))
