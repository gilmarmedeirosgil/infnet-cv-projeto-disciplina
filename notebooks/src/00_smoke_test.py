# %% [markdown]
# # 00 — Smoke test do ambiente (Etapa 0)
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gilmarmedeirosgil/infnet-cv-projeto-disciplina/blob/main/notebooks/00_smoke_test.ipynb)
#
# Verifica GPU, Secrets do Kaggle e o download dos 4 datasets do projeto. **Não faz parte da entrega.**
#
# **Requisitos:** runtime T4, ~3 GB de disco, ~5 min (download).

# %%
import os, sys, json, time
from pathlib import Path
import torch

print("Python:", sys.version.split()[0], "| PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NENHUMA")

# %%
from google.colab import userdata

def load_kaggle_credentials():
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        try:
            os.environ[name] = userdata.get(name)
        except Exception:
            pass
    found = [n for n in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN") if os.environ.get(n)]
    assert found, "Nenhum Secret do Kaggle encontrado."
    print("Secrets carregados:", found)

load_kaggle_credentials()

# %%
!pip install -q kagglehub
import kagglehub

DATASETS = {
    "A1_neu": "kaustubhdikshit/neu-surface-defect-database",
    "A2_ads16": "groffo/ads16-dataset",
    "A3_images": "pavansanagapati/images-dataset",
    "A4_covid": "tawsifurrahman/covid19-radiography-database",
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}

def summarize(root, max_depth=4):
    root = Path(root)
    counts, total_bytes = {}, 0
    for p in root.rglob("*"):
        if p.is_file():
            total_bytes += p.stat().st_size
            if p.suffix.lower() in IMG_EXT:
                rel = p.parent.relative_to(root)
                key = "/".join(rel.parts[:max_depth]) or "."
                counts[key] = counts.get(key, 0) + 1
    return counts, total_bytes

report = {}
for key, slug in DATASETS.items():
    t0 = time.time()
    try:
        path = kagglehub.dataset_download(slug)
        counts, size = summarize(path)
        report[key] = {"slug": slug, "path": path, "n_images": sum(counts.values()),
                       "size_mb": round(size / 1e6, 1), "seconds": round(time.time() - t0, 1)}
        print(f"\n=== {key} ({slug}) -> {path}")
        print(f"imagens: {sum(counts.values())} | tamanho: {size/1e6:.1f} MB | {time.time()-t0:.0f}s")
        for folder, n in sorted(counts.items()):
            print(f"  {n:6d}  {folder}")
    except Exception as e:
        report[key] = {"slug": slug, "error": repr(e)}
        print(f"\n=== {key} ({slug}) FALHOU: {e!r}")

# %%
# Arquivos não-imagem do ADS-16 (metadados/CSVs) para entender a estrutura
ads_path = report.get("A2_ads16", {}).get("path")
if ads_path:
    others = [p for p in Path(ads_path).rglob("*") if p.is_file() and p.suffix.lower() not in IMG_EXT]
    for p in others[:30]:
        print(p.relative_to(ads_path), f"{p.stat().st_size/1e3:.1f} KB")

# %%
print(json.dumps(report, indent=2))
