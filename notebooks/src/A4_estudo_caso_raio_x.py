# %% [markdown]
# # A4.1 — Estudo de caso: triagem de COVID-19 em raio-X de tórax com cGAN para *augmentation* sintética
# **Visão Computacional com CNNs e Transformers** · Faculdade Infnet — Pós-Graduação · Gilmar Oliveira de Medeiros
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gilmarmedeirosgil/infnet-cv-projeto-disciplina/blob/main/notebooks/A4_estudo_caso_raio_x.ipynb)
#
# **Cenário.** Um grupo treinou uma ResNet-18 do zero para triar radiografias de tórax em Normal / Pneumonia / COVID-19 com 1.200 imagens na proporção 7:2:1 (840/240/120), split 80/20, SGD com LR fixo 0,01 e 15 épocas, sem augmentation, e relatou só a accuracy (93% no treino, 61% na validação). **Objetivo deste notebook:** (1) diagnosticar os problemas desse projeto e o impacto clínico de cada um; (2) reproduzir o baseline e construir um pipeline corrigido; (3) treinar uma **GAN condicional (cGAN)** para gerar radiografias COVID sintéticas, documentando a instabilidade do treino adversarial e a mitigação; (4) medir, com várias seeds, se os sintéticos melhoram o **recall de COVID** num teste 100% real; (5) propor um plano de melhoria com critério de adoção clínica.
#
# **Requisitos de execução (Colab, runtime T4)** — valores medidos no "Executar tudo" de 28/09/2026 (Tesla T4; torch 2.11.0+cu128, torchvision 0.26.0+cu128):
#
# | Recurso | Medido | Observação |
# |---|---|---|
# | RAM | **~2,6 GB** (RSS) / **~2,8 GB** de pico (`ru_maxrss`) | processo principal; as ~2.700 imagens usadas ficam em 64×64 uint8 (~11 MB), o pico vem do PyTorch/Inception |
# | VRAM (pico no notebook) | **3.274 MB** | maior consumo entre baseline, corrigido, as 2 runs da cGAN e o *sweep*; medido por `max_memory_allocated` |
# | Disco | ~2,5 GB (estimativa) | dataset + pesos (ResNet-18, InceptionV3, AlexNet do LPIPS) + checkpoints no Drive |
# | Tempo total | **809,5 s (~13,5 min)** | pip 18,8 s; download 2,2 s; amostragem/varredura 11,6 s; baseline 18,0 s; corrigido 25,0 s; setup das métricas de GAN 13,3 s; **Run A 127,3 s**; **Run B 156,7 s**; controle real×sintético 7,4 s; *sweep* (3 multiplicadores × 3 seeds) 208,4 s; o resto é EDA e figuras |
#
# **Retomada.** As duas runs da cGAN e cada treino de classificador gravam o resultado no Drive. Se o notebook for reexecutado (por exemplo, após uma desconexão), o que já terminou é carregado e o treino é pulado; a cGAN também grava um checkpoint intermediário a cada 50 épocas. `FORCE_RETRAIN = True` força tudo de novo.
#
# **Mapa da rubrica:** 5.1 → seção 2 · 5.2 → seção 6 · 5.3 → seção 7 · 5.4 → seção 8 · 5.6 → seção 9.
#
# Estrutura: Problema → Decisão técnica → Código → Resultado → Análise. Figuras usadas no relatório são salvas em PNG em `MyDrive/infnet_cv_projeto/outputs/A4/`.

# %% [markdown]
# ## 1. Setup
# Seed 42 em `random`, `numpy` e `torch`, checagem da GPU, configuração central, montagem do Drive (com *fallback* local) e leitura dos Secrets do Kaggle. `torchmetrics[image]` traz o InceptionV3 do `torch-fidelity` (KID/FID) e o LPIPS; se algum deles falhar no Colab, o notebook usa um *fallback* declarado (seção 7.1).

# %%
import time
NOTEBOOK_T0 = time.time()
!pip install -q kagglehub imagehash "torchmetrics[image]"
PIP_TIME_S = time.time() - NOTEBOOK_T0
print(f"pip: {PIP_TIME_S:.1f}s")

# %%
import os, gc, math, json, random, hashlib, resource, warnings
from pathlib import Path
from urllib.parse import urlparse

import psutil
import imagehash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, f1_score, roc_auc_score

SEED = 42

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__} | torchvision {torchvision.__version__} | device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("ATENÇÃO: sem GPU. No Colab: Ambiente de execução > Alterar tipo > T4 GPU.")

C_BLUE, C_CYAN, C_RED, C_GREEN, C_ORANGE, C_GRAY = "#0A345D", "#1BB5D8", "#DC2626", "#15803D", "#EA580C", "#6B7280"
plt.rcParams.update({"axes.grid": True, "grid.linestyle": ":", "figure.dpi": 100})
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")

def ram_stats():
    """RSS atual e pico (ru_maxrss em KB no Linux) do processo, em MB."""
    return {"ram_rss_mb": round(psutil.Process().memory_info().rss / 2**20),
            "ram_maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}

def vram_peak_mb():
    return round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None

def vram_reserved_peak_mb():
    return round(torch.cuda.max_memory_reserved() / 2**20) if torch.cuda.is_available() else None

def reset_vram_peak():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

TIMES = {"pip_s": round(PIP_TIME_S, 1)}
print("RAM:", ram_stats())

# %% [markdown]
# **Configuração.** Todos os hiperparâmetros ficam aqui e vão para o JSON final.
# - **Classes com índices fixos**: `COVID = 0`, `Normal = 1`, `Pneumonia = 2`. "Pneumonia" do cenário = pasta **Viral Pneumonia** do dataset; `Lung_Opacity` é ignorada.
# - **Resolução**: a cGAN gera **64×64 em cinza** (decisão D8). Todos os classificadores (baseline, corrigido e *sweep*) recebem as imagens **reais também reduzidas a 64×64** e depois ampliadas para 224 px pelo **mesmo** pipeline, reais e sintéticas. Assim, "imagem de baixa resolução" não é pista de "sintético = COVID".
# - **Baseline** reproduz o grupo: ResNet-18 sem pré-treino, SGD LR 0,01 fixo, 15 épocas, sem augmentation, sem pesos de classe, split 80/20 **não estratificado**, modelo da última época. O cenário não diz se o SGD tinha *momentum*; usa-se 0,9, o valor padrão dos tutoriais do PyTorch (sem ele a rede do zero mal sai do lugar em 15 épocas e o baseline ficaria artificialmente pior).
# - **Pipeline corrigido**: ResNet-18 **pré-treinada** no ImageNet, split estratificado, CE com **pesos de classe**, augmentation leve, AdamW + *warmup* + cosseno, AMP, seleção da época por **macro-F1 de validação** (desempate pelo recall de COVID) com *early stopping*.
# - **cGAN**: DCGAN condicional da Aula 7 (`nz = 100`, embedding de classe), com **logits + `BCEWithLogitsLoss`**. Run A (ingênua) e Run B (mitigada) diferem **só** nos fatores listados em `RUN_A`/`RUN_B`.

# %%
FORCE_RETRAIN = False       # True: ignora resultados e checkpoints do Drive e treina tudo de novo
USE_DIFFAUG = False         # extra opcional: DiffAugment (translação + brilho/contraste) na Run B
USE_AMP = torch.cuda.is_available()

CLASS_NAMES = ["COVID", "Normal", "Pneumonia"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
COVID, NORMAL, PNEU = 0, 1, 2
SRC_DIRS = {"COVID": "COVID", "Normal": "Normal", "Pneumonia": "Viral Pneumonia"}
PREVALENCE = {"COVID": 0.1, "Normal": 0.7, "Pneumonia": 0.2}      # proporção 7:2:1 do cenário

IMG = 64                    # resolução da cGAN e das imagens guardadas
CLS_INPUT = 224             # entrada da ResNet-18 (64 -> 224, mesmo pipeline para reais e sintéticos)
IMG_CLS_HIRES = 224         # seção 9.1: resolução nativa só para o classificador (reais, sem sintéticos); a cGAN continua em IMG=64
N_DEV = {"COVID": 120, "Normal": 840, "Pneumonia": 240}
N_TEST_PER_CLASS = 200      # teste fixo, balanceado, disjunto do dev
N_REF_PER_CLASS = 300       # referência real para métricas da GAN e controle (fora do dev e do teste)
PHASH_THRESHOLD = 4         # Hamming <= 4 bits (de 64) = mesma radiografia (critério da A3)
VAL_FRACTION = 0.20

BASELINE_CFG = dict(pretrained=False, optimizer="sgd", lr=0.01, momentum=0.9, weight_decay=0.0, epochs=15,
                    batch_size=32, augment=False, class_weights=False, scheduler=None, warmup_epochs=0,
                    select="last", patience=None)
CORRECTED_CFG = dict(pretrained=True, optimizer="adamw", lr=3e-4, momentum=None, weight_decay=0.05, epochs=25,
                     batch_size=64, augment=True, class_weights=True, scheduler="cosine", warmup_epochs=1,
                     select="val_macro_f1", patience=7)
AUG = dict(rot_deg=7.0, zoom=0.10, shift=0.04, brightness=0.10, contrast=0.15)   # sem flips

GAN_CFG = dict(nz=100, embed_dim=16, ngf=64, ndf=64, batch_size=64, epochs=250, betas=(0.5, 0.999),
               class_balance_alpha=0.5, eval_epochs_every=25, ckpt_every=50, n_eval=300, n_fixed=8)
RUN_A = dict(name="runA_naive", label="Run A (ingênua)", spectral_norm=False, d_batchnorm=True,
             real_label=1.0, lr_d=2e-4, lr_g=2e-4, diffaug=False)
RUN_B = dict(name="runB_mitigated", label="Run B (SN + smoothing + TTUR)", spectral_norm=True, d_batchnorm=False,
             real_label=0.9, lr_d=4e-4, lr_g=1e-4, diffaug=USE_DIFFAUG)

SWEEP_MULTIPLIERS = [0, 1, 3]   # nº de sintéticos = k × nº de COVID reais do treino
SWEEP_SEEDS = [42, 43, 44]

# %%
try:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT_DIR = Path("/content/drive/MyDrive/infnet_cv_projeto/outputs/A4")
except Exception as e:
    print(f"Drive indisponível ({e!r}); usando armazenamento local do runtime.")
    OUT_DIR = Path("/content/outputs/A4") if Path("/content").exists() else Path("outputs/A4")
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
print("Artefatos em:", OUT_DIR)
FIGURES = []

def savefig(name, dpi=150, max_kb=1000):
    """Salva o PNG; se passar de max_kb, re-salva com dpi menor (determinístico)."""
    path = OUT_DIR / f"A4_{name}.png"
    for d in sorted({dpi, 100, 80, 60} - {x for x in (100, 80, 60) if x > dpi}, reverse=True):
        plt.savefig(path, dpi=d, bbox_inches="tight")
        if path.stat().st_size / 1024 <= max_kb:
            break
    FIGURES.append(path.name)
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
DATASET_PATH = Path(kagglehub.dataset_download("tawsifurrahman/covid19-radiography-database"))
TIMES["download_s"] = round(time.time() - t0, 1)
print(f"Dataset em: {DATASET_PATH} | download/cache: {TIMES['download_s']}s")

# %% [markdown]
# ## 2. Diagnóstico do projeto anterior — **Rubrica 5.1**
# O relatório do grupo tem uma accuracy de 61% na validação e nenhuma outra métrica. Os problemas abaixo são **metodológicos**: cada um, sozinho, já impediria confiar no número. O impacto é descrito do ponto de vista do paciente e do serviço de saúde. Os marcadores `<!-- ANALISE -->` recebem a confirmação numérica que a reprodução do baseline (seção 5) fornece.
#
# | # | Problema | Por que é um problema | Impacto clínico |
# |---|---|---|---|
# | 1 | **Desbalanceamento 7:2:1 não tratado** (sem pesos de classe, reamostragem ou ajuste de limiar) | A cross-entropy média é dominada por Normal (70% dos exemplos); o jeito mais barato de reduzir a loss é empurrar a fronteira para "Normal" | **Falsos negativos de COVID**: paciente infectado recebe laudo de triagem "normal", não é isolado nem testado, transmite o vírus e perde a janela de tratamento precoce. Em triagem, o erro caro é exatamente esse |
# | 2 | **Split 80/20 sem estratificação e sem teste separado** | Com 120 COVID, a validação tem em média 24 casos e, sem estratificar, pode ter bem menos; cada erro de COVID mexe ~4 p.p. no recall. A mesma validação escolhe o modelo e reporta o resultado (otimismo de seleção) | O número que embasaria a decisão de implantar **não se reproduz**: o hospital decide com um recall que pode não se repetir num teste novo — a seção 5 mede diretamente essa diferença (89,6% na validação aleatória contra 70,5% no teste, ~19 p.p.) |
# | 3 | **Só accuracy global** | Paradoxo da acurácia: prever tudo "Normal" dá **70%** na distribuição 7:2:1, acima dos 61% relatados. Accuracy não diz quantos COVID escaparam nem quantos alarmes falsos houve | Sem **sensibilidade (recall) de COVID**, especificidade, VPP/VPN e matriz de confusão, ninguém sabe o risco de liberar um infectado. É a métrica mandatória em triagem (Aula 7) |
# | 4 | **ResNet-18 sem pré-treino com 1.200 imagens** (~11 M parâmetros, ~12 mil por imagem de treino) | A rede precisa aprender bordas e texturas do zero com poucos exemplos; tende a decorar o treino e a usar atalhos (marcadores, bordas, intensidade global) | Features frágeis: o desempenho cai ao mudar de aparelho, protocolo ou hospital, e a queda é silenciosa |
# | 5 | **Overfitting (93% treino vs 61% validação) sem regularização nem early stopping** | O gap de 32 p.p. mostra memorização; o modelo final é o da última época, não o melhor | O modelo em produção é pior que o melhor que o próprio grupo treinou; erros imprevisíveis em casos fora do padrão |
# | 6 | **Sem data augmentation** | Radiografias variam em posicionamento, inclinação, exposição e contraste; sem augmentation o modelo só vê a variação das 960 imagens de treino | Pouca robustez a variações rotineiras de aquisição (paciente rodado, técnica de exposição diferente) |
# | 7 | **SGD com LR fixo 0,01, sem scheduler, 15 épocas arbitrárias** | Sem critério de parada nem ajuste do passo; o treino pode estar oscilando ou não convergido, e o resultado depende da época em que se parou | Resultado não reprodutível: retreinar com outra seed pode dar outro modelo com outro recall. Não dá para certificar |
# | 8 | **Vazamento por paciente e viés de fonte** (o dataset não tem ID de paciente; as classes vêm de repositórios diferentes) | Radiografias do mesmo paciente em treino e validação inflam a métrica. Se cada classe vem de um hospital ou população diferente (p.ex. pneumonia pediátrica), o modelo aprende a **fonte** (marcadores, texto, colimação, idade), não a doença (*shortcut learning*) | Falha silenciosa em produção: num hospital novo, a pista de fonte desaparece e o desempenho despenca, sem nenhum alarme. Pode também gerar viés contra subgrupos (idade, sexo, equipamento) |
#
# **Problemas adicionais de implantação** (não aparecem no relatório, mas pesam na adoção): nenhuma calibração de probabilidade nem limiar escolhido pelo custo clínico dos erros (FN de COVID ≫ FP), nenhuma validação externa e nenhum plano de humano no loop ou de monitoramento. Esses pontos entram no plano da seção 9.
#
# **Confirmação com o baseline reproduzido (seção 5).** A reprodução com a receita do grupo (SGD, sem pré-treino, sem augmentation, sem pesos de classe, split aleatório 80/20) bate os pontos acima com números:
# - **Paradoxo da acurácia, direto**: prever "Normal" para tudo dá **70,0%** na distribuição enviesada 7:2:1 do conjunto de desenvolvimento — acima dos 61% de validação e abaixo dos 93% de treino citados no relatório do grupo — mas **33,3%** de accuracy balanceada no teste equilibrado (200/200/200) e **0% de recall de COVID**. É a métrica do grupo reproduzida com uma regra que não olha a imagem.
# - **Gap treino×validação e validação não estratificada**: nosso baseline chega a 100% de accuracy no treino e **89,6%** na validação aleatória (24 COVID, 174 Normal, 44 Pneumonia — a mesma proporção enviesada do treino), um número que parece bom. No **teste** (200/200/200, nunca visto), a accuracy despenca para **70,5%** e o recall de COVID é **23,5%** (47 de 200 casos). A validação aleatória, embaralhada na mesma distribuição enviesada, não expôs o problema; só o teste estratificado e balanceado expôs.
# - **Predição dominada por Normal**: no teste, o baseline previu Normal **357** vezes (de 600 imagens, esperado 200) e COVID só **51** vezes — a fronteira de decisão está deslocada para a classe majoritária, exatamente como a Tabela prevê para uma cross-entropy sem pesos de classe.

# %% [markdown]
# ## 3. Dados
# **Fonte.** COVID-19 Radiography Database (Kaggle, `tawsifurrahman/covid19-radiography-database`): `COVID-19_Radiography_Dataset/<classe>/images/`, com COVID 3.616, Normal 10.192, Viral Pneumonia 1.345 e Lung_Opacity 6.012 (ignorada) radiografias PNG 299×299; as pastas `masks/` (segmentação pulmonar) não são usadas.
#
# **Decisões de amostragem (D3/D8).**
# 1. **Deduplicação por MD5 antes de amostrar**: cópias byte a byte são removidas (e hashes presentes em duas classes são descartados por completo).
# 2. **Conjunto de desenvolvimento (dev) = 840 Normal / 240 Pneumonia / 120 COVID**, sorteado com seed 42: é o cenário do enunciado, e é nele que baseline, pipeline corrigido e cGAN são treinados.
# 3. **Teste fixo, disjunto e balanceado: 200 por classe (600).** Balanceado porque o objetivo é ler o **recall de COVID** com precisão: com 200 COVID, o IC de Wilson de 95% tem meia-largura de ~4–5 p.p. para recall ≈ 0,85–0,90; com a proporção 7:2:1 e o mesmo total, seriam só 60 COVID e ~8 p.p. O recall por classe não depende da prevalência; a **precisão** depende, e por isso também é reportada a **VPP de COVID recalculada para a prevalência de 10%** do cenário e uma accuracy ponderada pela prevalência 7:2:1.
# 4. **Referência real para a GAN: 300 por classe**, também disjunta. KID, diversidade, memorização e o controle real vs sintético precisam de radiografias reais que a GAN não viu; usar a validação (24 COVID) ou o teste contaminaria o protocolo. O dataset tem ~3.300 COVID não usados, então isso é barato.
# 5. **Quase-duplicatas (pHash)**: além do MD5, cada imagem sorteada precisa estar a Hamming > 4 bits de **todas** as já aceitas (em qualquer classe e partição). A seleção é gulosa, na ordem dev → teste → referência, e uma candidata rejeitada é trocada pela próxima do sorteio. Assim a mesma radiografia recomprimida ou reescalada não aparece em dev e teste ao mesmo tempo.
# 6. **Limitação declarada:** o dataset **não tem ID de paciente**. O pHash remove cópias da mesma imagem, mas não impede **duas radiografias diferentes do mesmo paciente** (p.ex. exames seriados) em dev e teste. O resultado deve ser lido como otimista nesse aspecto.
#
# Cada imagem aceita é lida uma vez, convertida para cinza e reduzida a **64×64** (bicúbico com antialias do PIL).

# %%
IMG_EXT = {".png", ".jpg", ".jpeg"}
roots = sorted(p for p in DATASET_PATH.rglob("COVID-19_Radiography_Dataset") if p.is_dir())
DATA_ROOT = roots[0] if roots else DATASET_PATH
print("Raiz:", DATA_ROOT, "| subpastas:", sorted(d.name for d in DATA_ROOT.iterdir() if d.is_dir()))

def class_image_dir(src):
    d = DATA_ROOT / src
    return d / "images" if (d / "images").is_dir() else d

t0 = time.time()
records = []
for cls in CLASS_NAMES:
    for p in sorted(class_image_dir(SRC_DIRS[cls]).iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            records.append({"path": str(p), "file": p.name, "cls": cls, "label": CLASS_TO_IDX[cls],
                            "md5": hashlib.md5(p.read_bytes()).hexdigest()})
df_all = pd.DataFrame(records)
md5_classes = df_all.groupby("md5").cls.nunique()
conflicting = set(md5_classes[md5_classes > 1].index)
n_dup_exact = int(df_all.duplicated("md5").sum())
df = df_all[~df_all.md5.isin(conflicting)].drop_duplicates("md5", keep="first").reset_index(drop=True)
counts_raw = df_all.cls.value_counts().reindex(CLASS_NAMES)
counts_dedup = df.cls.value_counts().reindex(CLASS_NAMES)
print(f"MD5 de {len(df_all)} arquivos em {time.time() - t0:.1f}s")
print(f"Duplicatas exatas: {n_dup_exact} | hashes em mais de uma classe (removidos): {len(conflicting)}")
print(pd.DataFrame({"arquivos": counts_raw, "únicos (MD5)": counts_dedup}).to_string())

# %%
def load_gray(path):
    with Image.open(path) as im:
        mode, size = im.mode, im.size
        g = im.convert("L")
    return g, mode, size

def phash_bits(img):
    return np.unpackbits(np.frombuffer(bytes.fromhex(str(imagehash.phash(img))), dtype=np.uint8)).astype(bool)

def greedy_sample(df, quotas, order_seed, threshold, loader):
    """Percorre cada classe numa ordem sorteada e aceita a candidata só se estiver a Hamming > threshold de
    todas as aceitas (qualquer classe/partição). quotas: [(partição, {classe: n}), ...] na ordem de prioridade."""
    rng = np.random.default_rng(order_seed)
    order = {c: rng.permutation(df.index[df.cls == c].to_numpy()) for c in CLASS_NAMES}
    ptr = {c: 0 for c in CLASS_NAMES}
    total = sum(sum(q.values()) for _, q in quotas)
    acc_bits = np.zeros((total, 64), dtype=bool)
    n_acc, chosen, rejected = 0, [], []
    for split, quota in quotas:
        for cls in CLASS_NAMES:
            need = quota.get(cls, 0)
            while need > 0:
                if ptr[cls] >= len(order[cls]):
                    raise RuntimeError(f"Candidatas esgotadas em {cls} ({split})")
                i = order[cls][ptr[cls]]; ptr[cls] += 1
                item = loader(df.at[i, "path"])
                b = item.pop("bits")
                if n_acc:
                    dist = (acc_bits[:n_acc] != b).sum(1)
                    j = int(dist.argmin())
                    if dist[j] <= threshold:
                        rejected.append({"idx": i, "cls": cls, "split": split, "hamming": int(dist[j]),
                                         "match_idx": chosen[j]["idx"], "match_cls": chosen[j]["cls"], "match_split": chosen[j]["split"]})
                        continue
                acc_bits[n_acc] = b; n_acc += 1
                chosen.append({"idx": i, "cls": cls, "split": split, **item})
                need -= 1
    return chosen, rejected

def load_item(path):
    g, mode, size = load_gray(path)
    arr = np.asarray(g.resize((IMG, IMG), Image.Resampling.BICUBIC), dtype=np.uint8)
    return {"bits": phash_bits(g), "arr": arr, "mode": mode, "width": size[0], "height": size[1],
            "mean": float(np.asarray(g, dtype=np.float32).mean())}

QUOTAS = [("dev", N_DEV), ("test", {c: N_TEST_PER_CLASS for c in CLASS_NAMES}),
          ("ref", {c: N_REF_PER_CLASS for c in CLASS_NAMES})]
t0 = time.time()
chosen, rejected = greedy_sample(df, QUOTAS, SEED, PHASH_THRESHOLD, load_item)
TIMES["sampling_s"] = round(time.time() - t0, 1)
arrays = [c.pop("arr") for c in chosen]
sel = pd.DataFrame(chosen)
sel = sel.join(df[["path", "file", "label", "md5"]], on="idx").reset_index(drop=True)
X_np = np.stack(arrays)[:, None]                  # [N, 1, 64, 64] uint8
Y_np = sel.label.to_numpy()
rej_df = pd.DataFrame(rejected)
print(f"Amostragem com pHash em {TIMES['sampling_s']}s | aceitas: {len(sel)} | rejeitadas por quase-duplicata: {len(rej_df)}")
split_counts = pd.crosstab(sel.cls, sel.split).reindex(index=CLASS_NAMES, columns=["dev", "test", "ref"])
print(split_counts.to_string())
if len(rej_df):
    print("\nRejeitadas (classe da candidata × classe da imagem já aceita):\n", pd.crosstab(rej_df.cls, rej_df.match_cls).to_string())
    print("Partição da candidata × partição da aceita:\n", pd.crosstab(rej_df.split, rej_df.match_split).to_string())
print("\nModo de cor:", sel["mode"].value_counts().to_dict(), "| tamanhos:", sel.groupby(["width", "height"]).size().to_dict())
assert not (set(sel[sel.split == "dev"].md5) & set(sel[sel.split == "test"].md5))
sel[["path", "cls", "label", "split", "md5"]].to_csv(OUT_DIR / "A4_selection.csv", index=False)
del arrays

# %% [markdown]
# **Fonte das imagens (viés de aquisição).** O pacote traz planilhas `<classe>.metadata.xlsx` com a URL de origem de cada arquivo. Cruzar a fonte com a classe mostra se "classe" e "hospital/repositório" estão confundidos, o que é o problema 8 da seção 2.

# %%
def source_name(u):
    u = str(u).strip()
    p = urlparse(u)
    if not p.netloc:
        return u[:50] or "desconhecida"
    path = "/".join(s for s in p.path.split("/")[1:3] if s)
    return (p.netloc.replace("www.", "") + ("/" + path if path else ""))[:60]

def load_sources():
    rows = []
    for cls, src in SRC_DIRS.items():
        files = sorted(DATA_ROOT.glob(f"{src}*.metadata.xlsx")) + sorted(DATA_ROOT.glob(f"{src.upper()}*.metadata.xlsx"))
        if not files:
            continue
        meta = pd.read_excel(files[0])
        cols = {str(c).strip().upper(): c for c in meta.columns}
        for a, b in zip(meta[cols["FILE NAME"]], meta[cols["URL"]]):
            rows.append({"key": f"{cls}/{str(a).strip().lower()}", "source": source_name(b)})
    return pd.DataFrame(rows).drop_duplicates("key")

try:
    src_df = load_sources()
    sel["key"] = sel.cls + "/" + sel.file.str.rsplit(".", n=1).str[0].str.lower()
    sel["source"] = sel.key.map(dict(zip(src_df.key, src_df.source))).fillna("desconhecida")
    source_table = pd.crosstab(sel[sel.split.isin(["dev", "test"])].source, sel[sel.split.isin(["dev", "test"])].cls)
    print(source_table.to_string())
    SOURCES = {cls: {s: int(n) for s, n in source_table[cls].items() if n} for cls in source_table.columns}
except Exception as e:
    print(f"Metadados de fonte indisponíveis ({e!r})")
    SOURCES = None

# %%
# fig: data_overview
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
x = np.arange(len(CLASS_NAMES))
for k, (col, color) in enumerate([("dev", C_BLUE), ("test", C_CYAN), ("ref", C_GRAY)]):
    bars = axes[0].bar(x + (k - 1) * 0.27, split_counts[col].values, 0.27, color=color, label=col)
    axes[0].bar_label(bars, fontsize=8)
axes[0].set_xticks(x, CLASS_NAMES)
axes[0].set_ylabel("nº de imagens")
axes[0].set_title("Partições amostradas (dev 7:2:1 · teste e referência balanceados)", fontweight="bold")
axes[0].legend()
axes[1].boxplot([sel[(sel.split == "dev") & (sel.cls == c)]["mean"] for c in CLASS_NAMES], showfliers=True)
axes[1].set_xticks(range(1, 4), CLASS_NAMES)
axes[1].set_ylabel("nível de cinza médio (0–255)")
axes[1].set_title("Brilho médio por imagem (dev, 299 px)", fontweight="bold")
plt.tight_layout()
savefig("data_overview")
plt.show()
BRIGHTNESS = {c: {"mean": round(float(g["mean"].mean()), 1), "std": round(float(g["mean"].std()), 1)}
              for c, g in sel[sel.split == "dev"].groupby("cls")}
print("Brilho médio por classe (dev):", BRIGHTNESS)

# %%
# fig: samples_grid
N_SHOW = 8
rng = np.random.default_rng(SEED)
fig, axes = plt.subplots(2 * len(CLASS_NAMES), N_SHOW, figsize=(1.45 * N_SHOW, 1.55 * 2 * len(CLASS_NAMES)), dpi=80)
for r, cls in enumerate(CLASS_NAMES):
    rows = rng.choice(np.flatnonzero((sel.split == "dev").to_numpy() & (Y_np == CLASS_TO_IDX[cls])), N_SHOW, replace=False)
    for c, i in enumerate(rows):
        full, _, _ = load_gray(sel.path[i])
        for k, (img, tag) in enumerate([(np.asarray(full), "299 px"), (X_np[i, 0], "64 px")]):
            ax = axes[2 * r + k, c]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if c == 0:
                ax.set_ylabel(f"{cls}\n{tag}", fontsize=9, fontweight="bold")
plt.suptitle("Amostras do dev: original (299 px) e a versão 64 px usada por todos os modelos", fontweight="bold")
plt.tight_layout()
savefig("samples_grid", dpi=100)
plt.show()

# %% [markdown]
# **Análise dos dados**
# 1. **Duplicatas.** MD5 exato encontrou **54** duplicatas (removidas), nenhuma cruzando classes (`md5_cross_class = 0`). O pHash (Hamming ≤ 4, mais permissivo, pega recompressões/reescalas da mesma imagem) rejeitou mais **31** pares, **18 em COVID e 13 em Normal**, dos quais **4 cruzando classes ou partições** — ou seja, sem essa checagem extra, até 4 radiografias quase idênticas (a mesma captura salva duas vezes, uma vez rotulada diferente ou uma em cada partição) poderiam aparecer ao mesmo tempo em dev e teste, inflando artificialmente a accuracy medida.
# 2. **Confusão classe × fonte.** A tabela de fontes confirma a suspeita: **Pneumonia vem de uma única fonte** (`paultimothymooney/chest-xray-pneumonia`, 440 imagens, o repositório pediátrico de Guangzhou), enquanto **Normal** vem de duas fontes bem diferentes (RSNA, 893, e o mesmo repositório pediátrico, 147) e **COVID** vem de **6 fontes** (BIMCV 217, Eurorad 23, COVID-CXNet 43, ieee8023 17, ml-workgroup 12, SIRM 8) — cada uma com seu próprio equipamento, faixa etária e marcadores. O modelo pode estar aprendendo a **distinguir a fonte/o aparelho**, não a doença; como o teste vem dos mesmos repositórios do treino, o desempenho medido aqui é um **limite superior** do que se veria com dados de um hospital realmente novo.
# 3. **Brilho médio por classe** (um atalho possível): COVID **139,7 ± 26,7**, Normal **129,8 ± 22,3**, Pneumonia **127,9 ± 18,4** — COVID é, em média, ~10–12 unidades mais claro que as outras duas classes. A diferença é pequena frente ao desvio-padrão de cada classe (não separa sozinha), mas é uma pista de exposição/equipamento que um modelo pode explorar como atalho, coerente com o viés de fonte do item 2.
# 4. **Custo da grade 64 px.** Comparando com as imagens em 299 px na grade acima: a forma do tórax, a silhueta cardíaca, as clavículas e dispositivos grandes (tubos, eletrodos) continuam visíveis; opacidades finas em vidro fosco (a marca radiológica mais discutida do COVID) e textura fina do parênquima se perdem no *blur* do downsampling. Do lado positivo, marcadores de texto sobrepostos (nomes, datas, setas de laudo) ficam ilegíveis em 64 px, o que reduz — sem eliminar — o atalho de texto como fonte de informação de classe.

# %% [markdown]
# ### 3.1 Splits de treino/validação do dev
# - **Pipeline corrigido e cGAN**: split **estratificado** 80/20 → treino 672/192/96 (Normal/Pneumonia/COVID) e validação 168/48/24. A cGAN treina **só** nas 960 imagens de treino: nunca vê validação, teste ou referência.
# - **Baseline do grupo**: split 80/20 **aleatório, sem estratificação** (seed 42), como no relatório. A composição da validação resultante é impressa abaixo.

# %%
dev_idx = np.flatnonzero((sel.split == "dev").to_numpy())
tr_s, va_s = train_test_split(dev_idx, test_size=VAL_FRACTION, stratify=Y_np[dev_idx], random_state=SEED)
tr_r, va_r = train_test_split(dev_idx, test_size=VAL_FRACTION, shuffle=True, random_state=SEED)
test_np = np.flatnonzero((sel.split == "test").to_numpy())
ref_np = np.flatnonzero((sel.split == "ref").to_numpy())

def class_counts(idx):
    return {c: int((Y_np[idx] == i).sum()) for i, c in enumerate(CLASS_NAMES)}

split_summary = {"stratified": {"train": class_counts(tr_s), "val": class_counts(va_s)},
                 "random_baseline": {"train": class_counts(tr_r), "val": class_counts(va_r)},
                 "test": class_counts(test_np), "ref": class_counts(ref_np)}
print(pd.DataFrame({"estrat. treino": class_counts(tr_s), "estrat. val": class_counts(va_s),
                    "aleat. treino": class_counts(tr_r), "aleat. val": class_counts(va_r),
                    "teste": class_counts(test_np), "referência": class_counts(ref_np)}).to_string())
sel["dev_split_stratified"] = ""
sel.loc[tr_s, "dev_split_stratified"], sel.loc[va_s, "dev_split_stratified"] = "train", "val"
sel["dev_split_random"] = ""
sel.loc[tr_r, "dev_split_random"], sel.loc[va_r, "dev_split_random"] = "train", "val"
sel[["path", "cls", "split", "dev_split_stratified", "dev_split_random"]].to_csv(OUT_DIR / "A4_split.csv", index=False)

X_all = torch.from_numpy(X_np).to(device)                        # [N, 1, 64, 64] uint8 na GPU (~11 MB)
Y_all = torch.tensor(Y_np, dtype=torch.long, device=device)
to_t = lambda a: torch.tensor(a, dtype=torch.long, device=device)
SPLITS = {"stratified": (to_t(tr_s), to_t(va_s)), "random": (to_t(tr_r), to_t(va_r))}
test_idx, ref_idx = to_t(test_np), to_t(ref_np)
N_COVID_TRAIN = int((Y_all[SPLITS["stratified"][0]] == COVID).sum())

pred_all_normal = {"dev_accuracy": 100 * N_DEV["Normal"] / sum(N_DEV.values()), "test_balanced_accuracy": 100 / 3,
                   "covid_recall": 0.0}
print(f"\nParadoxo da acurácia: prever tudo 'Normal' dá {pred_all_normal['dev_accuracy']:.0f}% de accuracy no dev (7:2:1) "
      f"e recall de COVID = 0. COVID no treino estratificado: {N_COVID_TRAIN}")

# %% [markdown]
# ## 4. Pipeline de entrada e treino dos classificadores
# **Pipeline na GPU.** As imagens 64×64 uint8 ficam na GPU; cada batch é (i) aumentado em 64 px, se for treino, (ii) ampliado para 224 px (bilinear) e (iii) replicado em 3 canais com a normalização do ImageNet. **Reais e sintéticos passam exatamente pelo mesmo caminho**, inclusive a augmentation (uma fragilidade do notebook da aula era aplicar augmentation só aos reais). O cinza é replicado em 3 canais para aproveitar a `conv1` pré-treinada sem alterá-la (a Aula 7 trocava a `conv1` por uma de 1 canal com a média dos pesos RGB; replicar é equivalente a usar a **soma** dos pesos, e mantém a escala das ativações que o resto da rede espera).
#
# **Augmentation leve e clinicamente plausível** (só no treino do pipeline corrigido):
# - **Rotação pequena (±7°), zoom de até 10% e translação de até 4%**: simulam paciente levemente rodado, distância e centralização diferentes. Bordas preenchidas com replicação (sem faixas pretas artificiais).
# - **Brilho ±10% e contraste ±15%**: simulam técnica de exposição e equipamentos diferentes.
# - **Sem flip horizontal nem vertical**: o flip horizontal põe o coração à direita (dextrocardia simulada) e inverte a lateralidade das lesões; o vertical produz uma anatomia impossível. Nenhum dos dois é uma variação real de aquisição.

# %%
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

def augment_xray(x):
    """x: [B, 1, H, W] em [0, 1]. Rotação/zoom/translação pequenos + brilho/contraste, por imagem, sem flips."""
    B = x.size(0)
    r = lambda: torch.rand(B, device=x.device) * 2 - 1
    ang = r() * math.radians(AUG["rot_deg"])
    s = 1.0 / (1.0 + torch.rand(B, device=x.device) * AUG["zoom"])            # < 1: amostra área menor = zoom in
    theta = torch.stack([torch.stack([torch.cos(ang) * s, -torch.sin(ang) * s, r() * AUG["shift"]], 1),
                         torch.stack([torch.sin(ang) * s, torch.cos(ang) * s, r() * AUG["shift"]], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    m = x.mean(dim=(2, 3), keepdim=True)
    c = 1 + r().view(B, 1, 1, 1) * AUG["contrast"]
    b = 1 + r().view(B, 1, 1, 1) * AUG["brightness"]
    return (((x - m) * c + m) * b).clamp(0, 1)

def prep_cls(x_u8, train=False, augment=False):
    x = x_u8.float() / 255
    if train and augment:
        x = augment_xray(x)
    x = F.interpolate(x, size=(CLS_INPUT, CLS_INPUT), mode="bilinear", align_corners=False)
    return ((x.expand(-1, 3, -1, -1) - IMNET_MEAN) / IMNET_STD).contiguous(memory_format=torch.channels_last)

# %%
# fig: augmentation_examples
torch.manual_seed(SEED)
demo = X_all[to_t(tr_s[:1])].repeat(7, 1, 1, 1).float() / 255
demo_aug = torch.cat([demo[:1], augment_xray(demo[1:])])
fig, axes = plt.subplots(1, 7, figsize=(13, 2.2), dpi=90)
for k, ax in enumerate(axes):
    ax.imshow(demo_aug[k, 0].cpu(), cmap="gray", vmin=0, vmax=1)
    ax.set_title("original" if k == 0 else f"aumento {k}", fontsize=9); ax.axis("off")
plt.suptitle("Augmentation do pipeline corrigido (64 px, sem flips)", fontweight="bold")
plt.tight_layout()
savefig("augmentation_examples", dpi=100)
plt.show()

# %% [markdown]
# **Métricas.** Todas no **teste 100% real** (600 imagens, 200 por classe): accuracy, macro-F1, precisão/recall/F1 por classe, matriz de confusão, recall de COVID com **IC de Wilson de 95%**, especificidade de COVID (COVID vs resto), AUC de COVID (um contra todos), **VPP de COVID na prevalência de 10%** (Bayes a partir de sensibilidade e especificidade) e **accuracy ponderada pela prevalência 7:2:1** (Σ πc · recall_c), comparável à accuracy que o grupo mediria na distribuição original.

# %%
PREV_VEC = np.array([PREVALENCE[c] for c in CLASS_NAMES])

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (center - half, center + half)

def cls_metrics(y_true, y_pred, probs=None):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(3), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(3))
    pos, neg = y_true == COVID, y_true != COVID
    tp, fn = int((y_pred[pos] == COVID).sum()), int((y_pred[pos] != COVID).sum())
    tn = int((y_pred[neg] != COVID).sum())
    sens, spec = tp / max(1, tp + fn), tn / max(1, int(neg.sum()))
    pi = PREVALENCE["COVID"]
    ppv10 = sens * pi / (sens * pi + (1 - spec) * (1 - pi)) if (sens * pi + (1 - spec) * (1 - pi)) > 0 else float("nan")
    out = {"accuracy": float(100 * (y_true == y_pred).mean()), "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=range(3), zero_division=0)),
           "precision": {c: float(v) for c, v in zip(CLASS_NAMES, p)}, "recall": {c: float(v) for c, v in zip(CLASS_NAMES, r)},
           "f1": {c: float(v) for c, v in zip(CLASS_NAMES, f)}, "support": {c: int(v) for c, v in zip(CLASS_NAMES, s)},
           "confusion_matrix": cm.tolist(), "covid_recall": float(sens), "covid_recall_ci95": [float(v) for v in wilson(tp, tp + fn)],
           "covid_tp": tp, "covid_fn": fn, "covid_specificity": float(spec), "covid_ppv_at_prev10": float(ppv10),
           "prevalence_weighted_accuracy": float(100 * (PREV_VEC * r).sum()),
           "pred_distribution": {c: int((y_pred == i).sum()) for i, c in enumerate(CLASS_NAMES)}}
    if probs is not None and len(np.unique(pos)) == 2:
        out["covid_auc"] = float(roc_auc_score(pos, probs[:, COVID]))
    return out

def build_resnet18(pretrained):
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, len(CLASS_NAMES))
    return m.to(device).to(memory_format=torch.channels_last)

@torch.no_grad()
def predict(model, X_u8, batch_size=256):
    model.eval()
    logits = []
    for i in range(0, len(X_u8), batch_size):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
            logits.append(model(prep_cls(X_u8[i:i + batch_size])).float())
    return torch.cat(logits)

def evaluate_cls(model, idx, data_source=None):
    logits = predict(model, (X_all if data_source is None else data_source)[idx])
    y = Y_all[idx]
    probs = logits.softmax(1).cpu().numpy()
    y_true, y_pred = y.cpu().numpy(), probs.argmax(1)
    return {**cls_metrics(y_true, y_pred, probs), "loss": float(F.cross_entropy(logits, y)),
            "y_pred": y_pred.tolist(), "p_covid": np.round(probs[:, COVID], 4).tolist()}

def lr_lambda_factory(total_steps, warmup_steps, min_ratio=0.01):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return f

# %% [markdown]
# **Função de treino comum** (baseline, corrigido e *sweep*). A ordem dos batches vem de um gerador semeado por `seed·1000 + época`. Com `select="last"` o modelo final é o da última época (como o grupo fez); com `select="val_macro_f1"` guarda-se a época de maior macro-F1 de validação (desempate: recall de COVID, depois menor loss) e o treino para após `patience` épocas sem melhora. Os pesos de classe são $w_c = N / (K \cdot n_c)$, calculados **no conjunto de treino efetivo**: quando entram sintéticos COVID, o peso de COVID cai, para não compensar o desbalanceamento duas vezes. O resultado de cada treino (histórico, predições de validação e teste, tempo, VRAM) é gravado no Drive e reaproveitado na reexecução.

# %%
RESULTS = {}

def run_classifier(name, cfg, seed, split="stratified", syn_u8=None, save_weights=False, gan_tag=None, data_source=None):
    """Treina (ou carrega do Drive) um classificador. Retorna (resultado, modelo ou None).
    data_source: tensor [N,1,H,W] alternativo a X_all (mesma indexação); usado pela seção 9.1 para treinar
    com as imagens reais em resolução nativa (224 px) sem tocar na cGAN, que continua em X_all (64 px)."""
    src = X_all if data_source is None else data_source
    path = CKPT_DIR / f"A4_cls_{name}.pt"
    if path.exists() and not FORCE_RETRAIN:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["result"].get("gan_tag") == gan_tag:
            model = None
            if "model" in ck:
                model = build_resnet18(cfg["pretrained"]); model.load_state_dict(ck["model"]); model.eval()
            print(f"[{name}] resultado encontrado no Drive: treino pulado (época {ck['result']['best_epoch']}, "
                  f"recall COVID {ck['result']['test']['covid_recall']:.3f}).")
            RESULTS[name] = ck["result"]
            return ck["result"], model
        print(f"[{name}] resultado no Drive é de outra GAN: retreinando.")
    tr_idx, va_idx = SPLITS[split]
    Xtr, Ytr = src[tr_idx], Y_all[tr_idx]
    n_syn = 0 if syn_u8 is None else len(syn_u8)
    if n_syn:
        Xtr = torch.cat([Xtr, syn_u8.to(device)])
        Ytr = torch.cat([Ytr, torch.full((n_syn,), COVID, dtype=torch.long, device=device)])
    counts = torch.bincount(Ytr, minlength=3).float()
    weight = counts.sum() / (3 * counts) if cfg["class_weights"] else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    reset_vram_peak()
    set_seed(seed)
    model = build_resnet18(cfg["pretrained"])
    if cfg["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n, bs = len(Ytr), cfg["batch_size"]
    steps = math.ceil(n / bs)
    scheduler = (torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(cfg["epochs"] * steps, cfg["warmup_epochs"] * steps))
                 if cfg["scheduler"] == "cosine" else None)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    history, best, bad = [], {"key": None, "epoch": 0, "state": None}, 0
    t_start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1000 + epoch)).to(device)
        tr_loss, tr_correct, seen = 0.0, 0, 0
        for s in range(steps):
            b = order[s * bs:(s + 1) * bs]
            if len(b) < 2:
                continue
            x, y = prep_cls(Xtr[b], train=True, augment=cfg["augment"]), Ytr[b]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
                logits = model(x)
            loss = criterion(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            tr_loss += loss.item() * len(y); tr_correct += int((logits.argmax(1) == y).sum()); seen += len(y)
        va = evaluate_cls(model, va_idx, data_source=src)
        key = (va["macro_f1"], va["covid_recall"], -va["loss"])
        improved = cfg["select"] == "last" or best["key"] is None or key > best["key"]
        if improved:
            best = {"key": key, "epoch": epoch, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": tr_loss / seen,
                        "train_acc": 100 * tr_correct / seen, "val_loss": va["loss"], "val_acc": va["accuracy"],
                        "val_macro_f1": va["macro_f1"], "val_covid_recall": va["covid_recall"]})
        if cfg["patience"] and bad >= cfg["patience"]:
            break
    train_time = time.time() - t_start
    model.load_state_dict(best["state"])
    train_eval = evaluate_cls(model, tr_idx, data_source=src)          # só reais, sem augmentation
    va, te = evaluate_cls(model, va_idx, data_source=src), evaluate_cls(model, test_idx, data_source=src)
    result = {"name": name, "seed": seed, "split": split, "config": cfg, "n_train_real": int(len(tr_idx)), "n_synthetic": n_syn,
              "train_class_counts": {c: int(v) for c, v in zip(CLASS_NAMES, counts.tolist())},
              "class_weights": None if weight is None else [round(float(w), 4) for w in weight],
              "history": history, "best_epoch": best["epoch"], "epochs_run": len(history),
              "train_real_accuracy": train_eval["accuracy"], "val": va, "test": te, "train_time_s": round(train_time, 1),
              "peak_vram_mb": vram_peak_mb(), "peak_vram_reserved_mb": vram_reserved_peak_mb(), "gan_tag": gan_tag}
    ck = {"result": result}
    if save_weights:
        ck["model"] = best["state"]
    torch.save(ck, path)
    RESULTS[name] = result
    print(f"[{name}] {len(history)} épocas ({train_time:.0f}s), melhor {best['epoch']} | val acc {va['accuracy']:.1f}% F1 {va['macro_f1']:.3f} | "
          f"teste acc {te['accuracy']:.1f}% F1 {te['macro_f1']:.3f} recall COVID {te['covid_recall']:.3f} "
          f"[{te['covid_recall_ci95'][0]:.3f}; {te['covid_recall_ci95'][1]:.3f}]")
    del optimizer, scaler
    return result, model

# %% [markdown]
# ## 5. Baseline "como o grupo fez" e pipeline corrigido
# ### 5.1 Baseline reproduzido
# ResNet-18 **sem pré-treino**, SGD (LR 0,01 fixo, momentum 0,9), 15 épocas, batch 32, sem augmentation, CE sem pesos, split aleatório, modelo da última época. O grupo só olharia a **accuracy de validação**; aqui ela aparece ao lado da accuracy de treino (medida em modo `eval`, sem augmentation) e das métricas por classe no teste fixo.

# %%
t0 = time.time()
baseline, _ = run_classifier("baseline_group", BASELINE_CFG, SEED, split="random")
TIMES["baseline_s"] = round(time.time() - t0, 1)
b_te, b_va = baseline["test"], baseline["val"]
print(f"\nO que o grupo veria: treino {baseline['train_real_accuracy']:.1f}% | validação {b_va['accuracy']:.1f}% "
      f"(val com {split_summary['random_baseline']['val']['COVID']} COVID)")
print(f"No teste fixo: accuracy {b_te['accuracy']:.1f}% (ponderada 7:2:1: {b_te['prevalence_weighted_accuracy']:.1f}%) | "
      f"recall por classe {({c: round(v, 3) for c, v in b_te['recall'].items()})} | predições {b_te['pred_distribution']}")

# %% [markdown]
# ### 5.2 Pipeline corrigido
# ResNet-18 **pré-treinada** (ImageNet), split estratificado, pesos de classe, augmentation leve, AdamW (LR 3e-4, weight decay 0,05) com *warmup* de 1 época e cosseno, AMP, até 25 épocas com seleção por macro-F1 de validação e *early stopping* (paciência 7). Este treino (seed 42) é também o ponto "0 sintéticos, seed 42" do experimento da seção 8, e seus pesos são guardados para o teste de condicionamento da cGAN.

# %%
t0 = time.time()
corrected, corrected_model = run_classifier("sweep_m0_s42", CORRECTED_CFG, 42, save_weights=True)
TIMES["corrected_s"] = round(time.time() - t0, 1)

def summary_row(label, r):
    te = r["test"]
    return {"modelo": label, "val acc (%)": round(r["val"]["accuracy"], 1), "teste acc (%)": round(te["accuracy"], 1),
            "acc 7:2:1 (%)": round(te["prevalence_weighted_accuracy"], 1), "macro-F1": round(te["macro_f1"], 3),
            **{f"recall {c}": round(te["recall"][c], 3) for c in CLASS_NAMES},
            "IC95 recall COVID": f"[{te['covid_recall_ci95'][0]:.3f}; {te['covid_recall_ci95'][1]:.3f}]",
            "especif. COVID": round(te["covid_specificity"], 3), "VPP COVID @10%": round(te["covid_ppv_at_prev10"], 3),
            "AUC COVID": round(te.get("covid_auc", float("nan")), 3)}

compare_df = pd.DataFrame([summary_row("baseline do grupo", baseline), summary_row("pipeline corrigido", corrected)])
compare_df.to_csv(OUT_DIR / "A4_baseline_vs_corrected.csv", index=False)
print(compare_df.T.to_string())

# %%
# fig: baseline_vs_corrected
def plot_cm(ax, cm, title):
    cm = np.asarray(cm)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{100 * norm[i, j]:.0f}%", ha="center", va="center", fontsize=9,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(3), CLASS_NAMES); ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("predito"); ax.set_ylabel("real"); ax.set_title(title, fontweight="bold", fontsize=10); ax.grid(False)

fig, axes = plt.subplots(1, 4, figsize=(19, 4.3), gridspec_kw={"width_ratios": [1.2, 1.2, 1, 1]})
hb, hc = pd.DataFrame(baseline["history"]), pd.DataFrame(corrected["history"])
axes[0].plot(hb.epoch, hb.train_acc, "o-", color=C_RED, label="baseline treino")
axes[0].plot(hb.epoch, hb.val_acc, "s--", color=C_RED, label="baseline val")
axes[0].plot(hc.epoch, hc.train_acc, "o-", color=C_BLUE, label="corrigido treino (c/ aug)")
axes[0].plot(hc.epoch, hc.val_acc, "s--", color=C_BLUE, label="corrigido val")
axes[0].set_xlabel("época"); axes[0].set_ylabel("accuracy (%)"); axes[0].set_title("Accuracy por época", fontweight="bold"); axes[0].legend(fontsize=8)
x = np.arange(3)
for k, (r, color, lab) in enumerate([(baseline, C_RED, "baseline"), (corrected, C_BLUE, "corrigido")]):
    rec = [r["test"]["recall"][c] for c in CLASS_NAMES]
    bars = axes[1].bar(x + (k - 0.5) * 0.38, rec, 0.38, color=color, label=lab)
    axes[1].bar_label(bars, fmt="%.2f", fontsize=8)
ci = corrected["test"]["covid_recall_ci95"]
axes[1].errorbar([COVID + 0.19], [corrected["test"]["covid_recall"]], yerr=[[corrected["test"]["covid_recall"] - ci[0]], [ci[1] - corrected["test"]["covid_recall"]]],
                 color="black", capsize=4)
axes[1].axhline(0.9, color=C_GREEN, linestyle=":", label="meta recall 0,90 (Aula 7)")
axes[1].set_xticks(x, CLASS_NAMES); axes[1].set_ylim(0, 1.08); axes[1].set_title("Recall por classe no teste", fontweight="bold"); axes[1].legend(fontsize=8)
plot_cm(axes[2], baseline["test"]["confusion_matrix"], "Baseline do grupo (teste)")
plot_cm(axes[3], corrected["test"]["confusion_matrix"], "Pipeline corrigido (teste)")
plt.tight_layout()
savefig("baseline_vs_corrected")
plt.show()

# %% [markdown]
# **Análise: baseline vs pipeline corrigido**
# 1. **Baseline: o gap que o relatório do grupo escondeu.** Treino 100%, validação aleatória 89,6% — perto o bastante do "bom" para ninguém desconfiar — mas teste 70,5%: um gap de quase 20 p.p. entre a validação (mesma distribuição enviesada do treino) e o teste (balanceado, nunca visto). No teste, o recall de COVID é **23,5%** (47/200) e a distribuição de predições (COVID 51, Normal 357, Pneumonia 192 em 600) mostra o paradoxo da acurácia em ação: o modelo "acerta" 70,5% porque quase sempre chuta Normal, a classe majoritária do treino.
# 2. **Corrigido: os números que faltavam no relatório do grupo.** Recall de COVID no teste **67,5%** (135/200), IC 95% de Wilson **[60,7%; 73,6%]**; especificidade **100%** (nenhum Normal/Pneumonia é confundido com COVID); VPP a 10% de prevalência **100%** — ou seja, nessa prevalência hipotética, **cada alarme positivo de COVID é, em expectativa, um verdadeiro positivo** (zero falsos alarmes a cada verdadeiro, um resultado bom mas que também reflete a especificidade de 100% num teste de apenas 600 imagens, não uma garantia para produção). Comparando os dois modelos: accuracy 70,5%→87,5% (+17 p.p.), macro-F1 0,665→0,874, recall de COVID quase triplica (0,235→0,675).
# 3. **Quais correções pesam mais.** Não foi feita ablação fator a fator — a melhoria é do **pacote completo** (pré-treino ImageNet, augmentation, pesos de classe, AdamW com cosseno e *early stopping* por macro-F1), não de uma correção isolada; a leitura mais defensável é que o pré-treino e os pesos de classe atacam diretamente os dois problemas mais graves do baseline (features do zero com poucos dados, e fronteira deslocada para Normal), enquanto augmentation e *scheduler* contribuem para a estabilidade do treino.
# 4. **Meta da Aula 7 (recall ≥ 0,90).** **Não foi atingida**: 0,675 no teste, e o limite inferior do IC (0,607) fica bem abaixo até de 0,85, quanto mais de 0,90. O pipeline corrigido é uma melhoria grande e real sobre o baseline, mas **não é seguro para triagem clínica** nesse critério — ver seção 9.
# 5. **Custo da grade 64 px.** A resolução baixa (decisão anti-atalho da seção 3, que também reduz o atalho de texto) provavelmente contribui para o teto de 67,5%: como visto na EDA, opacidades finas em vidro fosco — o achado mais discutido do COVID em radiografia — se perdem no downsampling. Não há como isolar, neste experimento, quanto do erro vem da resolução vs da genuína dificuldade/ambiguidade dos casos, mas é uma decisão de projeto com custo mensurável e documentado.

# %% [markdown]
# ## 6. cGAN condicional — **Rubrica 5.2**
# **Arquitetura** (adaptada da `ConditionalGenerator`/`ConditionalDiscriminator` da Aula 7, 3 classes, 64×64×1):
# - **Gerador G(z, y)**: `Embedding(3, 16)` concatenado a z (100) → 5 `ConvTranspose2d(k=4)`: 116→512 (4×4) → 256 (8×8) → 128 (16×16) → 64 (32×32) → 1 (64×64), BatchNorm + ReLU no meio e `Tanh` na saída (imagens normalizadas para [-1, 1]).
# - **Discriminador D(x, y)**: `Embedding(3, 64·64)` vira um mapa espacial do rótulo, concatenado à imagem (2 canais) → 4 `Conv2d(k4, s2)` 2→64→128→256→512 com LeakyReLU 0,2 → `Conv2d(512, 1, k4)` que devolve um **logit** (sem `Sigmoid`).
# - Inicialização DCGAN: convoluções ~ N(0; 0,02), BatchNorm γ ~ N(1; 0,02).
#
# **Loop adversarial** (um passo de D e um de G por batch):
# 1. **Passo de D**: `fake = G(z, y)`; perda `BCE(D(x, y), y_real) + BCE(D(fake.detach(), y), 0)`. O `.detach()` corta o grafo em G: sem ele, `loss_D.backward()` retropropagaria até os pesos de G (gradientes espúrios em G e o dobro de memória).
# 2. **Passo de G, perda não saturante**: `BCE(D(fake, y), 1)`, ou seja, minimizar $-\log D(G(z))$ em vez de $\log(1 - D(G(z)))$. Com $D = \sigma(a)$, $\partial \log(1-D)/\partial a = -D \approx 0$ quando D rejeita as falsas com confiança (início do treino), enquanto $\partial(-\log D)/\partial a = -(1-D) \approx -1$: G continua recebendo gradiente útil. O alvo de G é sempre 1,0 (o *smoothing* só se aplica ao alvo real de D).
# 3. O `fake` do passo de D é reaproveitado (sem `detach`) no passo de G, como na Aula 7; os gradientes que esse passo deixa em D são descartados pelo `zero_grad` do passo seguinte de D.
#
# **Desvio do código da aula: logits + `BCEWithLogitsLoss` em vez de `Sigmoid` + `BCELoss`.** A soma `log σ(a)` é calculada de forma estável (*log-sum-exp*), sem o `log(0) = -inf` que aparece quando o D satura (exatamente o regime da Run A), e a função é segura no `autocast` fp16, enquanto `BCELoss` é bloqueada no AMP por ser numericamente insegura. O modelo é o mesmo: D(x) = σ(logit), que é o que se registra como "D(x)" e "D(G(z))".
#
# **Amostragem das classes no treino da GAN.** O treino tem 672/192/96 imagens. Com sorteio uniforme por imagem, só 10% dos batches seriam COVID; com balanceamento total, cada COVID seria repetida ~7× mais que uma Normal e o D a decoraria mais rápido. Compromisso: probabilidade de sortear a classe c ∝ $n_c^{0,5}$ (≈ 52% Normal, 28% Pneumonia, 20% COVID).

# %%
NZ = GAN_CFG["nz"]

def dcgan_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

class CondGenerator(nn.Module):
    def __init__(self, nz=100, n_classes=3, embed_dim=16, ngf=64):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)
        def up(cin, cout):
            return [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True)]
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz + embed_dim, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            *up(ngf * 8, ngf * 4), *up(ngf * 4, ngf * 2), *up(ngf * 2, ngf),
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False), nn.Tanh())
        self.apply(dcgan_init)

    def forward(self, z, y):
        return self.net(torch.cat([z, self.embed(y)], 1)[:, :, None, None])

class CondDiscriminator(nn.Module):
    def __init__(self, n_classes=3, img_size=64, ndf=64, spectral_norm=False, batchnorm=True):
        super().__init__()
        self.img_size = img_size
        self.embed = nn.Embedding(n_classes, img_size * img_size)
        def conv(cin, cout, k, s, p, bias):
            c = nn.Conv2d(cin, cout, k, s, p, bias=bias)
            nn.init.normal_(c.weight, 0.0, 0.02)       # inicializa antes de parametrizar com SN
            return nn.utils.parametrizations.spectral_norm(c) if spectral_norm else c
        def down(cin, cout, bn):
            return [conv(cin, cout, 4, 2, 1, not bn)] + ([nn.BatchNorm2d(cout)] if bn else []) + [nn.LeakyReLU(0.2, True)]
        self.net = nn.Sequential(*down(2, ndf, False), *down(ndf, ndf * 2, batchnorm), *down(ndf * 2, ndf * 4, batchnorm),
                                 *down(ndf * 4, ndf * 8, batchnorm), conv(ndf * 8, 1, 4, 1, 0, True))
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                dcgan_init(m)

    def forward(self, x, y):
        ymap = self.embed(y).view(-1, 1, self.img_size, self.img_size)
        return self.net(torch.cat([x, ymap], 1)).view(-1)      # logit

def diff_augment(x):
    """DiffAugment (Zhao et al., 2020) para cinza: brilho, contraste e translação de até 1/8, diferenciável; sem flips."""
    B = x.size(0)
    x = x + (torch.rand(B, 1, 1, 1, device=x.device, dtype=x.dtype) - 0.5)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * (torch.rand(B, 1, 1, 1, device=x.device, dtype=x.dtype) + 0.5) + m
    t = (torch.rand(B, 2, device=x.device) * 2 - 1) * 0.25
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1
    theta[:, :, 2] = t
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)

for cfg_run in (RUN_A, RUN_B):
    _G = CondGenerator(NZ, 3, GAN_CFG["embed_dim"], GAN_CFG["ngf"]).to(device)
    _D = CondDiscriminator(3, IMG, GAN_CFG["ndf"], cfg_run["spectral_norm"], cfg_run["d_batchnorm"]).to(device)
    _z, _y = torch.randn(4, NZ, device=device), torch.tensor([0, 1, 2, 0], device=device)
    _x = _G(_z, _y)
    assert _x.shape == (4, 1, IMG, IMG) and _x.min() >= -1 and _x.max() <= 1
    assert _D(_x, _y).shape == (4,)
    print(f"{cfg_run['label']}: G {sum(p.numel() for p in _G.parameters()):,} params | D {sum(p.numel() for p in _D.parameters()):,} params | "
          f"SN no D: {cfg_run['spectral_norm']} | BN no D: {cfg_run['d_batchnorm']}")
del _G, _D, _x

# %% [markdown]
# ## 7. Instabilidade e mitigação: Run A vs Run B — **Rubrica 5.3**
# As duas runs usam a **mesma** arquitetura de G, os mesmos dados, a mesma seed, o mesmo número de épocas (250 × 15 iterações = 3.750 passos) e o mesmo protocolo de medida. Mudam só:
#
# | | Run A (ingênua) | Run B (mitigada, slide 9 da Aula 7) |
# |---|---|---|
# | Normalização do D | BatchNorm (DCGAN padrão) | **Spectral Normalization** em todas as convoluções, **sem BN** (Miyato et al., 2018): limita a constante de Lipschitz do D, que não consegue ficar arbitrariamente confiante |
# | Alvo real do D | 1,0 | **0,9** (*one-sided label smoothing*, Salimans et al., 2016; o alvo falso fica 0) |
# | LR | D = G = 2e-4 | **TTUR**: D 4e-4, G 1e-4 (Heusel et al., 2017) |
# | DiffAugment | não | opcional (`USE_DIFFAUG`) |
#
# **O que se mede, a cada época** (média das 15 iterações; as curvas por iteração também são guardadas): `loss_D`, `loss_G`, **D(x)** = σ(D) médio nas reais e **D(G(z))** = σ(D) médio nas falsas antes (passo de D) e depois (passo de G) da atualização de D. Critério de saúde da Aula 7: perdas oscilando em torno de um equilíbrio, D(x) > 0,6 e D(G(z)) subindo para 0,3–0,5. **Divergência** = `loss_D → 0`, `loss_G` crescendo e D(G(z)) preso em ~0.
#
# **E a cada 25 épocas** (mais as épocas 1, 5 e 10), com G em `eval()` e 300 COVID sintéticas geradas de um ruído fixo:
# - **KID** (Kernel Inception Distance, Bińkowski et al., 2018) contra as 300 COVID **reais da referência** (nunca vistas pela GAN): MMD² não enviesado com kernel polinomial cúbico sobre as features de 2048-d do InceptionV3; 100 subconjuntos de 100. Menor = distribuições mais próximas. É a métrica principal porque não tem o viés de amostra pequena do FID.
# - **Diversidade LPIPS intra-classe**: distância perceptual média entre 200 pares de COVID sintéticas, comparada com o mesmo valor em COVID reais. *Mode collapse* aparece como diversidade despencando para uma fração da real (a "Batch Clone Syndrome" do slide 6).
# - **Diversidade em pixels** (RMS da diferença entre pares), mais barata e sem rede.
# - **Overfitting do D**: σ(D) médio em 256 reais do **treino** menos em 256 reais da **referência**. Um D que decorou o treino dá notas altas só às reais que já viu (heurística de Karras et al., 2020); com 96 COVID de treino esse é o risco central.
# - **Grade de ruído fixo**: os mesmos 8 vetores z para as 3 classes, guardada nessas épocas.
#
# O **FID** é calculado só no fim, e apenas para comparar A e B com o mesmo N: com 300 amostras e 2048 dimensões a covariância é singular e o FID tem viés positivo grande (slide 13).

# %% [markdown]
# ### 7.1 Extrator de features, KID/FID e LPIPS
# InceptionV3 do `torch-fidelity` (via `torchmetrics`), o mesmo das implementações de referência de FID/KID. **Fallback** (se o pacote ou os pesos falharem no Colab): features de 512-d da ResNet-18 do ImageNet; os números deixam de ser comparáveis com a literatura, mas continuam comparáveis entre A e B. LPIPS com AlexNet (`torchmetrics`, entradas ampliadas para 128 px); *fallback*: distância de cosseno nas mesmas features. KID e FID são implementados abaixo em NumPy sobre as features, para usar exatamente o mesmo extrator nos dois.

# %%
def build_feature_extractor():
    try:
        from torchmetrics.image.fid import NoTrainInceptionV3
        net = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).to(device).eval()
        def feats(u8):
            with torch.no_grad():
                return net(u8.expand(-1, 3, -1, -1).contiguous()).double().cpu().numpy()
        feats(torch.zeros(2, 1, IMG, IMG, dtype=torch.uint8, device=device))
        return feats, "InceptionV3 (torch-fidelity, pool3 2048-d)"
    except Exception as e:
        print(f"InceptionV3 indisponível ({e!r}); usando ResNet-18 ImageNet como fallback.")
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Identity()
        net = net.to(device).eval()
        def feats(u8):
            with torch.no_grad():
                return net(prep_cls(u8)).double().cpu().numpy()
        return feats, "ResNet-18 ImageNet (fallback, 512-d)"

def features(u8, batch=100):
    return np.concatenate([FEAT_FN(u8[i:i + batch]) for i in range(0, len(u8), batch)])

def kid(f_real, f_fake, n_subsets=100, subset_size=100, seed=0):
    """KID = MMD² não enviesado, kernel k(x,y) = (x·y/d + 1)^3; média e desvio sobre subconjuntos."""
    rng = np.random.default_rng(seed)
    m = min(subset_size, len(f_real), len(f_fake))
    d = f_real.shape[1]
    vals = []
    for _ in range(n_subsets):
        x = f_real[rng.choice(len(f_real), m, replace=False)]
        y = f_fake[rng.choice(len(f_fake), m, replace=False)]
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))

def fid(f1, f2):
    """FID = |mu1-mu2|² + Tr(S1 + S2 - 2 (S1 S2)^½), com Tr((S1 S2)^½) pelos autovalores de S1^½ S2 S1^½."""
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    w, v = np.linalg.eigh(s1)
    sq1 = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    tr_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh(sq1 @ s2 @ sq1), 0, None)).sum()
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_sqrt)

def build_lpips():
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
        def dist(a_u8, b_u8):
            up = lambda u: F.interpolate(u.float() / 255, size=(128, 128), mode="bilinear", align_corners=False).expand(-1, 3, -1, -1)
            with torch.no_grad():
                v = float(metric(up(a_u8), up(b_u8)))
            metric.reset()
            return v
        dist(torch.zeros(2, 1, IMG, IMG, dtype=torch.uint8, device=device), torch.full((2, 1, IMG, IMG), 255, dtype=torch.uint8, device=device))
        return dist, "LPIPS AlexNet (torchmetrics, 128 px)"
    except Exception as e:
        print(f"LPIPS indisponível ({e!r}); usando distância de cosseno nas features como fallback.")
        def dist(a_u8, b_u8):
            fa, fb = FEAT_FN(a_u8), FEAT_FN(b_u8)
            fa /= np.linalg.norm(fa, axis=1, keepdims=True); fb /= np.linalg.norm(fb, axis=1, keepdims=True)
            return float(1 - (fa * fb).sum(1).mean())
        return dist, "1 - cosseno nas features (fallback)"

def pair_indices(n, n_pairs, seed):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, n_pairs)
    j = (i + rng.integers(1, n, n_pairs)) % n
    return torch.tensor(i, device=device), torch.tensor(j, device=device)

def lpips_diversity(u8, n_pairs=200, seed=0, batch=100):
    i, j = pair_indices(len(u8), n_pairs, seed)
    vals = [LPIPS_FN(u8[i[k:k + batch]], u8[j[k:k + batch]]) * len(i[k:k + batch]) for k in range(0, n_pairs, batch)]
    return float(sum(vals) / n_pairs)

def pixel_diversity(u8):
    x = u8.float().flatten(1) / 255
    d = torch.cdist(x, x) / math.sqrt(x.shape[1])
    n = len(x)
    return float(d.sum() / (n * (n - 1)))

t0 = time.time()
FEAT_FN, FEAT_NAME = build_feature_extractor()
LPIPS_FN, LPIPS_NAME = build_lpips()
REF_BY_CLASS = {c: ref_idx[Y_all[ref_idx] == i] for i, c in enumerate(CLASS_NAMES)}
REF_FEATS = {c: features(X_all[REF_BY_CLASS[c]]) for c in CLASS_NAMES}
REAL_COVID_REF = X_all[REF_BY_CLASS["COVID"]]
REAL_DIV = {"lpips": lpips_diversity(REAL_COVID_REF), "pixel": pixel_diversity(REAL_COVID_REF)}
train_covid = X_all[SPLITS["stratified"][0][Y_all[SPLITS["stratified"][0]] == COVID]]
KID_TRAIN_VS_REF = kid(REF_FEATS["COVID"], features(train_covid))
TIMES["metric_setup_s"] = round(time.time() - t0, 1)
print(f"Features: {FEAT_NAME} | LPIPS: {LPIPS_NAME} | setup {TIMES['metric_setup_s']}s")
print(f"Diversidade das COVID reais (referência): LPIPS {REAL_DIV['lpips']:.4f} | pixel {REAL_DIV['pixel']:.4f}")
print(f"Piso de KID: COVID reais do treino (96) vs referência (300) = {KID_TRAIN_VS_REF[0]:.4f} ± {KID_TRAIN_VS_REF[1]:.4f}")

# %% [markdown]
# **Piso de KID.** O KID entre as 96 COVID reais do treino e as 300 da referência mostra o valor que um gerador "perfeito" (que amostrasse a mesma distribuição dos dados de treino) obteria com este protocolo; valores de KID das runs devem ser lidos contra esse piso, não contra zero.

# %% [markdown]
# ### 7.2 Treino das duas runs

# %%
GAN_TRAIN_IDX = SPLITS["stratified"][0]
_counts = torch.bincount(Y_all[GAN_TRAIN_IDX], minlength=3).float()
GAN_SAMPLE_W = (_counts ** -GAN_CFG["class_balance_alpha"])[Y_all[GAN_TRAIN_IDX]]
_p = (_counts ** (1 - GAN_CFG["class_balance_alpha"])) / (_counts ** (1 - GAN_CFG["class_balance_alpha"])).sum()
print("Probabilidade de cada classe num batch da GAN:", {c: round(float(v), 3) for c, v in zip(CLASS_NAMES, _p)})
ITERS_PER_EPOCH = math.ceil(len(GAN_TRAIN_IDX) / GAN_CFG["batch_size"])
EVAL_EPOCHS = sorted({1, 5, 10} | set(range(GAN_CFG["eval_epochs_every"], GAN_CFG["epochs"] + 1, GAN_CFG["eval_epochs_every"])))
_g = torch.Generator().manual_seed(SEED + 1)
FIXED_Z = torch.randn(GAN_CFG["n_fixed"], NZ, generator=_g).to(device)
EVAL_Z = torch.randn(GAN_CFG["n_eval"], NZ, generator=_g).to(device)
_g = torch.Generator().manual_seed(SEED + 2)
D_PROBE_TRAIN = GAN_TRAIN_IDX[torch.randperm(len(GAN_TRAIN_IDX), generator=_g)[:256].to(device)]
D_PROBE_REF = ref_idx[torch.randperm(len(ref_idx), generator=_g)[:256].to(device)]
to_pm1 = lambda u8: u8.float() / 127.5 - 1
to_u8 = lambda x: ((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)

@torch.no_grad()
def generate(G, cls, n, seed=None, z=None, batch=256):
    """n imagens uint8 [n, 1, 64, 64] da classe cls, com G em eval()."""
    G.eval()
    if z is None:
        z = torch.randn(n, NZ, generator=torch.Generator().manual_seed(seed)).to(device)
    out = [to_u8(G(z[i:i + batch], torch.full((len(z[i:i + batch]),), cls, dtype=torch.long, device=device)))
           for i in range(0, len(z), batch)]
    return torch.cat(out)

@torch.no_grad()
def fixed_grid(G):
    return torch.cat([generate(G, c, 0, z=FIXED_Z) for c in range(3)]).cpu().numpy()     # [3*8, 1, 64, 64]

@torch.no_grad()
def gan_eval(G, D, epoch):
    fake = generate(G, COVID, 0, z=EVAL_Z)
    k_mean, k_std = kid(REF_FEATS["COVID"], features(fake))
    D.eval()
    d_tr = torch.sigmoid(D(to_pm1(X_all[D_PROBE_TRAIN]), Y_all[D_PROBE_TRAIN]).float()).mean().item()
    d_ref = torch.sigmoid(D(to_pm1(X_all[D_PROBE_REF]), Y_all[D_PROBE_REF]).float()).mean().item()
    D.train()
    return {"epoch": epoch, "kid_covid": k_mean, "kid_covid_std": k_std, "lpips_div_covid": lpips_diversity(fake),
            "pixel_div_covid": pixel_diversity(fake), "D_real_train": d_tr, "D_real_ref": d_ref, "D_overfit_gap": d_tr - d_ref}

def train_gan(run):
    """Treina (ou carrega do Drive) uma run da cGAN. Retorna (G com o melhor KID, G final, resultado)."""
    final_path, last_path = CKPT_DIR / f"A4_gan_{run['name']}_final.pt", CKPT_DIR / f"A4_gan_{run['name']}_last.pt"
    build_G = lambda: CondGenerator(NZ, 3, GAN_CFG["embed_dim"], GAN_CFG["ngf"]).to(device)
    if final_path.exists() and not FORCE_RETRAIN:
        ck = torch.load(final_path, map_location="cpu", weights_only=False)
        G_best, G_final = build_G(), build_G()
        G_best.load_state_dict(ck["G_best"]); G_final.load_state_dict(ck["G_final"])
        print(f"[{run['name']}] checkpoint final encontrado: treino pulado (melhor KID na época {ck['result']['best_epoch']}).")
        return G_best, G_final, ck["result"]
    reset_vram_peak()
    set_seed(SEED)
    G = build_G()
    D = CondDiscriminator(3, IMG, GAN_CFG["ndf"], run["spectral_norm"], run["d_batchnorm"]).to(device)
    optD = torch.optim.Adam(D.parameters(), lr=run["lr_d"], betas=GAN_CFG["betas"])
    optG = torch.optim.Adam(G.parameters(), lr=run["lr_g"], betas=GAN_CFG["betas"])
    scalerD, scalerG = torch.amp.GradScaler("cuda", enabled=USE_AMP), torch.amp.GradScaler("cuda", enabled=USE_AMP)
    bce = nn.BCEWithLogitsLoss()
    aug = diff_augment if run["diffaug"] else (lambda t: t)
    bs = GAN_CFG["batch_size"]
    history, iters, evals, grids = [], [], [], {}
    best = {"kid": float("inf"), "epoch": 0, "state": None}
    start, elapsed = 1, 0.0
    if last_path.exists() and not FORCE_RETRAIN:
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"]); optG.load_state_dict(ck["optG"]); optD.load_state_dict(ck["optD"])
        scalerG.load_state_dict(ck["scalerG"]); scalerD.load_state_dict(ck["scalerD"])
        history, iters, evals, grids, best, elapsed = ck["history"], ck["iters"], ck["evals"], ck["grids"], ck["best"], ck["elapsed"]
        torch.set_rng_state(ck["rng_cpu"])
        if torch.cuda.is_available() and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        start = ck["epoch"] + 1
        print(f"[{run['name']}] retomando da época {start}")
    for epoch in range(start, GAN_CFG["epochs"] + 1):
        t0 = time.time()
        G.train(); D.train()
        buf = torch.zeros(ITERS_PER_EPOCH, 6, device=device)
        for it in range(ITERS_PER_EPOCH):
            idx = GAN_TRAIN_IDX[torch.multinomial(GAN_SAMPLE_W, bs, replacement=True)]
            real, y = to_pm1(X_all[idx]), Y_all[idx]
            z = torch.randn(bs, NZ, device=device)
            # passo de D: reais -> real_label, falsas (detach) -> 0
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
                fake = G(z, y)
                d_real = D(aug(real), y).float()
                d_fake = D(aug(fake.detach()), y).float()
            loss_d = bce(d_real, torch.full_like(d_real, run["real_label"])) + bce(d_fake, torch.zeros_like(d_fake))
            optD.zero_grad(set_to_none=True)
            scalerD.scale(loss_d).backward()
            scalerD.step(optD); scalerD.update()
            # passo de G (não saturante): falsas -> 1
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=USE_AMP):
                d_fake_g = D(aug(fake), y).float()
            loss_g = bce(d_fake_g, torch.ones_like(d_fake_g))
            optG.zero_grad(set_to_none=True)
            scalerG.scale(loss_g).backward()
            scalerG.step(optG); scalerG.update()
            buf[it] = torch.stack([loss_d.detach(), loss_g.detach(), torch.sigmoid(d_real).mean().detach(),
                                   torch.sigmoid(d_fake).mean().detach(), torch.sigmoid(d_fake_g).mean().detach(),
                                   (~torch.isfinite(loss_d + loss_g)).float()])
        b = buf.cpu().numpy()
        iters.extend(np.round(b[:, :5], 5).tolist())
        m = np.nanmean(np.where(np.isfinite(b), b, np.nan), axis=0)
        elapsed += time.time() - t0
        history.append({"epoch": epoch, "loss_D": float(m[0]), "loss_G": float(m[1]), "D_x": float(m[2]),
                        "D_G_z1": float(m[3]), "D_G_z2": float(m[4]), "nonfinite_iters": int(b[:, 5].sum())})
        if epoch in EVAL_EPOCHS:
            ev = gan_eval(G, D, epoch)
            evals.append(ev)
            grids[epoch] = fixed_grid(G)
            if ev["kid_covid"] < best["kid"]:
                best = {"kid": ev["kid_covid"], "epoch": epoch, "state": {k: v.detach().cpu().clone() for k, v in G.state_dict().items()}}
            print(f"[{run['name']}] ep {epoch:3d} | loss D {m[0]:.3f} G {m[1]:.3f} | D(x) {m[2]:.3f} D(G(z)) {m[3]:.3f}/{m[4]:.3f} | "
                  f"KID {ev['kid_covid']:.4f} | div LPIPS {ev['lpips_div_covid']:.3f} (real {REAL_DIV['lpips']:.3f}) | "
                  f"gap D {ev['D_overfit_gap']:+.3f} | {elapsed:.0f}s")
        if epoch % GAN_CFG["ckpt_every"] == 0 and epoch < GAN_CFG["epochs"]:
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "optG": optG.state_dict(), "optD": optD.state_dict(),
                        "scalerG": scalerG.state_dict(), "scalerD": scalerD.state_dict(), "history": history, "iters": iters,
                        "evals": evals, "grids": grids, "best": best, "elapsed": elapsed, "epoch": epoch,
                        "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                       last_path)
    result = {"run": run, "history": history, "iters": iters, "evals": evals, "grids": grids, "best_epoch": best["epoch"],
              "best_kid": best["kid"], "train_time_s": round(elapsed, 1), "iterations": len(iters),
              "peak_vram_mb": vram_peak_mb(), "peak_vram_reserved_mb": vram_reserved_peak_mb()}
    G_final = G
    G_best = build_G(); G_best.load_state_dict(best["state"])
    torch.save({"G_best": best["state"], "G_final": G_final.state_dict(), "D_final": D.state_dict(), "result": result}, final_path)
    last_path.unlink(missing_ok=True)
    del D, optD, optG
    return G_best, G_final, result

GAN = {}
for run in (RUN_A, RUN_B):
    t0 = time.time()
    G_best, G_final, res = train_gan(run)
    GAN[run["name"]] = {"G_best": G_best, "G_final": G_final, "result": res, "label": run["label"]}
    TIMES[f"gan_{run['name']}_s"] = round(time.time() - t0, 1)

# %%
# fig: gan_training_curves
fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex="col")
for r, (name, g) in enumerate(GAN.items()):
    h = pd.DataFrame(g["result"]["history"])
    it = np.array(g["result"]["iters"])
    x_it = np.arange(1, len(it) + 1) / ITERS_PER_EPOCH
    ax = axes[r, 0]
    ax.plot(x_it, it[:, 0], color=C_BLUE, alpha=0.15, lw=0.6); ax.plot(x_it, it[:, 1], color=C_ORANGE, alpha=0.15, lw=0.6)
    ax.plot(h.epoch, h.loss_D, color=C_BLUE, label="loss D"); ax.plot(h.epoch, h.loss_G, color=C_ORANGE, label="loss G")
    ax.set_ylabel(g["label"], fontweight="bold"); ax.set_title("Perdas (por iteração e média por época)", fontsize=10); ax.legend(fontsize=8)
    ax = axes[r, 1]
    ax.plot(h.epoch, h.D_x, color=C_GREEN, label="D(x) reais")
    ax.plot(h.epoch, h.D_G_z1, color=C_RED, label="D(G(z)) passo de D")
    ax.plot(h.epoch, h.D_G_z2, color=C_RED, linestyle="--", label="D(G(z)) passo de G")
    ax.axhline(0.5, color=C_GRAY, linestyle=":"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Saídas do discriminador (σ)", fontsize=10); ax.legend(fontsize=8)
    ev = pd.DataFrame(g["result"]["evals"])
    ax = axes[r, 2]
    ax.plot(ev.epoch, ev.D_real_train, "o-", color=C_BLUE, label="D(x) reais do treino")
    ax.plot(ev.epoch, ev.D_real_ref, "s-", color=C_CYAN, label="D(x) reais nunca vistas")
    ax.fill_between(ev.epoch, ev.D_real_ref, ev.D_real_train, color=C_RED, alpha=0.15, label="gap = overfitting do D")
    ax.set_ylim(-0.02, 1.02); ax.set_title("Overfitting do discriminador (D em eval)", fontsize=10); ax.legend(fontsize=8)
for ax in axes[1]:
    ax.set_xlabel("época")
plt.suptitle("cGAN: Run A (ingênua) vs Run B (SN + label smoothing + TTUR)", fontweight="bold")
plt.tight_layout()
savefig("gan_training_curves")
plt.show()

# %%
# fig: gan_quality_curves
fig, axes = plt.subplots(1, 3, figsize=(18, 4.3))
for name, g in GAN.items():
    ev = pd.DataFrame(g["result"]["evals"])
    color = C_RED if name == RUN_A["name"] else C_BLUE
    axes[0].errorbar(ev.epoch, ev.kid_covid, yerr=ev.kid_covid_std, fmt="o-", color=color, capsize=3, label=g["label"])
    axes[1].plot(ev.epoch, ev.lpips_div_covid, "o-", color=color, label=g["label"])
    axes[2].plot(ev.epoch, ev.pixel_div_covid, "o-", color=color, label=g["label"])
axes[0].axhline(KID_TRAIN_VS_REF[0], color=C_GREEN, linestyle=":", label="piso: reais treino vs ref.")
axes[1].axhline(REAL_DIV["lpips"], color=C_GREEN, linestyle=":", label="COVID reais")
axes[2].axhline(REAL_DIV["pixel"], color=C_GREEN, linestyle=":", label="COVID reais")
axes[0].set_yscale("symlog", linthresh=1e-3)
for ax, t in zip(axes, [f"KID COVID vs referência real ({FEAT_NAME.split(' ')[0]}), menor = melhor",
                        f"Diversidade intra-classe COVID ({LPIPS_NAME.split(' (')[0]})", "Diversidade em pixels (RMS entre pares)"]):
    ax.set_title(t, fontsize=10, fontweight="bold"); ax.set_xlabel("época"); ax.legend(fontsize=8)
plt.tight_layout()
savefig("gan_quality_curves")
plt.show()

# %%
def mosaic(u8, ncol, pad=2):
    a = np.asarray(u8)[:, 0]
    n, h, w = a.shape
    nrow = math.ceil(n / ncol)
    out = np.full((nrow * (h + pad) + pad, ncol * (w + pad) + pad), 255, dtype=np.uint8)
    for k in range(n):
        r, c = divmod(k, ncol)
        out[pad + r * (h + pad):pad + r * (h + pad) + h, pad + c * (w + pad):pad + c * (w + pad) + w] = a[k]
    return out

# %%
# fig: gan_fixed_noise_by_epoch
show_epochs = [e for e in [1, 10, 25, 50, 100, 150, 200, 250] if e in EVAL_EPOCHS]
fig, axes = plt.subplots(1, 2, figsize=(14, 1.05 * len(show_epochs) + 1), dpi=90)
nf = GAN_CFG["n_fixed"]
for ax, (name, g) in zip(axes, GAN.items()):
    rows = np.concatenate([g["result"]["grids"][e][COVID * nf:(COVID + 1) * nf] for e in show_epochs])
    ax.imshow(mosaic(rows, nf), cmap="gray", vmin=0, vmax=255)
    step = IMG + 2
    ax.set_yticks([2 + step * k + IMG / 2 for k in range(len(show_epochs))], [f"época {e}" for e in show_epochs])
    ax.set_xticks([]); ax.grid(False); ax.set_title(f"{g['label']}: COVID, mesmos 8 z", fontweight="bold", fontsize=10)
plt.suptitle("Evolução das amostras com ruído fixo (cada coluna = um vetor z)", fontweight="bold")
plt.tight_layout()
savefig("gan_fixed_noise_by_epoch", dpi=110)
plt.show()

# %% [markdown]
# ### 7.3 Métricas finais das runs, memorização e condicionamento
# Para cada run, com G no estado de **menor KID** (seleção feita contra a referência, que não é teste nem validação) e no estado **final**:
# - **KID por classe** (300 sintéticas de cada classe vs 300 reais da referência) e **FID de COVID** (só comparativo, mesmo N);
# - **diversidade** LPIPS e em pixels relativa às reais;
# - **memorização**: para cada COVID sintética, a distância L2 (pixels, [0, 1]) à COVID de **treino** mais próxima, comparada com a mesma distância das COVID reais da referência ao treino. Razão ≪ 1 = o gerador está copiando o treino;
# - **condicionamento**: o classificador corrigido (seção 5.2) classifica 300 sintéticas de cada classe; se G usasse mal o rótulo, "COVID sintético" não seria reconhecido como COVID.

# %%
@torch.no_grad()
def nn_l2(query_u8, bank_u8):
    q, b = query_u8.float().flatten(1) / 255, bank_u8.float().flatten(1) / 255
    d = torch.cdist(q, b) / math.sqrt(q.shape[1])
    v, i = d.min(1)
    return v.cpu().numpy(), i.cpu().numpy()

REF_NN = nn_l2(REAL_COVID_REF, train_covid)[0]

def final_gan_metrics(G):
    out = {"kid_by_class": {}, "conditioning_acc": {}}
    fake_by_class = {c: generate(G, i, GAN_CFG["n_eval"], seed=SEED + 100 + i) for i, c in enumerate(CLASS_NAMES)}
    f_covid = None
    for i, c in enumerate(CLASS_NAMES):
        f = features(fake_by_class[c])
        out["kid_by_class"][c] = kid(REF_FEATS[c], f)[0]
        if c == "COVID":
            f_covid = f
        pred = predict(corrected_model, fake_by_class[c]).argmax(1).cpu().numpy()
        out["conditioning_acc"][c] = float((pred == i).mean())
    fake = fake_by_class["COVID"]
    out["fid_covid"] = fid(REF_FEATS["COVID"], f_covid)
    out["lpips_div_covid"] = lpips_diversity(fake)
    out["lpips_div_ratio"] = out["lpips_div_covid"] / REAL_DIV["lpips"]
    out["pixel_div_ratio"] = pixel_diversity(fake) / REAL_DIV["pixel"]
    d_syn, _ = nn_l2(fake, train_covid)
    out["nn_l2_synth_median"] = float(np.median(d_syn))
    out["nn_l2_real_ref_median"] = float(np.median(REF_NN))
    out["memorization_ratio"] = out["nn_l2_synth_median"] / out["nn_l2_real_ref_median"]
    out["frac_synth_closer_than_min_real"] = float((d_syn < REF_NN.min()).mean())
    return out

GAN_FINAL = {}
for name, g in GAN.items():
    for state in ("G_best", "G_final"):
        GAN_FINAL[f"{name}/{state}"] = final_gan_metrics(g[state])

gan_table = pd.DataFrame([{"run/estado": k, "época": GAN[k.split('/')[0]]["result"]["best_epoch"] if k.endswith("best") else GAN_CFG["epochs"],
                           "KID COVID": v["kid_by_class"]["COVID"], "KID Normal": v["kid_by_class"]["Normal"],
                           "KID Pneumonia": v["kid_by_class"]["Pneumonia"], "FID COVID (comparativo)": v["fid_covid"],
                           "div. LPIPS / real": v["lpips_div_ratio"], "div. pixel / real": v["pixel_div_ratio"],
                           "memorização (NN sint./NN real)": v["memorization_ratio"],
                           **{f"cond. {c}": v["conditioning_acc"][c] for c in CLASS_NAMES}} for k, v in GAN_FINAL.items()])
gan_table.to_csv(OUT_DIR / "A4_gan_final_metrics.csv", index=False)
with pd.option_context("display.float_format", "{:.4f}".format):
    print(gan_table.T.to_string())

# %%
# fig: gan_final_samples
rows = []
for i, c in enumerate(CLASS_NAMES):
    real = X_all[REF_BY_CLASS[c][:8]].cpu().numpy()
    rows += [real] + [generate(GAN[name]["G_best"], i, 8, seed=SEED + 200 + i).cpu().numpy() for name in GAN]
fig, ax = plt.subplots(figsize=(8, 10), dpi=100)
ax.imshow(mosaic(np.concatenate(rows), 8), cmap="gray", vmin=0, vmax=255)
labels = [f"{c}: {tag}" for c in CLASS_NAMES for tag in ["real (ref.)", "Run A", "Run B"]]
ax.set_yticks([2 + (IMG + 2) * k + IMG / 2 for k in range(len(labels))], labels, fontsize=9)
ax.set_xticks([]); ax.grid(False)
ax.set_title("Reais vs sintéticas (G com menor KID de cada run)", fontweight="bold")
plt.tight_layout()
savefig("gan_final_samples", dpi=110)
plt.show()

# %%
# fig: gan_memorization
G_AUG = GAN[RUN_B["name"]]["G_best"]          # gerador usado no experimento da seção 8
fake = generate(G_AUG, COVID, 200, seed=SEED + 300)
d_syn, nn_i = nn_l2(fake, train_covid)
order = np.argsort(d_syn)[:8]                 # as 8 sintéticas MAIS próximas do treino (pior caso)
pairs = np.concatenate([fake[order].cpu().numpy(), train_covid[nn_i[order]].cpu().numpy()])
fig, axes = plt.subplots(1, 2, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
axes[0].imshow(mosaic(pairs, 8), cmap="gray", vmin=0, vmax=255); axes[0].grid(False)
axes[0].set_yticks([2 + IMG / 2, 4 + IMG * 1.5], ["sintética", "vizinha no treino"]); axes[0].set_xticks([])
axes[0].set_title("Run B: as 8 COVID sintéticas mais próximas do treino (pior caso)", fontweight="bold", fontsize=10)
axes[1].hist(REF_NN, bins=30, alpha=0.6, color=C_GREEN, label="real (ref.) → treino", density=True)
axes[1].hist(d_syn, bins=30, alpha=0.6, color=C_BLUE, label="sintética → treino", density=True)
axes[1].set_xlabel("distância L2 (RMS) ao vizinho mais próximo no treino COVID"); axes[1].legend(fontsize=8)
axes[1].set_title("Memorização: distância ao treino", fontweight="bold", fontsize=10)
plt.tight_layout()
savefig("gan_memorization")
plt.show()

# %% [markdown]
# ### 7.4 Teste de controle: real vs sintético
# Se um classificador simples separa com facilidade COVID reais de COVID sintéticas, os sintéticos carregam uma **assinatura** própria (textura, borrão, artefatos de *checkerboard* das convoluções transpostas). No experimento da seção 8 essa assinatura só aparece em imagens rotuladas COVID, então o classificador pode aprender "cara de sintético → COVID", um atalho que não ajuda nas COVID reais. Dois classificadores, com validação cruzada estratificada de 5 *folds* sobre 300 COVID reais da referência + 300 sintéticas da Run B:
# - **regressão logística** sobre as features do extrator (as mesmas do KID);
# - **CNN pequena** (3 convoluções) treinada do zero nos pixels 64×64, que pega artefatos de alta frequência.
#
# Leitura: AUC ≈ 0,5 = indistinguíveis; AUC ≥ 0,9 = **alerta** (separáveis com facilidade).

# %%
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                                 nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1))
    def forward(self, x):
        return self.net(x).view(-1)

def control_real_vs_synth(real_u8, fake_u8, epochs=30, seed=SEED):
    X = torch.cat([real_u8, fake_u8]); y = np.r_[np.zeros(len(real_u8)), np.ones(len(fake_u8))]
    f = features(X)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    auc_lr, auc_cnn = [], []
    for k, (tr, te) in enumerate(skf.split(f, y)):
        sc = StandardScaler().fit(f[tr])
        lr = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(f[tr]), y[tr])
        auc_lr.append(roc_auc_score(y[te], lr.predict_proba(sc.transform(f[te]))[:, 1]))
        set_seed(seed + k)
        net = SmallCNN().to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        xt, yt = to_pm1(X[to_t(tr)]), torch.tensor(y[tr], dtype=torch.float32, device=device)
        for ep in range(epochs):
            perm = torch.randperm(len(tr), device=device)
            for s in range(0, len(tr), 64):
                b = perm[s:s + 64]
                loss = F.binary_cross_entropy_with_logits(net(xt[b]), yt[b])
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.sigmoid(net(to_pm1(X[to_t(te)]))).cpu().numpy()
        auc_cnn.append(roc_auc_score(y[te], p))
    return {"auc_logreg_features_mean": float(np.mean(auc_lr)), "auc_logreg_features_std": float(np.std(auc_lr)),
            "auc_small_cnn_mean": float(np.mean(auc_cnn)), "auc_small_cnn_std": float(np.std(auc_cnn)), "n_per_class": int(len(real_u8))}

t0 = time.time()
CONTROL = control_real_vs_synth(REAL_COVID_REF, generate(G_AUG, COVID, len(REAL_COVID_REF), seed=SEED + 400))
TIMES["control_s"] = round(time.time() - t0, 1)
print(f"Real vs sintético (COVID, Run B): AUC regressão logística {CONTROL['auc_logreg_features_mean']:.3f} ± {CONTROL['auc_logreg_features_std']:.3f} | "
      f"AUC CNN pequena {CONTROL['auc_small_cnn_mean']:.3f} ± {CONTROL['auc_small_cnn_std']:.3f}")

# %% [markdown]
# **Análise da cGAN: instabilidade, mitigação e qualidade**
#
# **Run A: o modo de falha é discriminador dominante com gradiente de gerador evanescente, mais overfitting do D — não colapso total.** Não houve divergência catastrófica (0 iterações não finitas, sem NaN) nem um colapso completo de modo, mas o modo de falha que de fato ocorreu tem nome: **D dominante**. Na época final, $D(x) = 0{,}847$ (perto de 1: o D reconhece as reais com confiança alta) contra $D(G(z_1)) = 0{,}169$ e $D(G(z_2)) = 0{,}036$ (perto de 0: o D rejeita as sintéticas com quase certeza, mesmo logo após o passo do G) — o gradiente que chega ao G nessa região da sigmoide é pequeno (gradiente evanescente), e é por isso que $\text{loss}_G$ termina em **4,40**, bem acima da Run B (1,30). O **gap de overfitting do D** ($D_{\text{real,treino}} - D_{\text{real,ref}}$) cresce de $-0{,}04$ na época 1 para um pico de **0,70** na época 225 (fecha em 0,55 na 250): o D fica cada vez mais confiante nas reais que já viu no treino do que nas reais de referência nunca vistas — um overfitting do próprio D, que reforça o domínio sobre o G. O **KID** (com desvio) melhora até a época 150 (0,288 ± 0,010) e depois **piora** até o fim (0,296 ± 0,010 na 250) — o piso de referência do notebook (KID entre duas amostras de imagens reais, treino vs. referência) é **−0,0003 ± 0,0012** (média ± desvio, ou seja, ~0 como esperado entre duas amostras de reais), então mesmo o melhor ponto da Run A (0,288) fica duas ordens de grandeza acima do que seria "indistinguível de real". A **diversidade** (LPIPS) segue o mesmo arco do KID: sobe até a época 100 (0,337) e cai ~13% até a 250 (0,292), um enfraquecimento real mas não um colapso total para um único modo.
#
# **Run B: mesmo modo de falha, bem mais fraco.** As perdas terminam num patamar mais equilibrado ($\text{loss}_D = 1{,}10$, $\text{loss}_G = 1{,}30$, contra 0,41/4,40 da Run A). Pelo critério da aula ($D(x) > 0{,}6$ e $D(G(z))$ subindo de ~0 para 0,3–0,5), a Run B chega perto mas não cumpre nenhum dos dois: $D(x) = 0{,}550$ fica logo abaixo de 0,6, e $D(G(z_2)) = 0{,}286$ fica logo abaixo de 0,3 — o D ainda vence, mas por uma margem bem menor que na Run A, e sem o gradiente evanescente da Run A (D(G(z)) não fica preso perto de 0, como estava $0{,}036$ na Run A). O **gap de overfitting do D** fica pequeno o treino todo (entre $-0{,}05$ e **0,11**, contra o pico de 0,70 da Run A). O **KID melhora monotonicamente até o fim** (best_kid_epoch = 250, valor **0,264 ± 0,010**, contra 0,296 ± 0,010 da Run A na mesma época) — não há a piora tardia da Run A, embora o piso de referência (KID entre duas amostras de reais, treino vs. referência: **−0,0003 ± 0,0012**, ou seja, ~0 como esperado) mostre que ainda há uma distância grande até "indistinguível". A diversidade (LPIPS 0,307 na época 250) fica na mesma ordem de grandeza da Run A, mas sem o enfraquecimento no fim do treino.
#
# **O que cada mitigação faz.** *Spectral norm* no D limita a constante de Lipschitz de cada camada, impedindo que o D fique arbitrariamente "afiado" perto da fronteira real/fake (a fonte do gradiente quase nulo que trava o G). *Label smoothing* (rótulo real = 0,9 em vez de 1,0) tira do D a confiança extrema, suavizando os gradientes que ele manda para trás. TTUR (LR do D 4× maior que o do G aqui: 4e-4 vs 1e-4) deixa o D acompanhar a evolução do G sem, num único passo, esmagá-lo. **Não foi feita ablação fator a fator** — a melhoria observada (D-overfitting menor, KID melhor e ainda melhorando, sem a piora tardia) é do **pacote das três mitigações juntas**, não atribuível a uma só.
#
# **Tabela final (gerador escolhido para a seção 8: Run B, época 250).**
# - **KID por classe.** Run A: COVID 0,282 · Normal 0,379 · Pneumonia 0,444. Run B: COVID 0,268 · Normal 0,362 · Pneumonia 0,368. A Run B melhora nas 3 classes, com o ganho maior justamente em Pneumonia (−0,076). **COVID não é a classe com pior KID em nenhuma das duas runs** apesar de ser a de menos dados (96 imagens de treino) — Normal e Pneumonia, com mais dados, saem piores; hipótese: COVID tem menos variação interna nas fontes usadas (mais radiografias parecidas entre si), o que facilita o gerador, e não uma vantagem do tamanho da amostra. FID (só comparativo, mesma classe): 261,8 (A) → 250,6 (B), coerente com a melhora do KID.
# - **Condicionamento** (fração das sintéticas que um classificador externo reconhece como a classe pedida). COVID sai muito bem nas duas runs (0,987 → 1,0). **Pneumonia piora bastante com as mitigações** (0,78 → 0,20) e **Normal é o pior caso nas duas runs** (0,0067 e 0,0033 — o classificador quase nunca rotula a sintética de "Normal" como Normal). A leitura direta é que o gerador não captura bem o que torna uma radiografia "normal" (a ausência de opacidade, mais difícil de sintetizar de forma convincente que um padrão positivo), e que a Run B piorou o condicionamento de Pneumonia apesar de melhorar o KID. **Mas há uma explicação alternativa que este notebook não descarta**: como o controle real-vs-sintético logo abaixo dá AUC 1,00 (as sintéticas têm uma "cara" reconhecível), e o classificador usado para medir condicionamento é o pipeline corrigido (que pondera COVID em ~3,3× por causa do desbalanceamento), é possível que esse classificador tenda a rotular qualquer imagem "com cara de sintética" como COVID, inflando o condicionamento de COVID e "roubando" precisão do condicionamento de Normal/Pneumonia — não porque o gerador acerte COVID e erre as outras duas, mas porque o classificador está enviesado para COVID independente da classe pedida. Não foi medida a distribuição completa de predições das sintéticas de Normal/Pneumonia (só a fração que acerta a classe pedida), o que seria necessário para distinguir as duas explicações — fica registrado como limite desta análise, não como conclusão fechada.
# - **Memorização.** A razão NN sintética/NN real fica em **0,97–1,00** nas duas runs (mediana da distância ao vizinho mais próximo do treino, sintética vs real-de-referência) — **perto de 1, não muito menor**, e a fração de sintéticas mais próximas do treino que qualquer imagem real de referência é **0,0** nas duas runs. Não há evidência de memorização/cópia direta do treino; ver também a figura do pior caso (memorização) para inspeção visual dessa conclusão.
# - **Controle real vs. sintético.** AUC de **1,00 ± 0,00** (300 por classe), tanto com uma regressão logística sobre features do Inception quanto com uma CNN pequena treinada direto sobre os pixels — separação **perfeita** entre real e sintético nos dois métodos. É um alerta forte de atalho para a seção 8: qualquer classificador treinado com as duas populações juntas pode aprender a diferenciar "tem cara de sintético" em vez de aprender a doença, o que pesa contra a decisão de usar sintéticos para aumentar o treino.
# - **Em 64 px, o que as sintéticas reproduzem.** Nas grades de amostras finais: a forma geral do tórax, a caixa torácica, e a silhueta cardíaca aparecem de forma reconhecível; o que não aparece de forma confiável são opacidades finas e marcadores de texto/dispositivos — a mesma perda de detalhe discutida na EDA (seção 3) para as imagens reais em 64 px, aqui herdada (e provavelmente amplificada) pelo gerador.

# %% [markdown]
# ## 8. Experimento: com vs sem sintéticos — **Rubrica 5.4**
# **Desenho.** Pipeline corrigido (seção 5.2) treinado com **0, 1× e 3× o número de COVID reais do treino** em sintéticos COVID da **Run B** (G com menor KID), com **3 seeds** (42, 43, 44) em cada nível: 9 treinos. Os sintéticos entram **só no treino**; validação e teste são 100% reais e iguais em todos os treinos. A seed controla a inicialização do head, a ordem dos batches, a augmentation e o sorteio dos sintéticos; o split, a GAN e o teste ficam fixos (a variância da própria GAN não é medida: limitação declarada). Os pesos de classe são recalculados com os sintéticos contados como COVID.
#
# **Estatística.** Para cada nível: média ± desvio-padrão das 3 seeds. Para o efeito dos sintéticos, a comparação é **pareada por seed** (mesma seed com e sem sintéticos): diferença média do recall de COVID com IC de 95% pela t de Student (gl = 2, portanto largo) e, em cada seed, o **teste de McNemar exato** nas 200 COVID do teste (discordantes: acertou sem sintéticos e errou com, e vice-versa).

# %%
GAN_TAG = f"{RUN_B['name']}@{GAN[RUN_B['name']]['result']['best_epoch']}:{float(sum(p.double().sum() for p in G_AUG.parameters())):.6f}"
t0 = time.time()
SWEEP = []
for mult in SWEEP_MULTIPLIERS:
    for seed in SWEEP_SEEDS:
        name = f"sweep_m{mult}_s{seed}"
        n_syn = mult * N_COVID_TRAIN
        if name in RESULTS:
            r = RESULTS[name]
        else:
            syn = generate(G_AUG, COVID, n_syn, seed=10_000 + seed) if n_syn else None
            r, _ = run_classifier(name, CORRECTED_CFG, seed, syn_u8=syn, gan_tag=GAN_TAG if n_syn else None)
        SWEEP.append({"mult": mult, "seed": seed, "n_synthetic": n_syn, "result": r})
TIMES["sweep_s"] = round(time.time() - t0, 1)
SWEEP_PEAK_VRAM = max((s["result"]["peak_vram_mb"] or 0) for s in SWEEP) or None

rows = []
for s in SWEEP:
    te = s["result"]["test"]
    rows.append({"mult": s["mult"], "seed": s["seed"], "n_synthetic": s["n_synthetic"], "best_epoch": s["result"]["best_epoch"],
                 "accuracy": te["accuracy"], "macro_f1": te["macro_f1"], "acc_7:2:1": te["prevalence_weighted_accuracy"],
                 **{f"{m}_{c}": te[m][c] for m in ("precision", "recall", "f1") for c in CLASS_NAMES},
                 "covid_specificity": te["covid_specificity"], "covid_ppv@10%": te["covid_ppv_at_prev10"],
                 "covid_auc": te.get("covid_auc"), "covid_recall_ci_low": te["covid_recall_ci95"][0], "covid_recall_ci_high": te["covid_recall_ci95"][1]})
sweep_df = pd.DataFrame(rows)
sweep_df.to_csv(OUT_DIR / "A4_sweep_per_seed.csv", index=False)
metric_cols = ["accuracy", "macro_f1", "acc_7:2:1"] + [f"{m}_{c}" for m in ("recall", "precision", "f1") for c in CLASS_NAMES] + ["covid_specificity", "covid_ppv@10%", "covid_auc"]
agg = sweep_df.groupby("mult")[metric_cols].agg(["mean", "std"])
agg_fmt = pd.DataFrame({col: [f"{agg.loc[m, (col, 'mean')]:.3f} ± {agg.loc[m, (col, 'std')]:.3f}" for m in agg.index] for col in metric_cols},
                       index=[f"{m}× ({m * N_COVID_TRAIN} sint.)" for m in agg.index]).T
agg_fmt.to_csv(OUT_DIR / "A4_sweep_summary.csv")
print(agg_fmt.to_string())

# %%
STATS = {}
t_crit = stats.t.ppf(0.975, df=len(SWEEP_SEEDS) - 1)
base = {s["seed"]: s["result"]["test"] for s in SWEEP if s["mult"] == 0}
y_test = Y_all[test_idx].cpu().numpy()
covid_mask = y_test == COVID
for mult in [m for m in SWEEP_MULTIPLIERS if m > 0]:
    diffs, mcn = [], []
    for s in [s for s in SWEEP if s["mult"] == mult]:
        te0, te1 = base[s["seed"]], s["result"]["test"]
        diffs.append(te1["covid_recall"] - te0["covid_recall"])
        p0 = np.array(te0["y_pred"])[covid_mask] == COVID
        p1 = np.array(te1["y_pred"])[covid_mask] == COVID
        b, c = int((p0 & ~p1).sum()), int((~p0 & p1).sum())
        pval = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
        mcn.append({"seed": s["seed"], "lost": b, "gained": c, "p_value": pval})
    d = np.array(diffs)
    half = t_crit * d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    STATS[f"{mult}x_vs_0x"] = {"delta_covid_recall_per_seed": d.round(4).tolist(), "mean_delta": float(d.mean()),
                               "ci95_t": [float(d.mean() - half), float(d.mean() + half)], "mcnemar_covid_per_seed": mcn,
                               "delta_macro_f1_mean": float(np.mean([s["result"]["test"]["macro_f1"] - base[s["seed"]]["macro_f1"]
                                                                     for s in SWEEP if s["mult"] == mult]))}
    print(f"{mult}× vs 0×: Δ recall COVID por seed {d.round(3).tolist()} | média {d.mean():+.3f} IC95 t [{d.mean() - half:+.3f}; {d.mean() + half:+.3f}] | "
          f"McNemar (perdidos/ganhos, p): {[(m['lost'], m['gained'], round(m['p_value'], 3)) for m in mcn]}")

# %%
# fig: sweep_covid_metrics
fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
for ax, col, title in zip(axes, ["recall_COVID", "precision_COVID", "f1_COVID", "macro_f1"],
                          ["Recall COVID (sensibilidade)", "Precisão COVID (teste balanceado)", "F1 COVID", "Macro-F1"]):
    for k, seed in enumerate(SWEEP_SEEDS):
        d = sweep_df[sweep_df.seed == seed]
        ax.plot(d.mult + (k - 1) * 0.06, d[col], "o", color=C_GRAY, alpha=0.8, label="seeds" if k == 0 else None)
    m, s = sweep_df.groupby("mult")[col].mean(), sweep_df.groupby("mult")[col].std()
    ax.errorbar(m.index, m.values, yerr=s.values, fmt="s-", color=C_BLUE, capsize=5, lw=2, label="média ± desvio")
    ax.set_xticks(SWEEP_MULTIPLIERS, [f"{k}×\n({k * N_COVID_TRAIN})" for k in SWEEP_MULTIPLIERS])
    ax.set_xlabel("sintéticos COVID no treino"); ax.set_title(title, fontweight="bold", fontsize=10)
    if col == "recall_COVID":
        ax.axhline(0.9, color=C_GREEN, linestyle=":", label="meta 0,90")
    ax.legend(fontsize=8)
plt.suptitle("Efeito dos sintéticos da cGAN no teste 100% real (200 por classe)", fontweight="bold")
plt.tight_layout()
savefig("sweep_covid_metrics")
plt.show()

# %%
# fig: sweep_confusion
fig, axes = plt.subplots(1, len(SWEEP_MULTIPLIERS), figsize=(5.2 * len(SWEEP_MULTIPLIERS), 4.4))
for ax, mult in zip(np.atleast_1d(axes), SWEEP_MULTIPLIERS):
    cm = sum(np.array(s["result"]["test"]["confusion_matrix"]) for s in SWEEP if s["mult"] == mult)
    plot_cm(ax, cm, f"{mult}× sintéticos — soma das {len(SWEEP_SEEDS)} seeds")
plt.tight_layout()
savefig("sweep_confusion")
plt.show()

# %% [markdown]
# **Análise do experimento**
# **O que o experimento mostra, com números.** Com 200 COVID no teste, 1 caso vale 0,5 p.p. — a régua para julgar os deltas abaixo.
# 1. **Recall de COVID médio ± desvio, 3 seeds**: 0× = **72,3% ± 4,5** · 1× = **66,5% ± 3,9** · 3× = **66,3% ± 4,3**. A média **cai** com sintéticos, nas duas doses. A diferença pareada (1× − 0×) tem IC 95% t (gl=2) de **[−22,0; +10,4] p.p.**, e (3× − 0×) de **[−15,4; +3,4] p.p.** — os dois **incluem zero** (não significativos na média de 3 seeds), mas o McNemar por seed mostra que o zero esconde uma tendência real: em **1×**, a seed 43 (p = 0,014, 27 perdidos vs. 11 ganhos) e a seed 44 (p = 0,0016, 34 vs. 12) pioram significativamente; só a seed 42 melhora (+1,5 p.p., não significativo). Em **3×**, a seed 43 piora significativamente (p = 0,0055); as outras duas não têm efeito significativo. Ou seja: **2 de 3 seeds (1×) e 1 de 3 (3×) pioram de forma estatisticamente significativa**; nenhuma seed melhora de forma significativa.
# 2. **Precisão e especificidade não sobem à custa do recall — pelo contrário.** Se os sintéticos deslocassem a fronteira *para* COVID (efeito de baixar o limiar), esperaríamos recall subir e precisão cair. **O oposto aconteceu**: precisão de COVID sobe (98,7%→99,5%→99,0%) enquanto o recall cai (72,3%→66,5%→66,3%); especificidade de COVID já era alta e quase não muda (99,5%→99,8%→99,7%). O recall e a precisão de Normal e Pneumonia também ficam estáveis (Normal: recall 98,5→99,3→99,2%; Pneumonia: recall 94,7→94,7→92,8%). A leitura mais provável não é deslocamento de fronteira, é que os sintéticos **diluíram** o sinal de treino: parte da capacidade do modelo foi para aprender características do COVID **sintético**, que não generalizam para o COVID **real** do teste — coerente com o item 4.
# 3. **Comparado ao que os pesos de classe já fazem.** O ponto 0× **já treina com pesos de classe** (COVID ponderado ~3,3×); é ele quem entrega o ganho grande sobre o baseline sem pesos (seção 5). Os sintéticos, aqui, não têm ganho incremental sobre o que os pesos de classe já entregam — têm um custo (o recall médio cai).
# 4. **Ligação com o controle real vs. sintético (seção 7) e a memorização.** O AUC de 1,00 do classificador real-vs-sintético é a evidência mais direta de que as sintéticas têm uma "cara" reconhecível e diferente das reais — plausivelmente a mesma característica que o classificador de triagem aprende a associar (parcialmente) a COVID quando treinado com sintéticos, sem que isso ajude nas reais do teste. A razão de memorização (~0,97–1,00, seção 7.3) descarta que o problema seja cópia direta do treino: o efeito negativo não vem de "decorar" imagens, vem de um sinal que não transfere.
# 5. **Conclusão honesta.** Neste experimento (3 classes, resolução 64 px, pipeline já corrigido com pesos de classe, 3 seeds), a GAN **não ajudou — e há indícios reais, embora não conclusivos na média, de que atrapalhou** o recall de COVID: o efeito médio é negativo nos dois multiplicadores testados, com 3 de 6 combinações seed×multiplicador estatisticamente significativas, todas na direção de piora. Isso contrasta com o resultado da Aula 7 (+13 p.p., 1 seed): a diferença mais provável não é o pipeline em si, é o **número de seeds** — um único treino pode acertar ou errar por sorte da inicialização/split, e só com 3 seeds (ainda poucas) dá para ver que a direção do efeito é, na média, contrária à da aula. **Recomendação prática**: não adotar estes sintéticos em produção neste ponto (ver critério formal na seção 9).

# %% [markdown]
# ## 9. Plano de melhoria integrado — **Rubrica 5.6**
# Rascunho do plano para levar o sistema de triagem do protótipo à adoção clínica. Os números entre marcadores vêm das seções 5, 7 e 8.
#
# **1. Dados e protocolo de avaliação (a base de tudo)**
# - Coletar dados com **ID de paciente** e fazer o split **por paciente** (nenhum paciente em treino e teste); registrar hospital, equipamento, incidência (PA/AP, leito) e idade.
# - **Teste externo multicêntrico**: pelo menos um hospital que não contribuiu para o treino, com a **prevalência real** do serviço de triagem, rótulo por RT-PCR (não por outro laudo de imagem) e laudo de radiologista cego ao modelo.
# - Auditar **atalhos de fonte** antes de treinar: tabela classe × fonte (seção 3), classificador "adivinhe o hospital", Grad-CAM nas predições corretas e erradas. Neste dataset, a confusão é real: **Pneumonia vem só do repositório pediátrico de Guangzhou** (440 imagens, uma única fonte) e **COVID vem de 6 fontes diferentes**, cada uma com seu equipamento e faixa etária — o modelo pode estar aprendendo a reconhecer a fonte, não a doença.
#
# **2. Modelo**
# - Manter o pipeline corrigido como base (recall de COVID 67,5% e macro-F1 0,874 no teste, contra 23,5% e 0,665 do baseline — ganho grande e real, mas ainda abaixo da meta clínica) e subir a resolução para 224–512 px com a GAN retreinada na mesma resolução (ou sem GAN), já que 64 px apaga opacidades finas.
# - Backbone pré-treinado **no domínio** (p.ex. modelos de raio-X de tórax treinados em CheXpert/MIMIC-CXR, ou auto-supervisionados em radiografias), comparado com o ImageNet no mesmo protocolo.
# - Ensemble de seeds ou *test-time augmentation* leve para reduzir a variância entre treinos (o desvio-padrão do recall de COVID entre as 3 seeds do pipeline corrigido, no sweep, ficou em ±4 a 5 p.p. — grande o bastante para mudar a decisão de aprovar ou não um modelo, se olhado com uma seed só).
#
# **3. Métricas e decisão**
# - Métrica primária: **sensibilidade de COVID** com IC de 95%; secundárias: especificidade, VPP/VPN **na prevalência local**, AUC, macro-F1 e matriz de confusão por classe. Accuracy global só como contexto.
# - **Limiar escolhido pelo custo clínico** na validação (não argmax): fixar sensibilidade-alvo e aceitar a especificidade resultante, porque um falso negativo (infectado liberado) custa muito mais que um falso positivo (teste confirmatório extra).
# - **Calibração** (temperature scaling; ECE e diagrama de confiabilidade), porque o escore será lido como probabilidade pelo clínico.
# - Análise por subgrupo (idade, sexo, hospital, equipamento, AP vs PA) com o mesmo critério mínimo em cada um.
#
# **4. Augmentation sintética: política de uso**
# - Só adotar sintéticos se, em ≥ 5 seeds e com a GAN também variando, o **ganho de sensibilidade de COVID tiver IC acima de zero** sem perda de especificidade maior que uma margem pré-definida. Neste experimento: Δ recall médio de −5,8 p.p. (1×) e −6,0 p.p. (3×), com IC 95% incluindo zero mas a favor da piora nos dois casos — **pelo critério, a GAN não entra no pipeline de produção**.
# - Pré-requisitos a cada nova GAN: KID e diversidade contra reais não vistas, checagem de **memorização** (vizinho mais próximo; nenhuma sintética mais próxima do treino que as reais), **controle real vs sintético** com AUC baixo (aqui deu **1,00** — falhou esse pré-requisito, e é coerente com a GAN não ter ajudado no sweep) e revisão visual por radiologista de uma amostra (alucinações anatômicas).
# - Sintéticos **nunca** em validação ou teste; o teste é sempre 100% real.
# - Alternativas a comparar com o mesmo protocolo: pesos de classe/limiar (já no pipeline), augmentation clássica mais forte, *oversampling*, e modelos de difusão condicionais, que costumam dar mais diversidade que GANs com poucos dados.
#
# **5. Critério de adoção clínica (proposta a validar com o time clínico)**
# - **Sensibilidade de COVID ≥ 0,90 no teste externo, com o limite inferior do IC de 95% ≥ 0,85**; especificidade ≥ 0,80 na prevalência local (limiares ilustrativos, a acordar com a infectologia/radiologia). O tamanho do teste externo sai desse critério (cálculo abaixo).
# - Calibração com ECE ≤ 0,05 e nenhum subgrupo abaixo do critério mínimo.
# - **Humano no loop**: o modelo **prioriza** a fila de laudos e sinaliza suspeitos; não libera paciente sozinho. Casos negativos com sintomas seguem o fluxo clínico normal (teste confirmatório).
# - **Estudo prospectivo silencioso** (modelo roda em paralelo sem influenciar decisões) antes de qualquer uso, comparando com o laudo e o RT-PCR.
# - **Monitoramento pós-implantação**: sensibilidade mensal auditada numa amostra com RT-PCR, *drift* da distribuição de entrada (equipamento, protocolo) e das saídas, gatilho de revisão/retreino e plano de *rollback*. Documentação do modelo (*model card*) e trilha de auditoria, conforme a regulação de software como dispositivo médico (ANVISA/RDC 657/2022).

# %%
def min_n_for_lower_bound(p_true, lower_target, z=1.96, n_max=5000):
    """Menor nº de casos positivos para que o limite inferior de Wilson fique >= lower_target se a sensibilidade observada for p_true."""
    for n in range(10, n_max):
        if wilson(round(p_true * n), n, z)[0] >= lower_target:
            return n
    return None

SAMPLE_SIZE = {f"sens_{p:.2f}_lower_{lo:.2f}": min_n_for_lower_bound(p, lo) for p, lo in [(0.92, 0.85), (0.93, 0.85), (0.95, 0.90)]}
for k, n in SAMPLE_SIZE.items():
    print(f"{k}: {n} casos COVID confirmados no teste externo" + (f" (~{math.ceil(n / PREVALENCE['COVID'])} exames na prevalência de 10%)" if n else ""))

# %% [markdown]
# **Tamanho do teste externo.** Para que o critério "limite inferior do IC ≥ 0,85" seja atingível quando a sensibilidade real é ~0,92, são necessários os números de casos COVID impressos acima; na prevalência de 10% do cenário, isso vira o número de exames indicado. É por isso que o teste deste notebook (200 COVID) mede o recall com meia-largura de ~4–5 p.p., mas o **critério de adoção** precisa ser verificado num teste externo, com pacientes e hospitais independentes.
#
# **Priorização final, guiada pelos resultados.** Com o recall do pipeline corrigido em 67,5% (IC 95% [60,7%; 73,6%]) — abaixo da meta de 0,90 e longe até de 0,85 no limite inferior — e a GAN não trazendo ganho no sweep (item 4 desta seção vira **"não adotar" a GAN nesta configuração**), o gargalo não é validação externa ainda: é fechar a distância até a meta **no próprio teste interno**. Ordem de prioridade: (1) subir a resolução de entrada (item 2 da lista acima, o teto mais provável de estar limitando o recall, pela perda de opacidades finas documentada na seção 3); (2) auditar e mitigar os atalhos de fonte (item 1), porque o ganho medido aqui pode estar parcialmente inflado por viés classe×fonte; (3) reduzir a variância entre seeds (item 2 da lista acima, "Modelo") para que a próxima medição de recall seja confiável; (4) esforço em GAN/sintéticos fica pausado até que (1)–(3) sejam resolvidos — não há por que gastar tempo aperfeiçoando dados sintéticos enquanto o classificador de dados reais ainda está abaixo da meta e o controle real-vs-sintético falha o pré-requisito de AUC baixo. Só depois de bater a meta de sensibilidade no teste interno (com IC acima de 0,85) o próximo passo lógico é o cálculo de amostra acima (92 a 127 casos COVID, conforme a sensibilidade real) para um teste externo de verdade.

# %% [markdown]
# ### 9.1 Verificação da prioridade (1): resolução nativa (224 px) no pipeline real
# A EDA (seção 3) e a análise da seção 5 levantaram a hipótese de que os 64 px — decisão anti-atalho (reduz o atalho de texto sobreposto) e de custo — apagam opacidades finas em vidro fosco e impõem um teto ao recall de COVID. Aqui testamos essa hipótese isolada: **mesmas imagens, mesmo split, mesma arquitetura e hiperparâmetros do pipeline corrigido (seção 5.2), única mudança é a resolução de armazenamento (64 px → 224 px, nativa, sem upsample) — e 3 seeds, para comparar com a mesma robustez estatística do *sweep* da seção 8**. A cGAN **não é retreinada**: continua em 64 px (`IMG`), intocada — subir a resolução dela é uma mudança bem mais cara (GANs escalam mal com resolução) e, com os sintéticos já descartados da produção (seção 8), não é a prioridade agora. `X_all` (64 px, usado pela cGAN/KID/controle) e este novo `X_all_hires` (224 px, só para os treinos desta seção) coexistem sem conflito — mesma indexação (`sel`, `tr_idx`/`va_idx`/`test_idx`), pixels diferentes.

# %%
t0 = time.time()
arrays_hires = []
for p in sel.path:
    g, _, _ = load_gray(p)
    arrays_hires.append(np.asarray(g.resize((IMG_CLS_HIRES, IMG_CLS_HIRES), Image.Resampling.BICUBIC), dtype=np.uint8))
X_np_hires = np.stack(arrays_hires)[:, None]     # [N, 1, 224, 224] uint8 — mesma ordem/index de X_np
del arrays_hires
X_all_hires = torch.from_numpy(X_np_hires).to(device)
TIMES["hires_reload_s"] = round(time.time() - t0, 1)
print(f"Recarregadas {len(sel)} imagens em {IMG_CLS_HIRES}px nativos em {TIMES['hires_reload_s']}s "
      f"({X_all_hires.element_size() * X_all_hires.nelement() / 2**20:.0f} MB na GPU; cGAN e X_all de 64px continuam intocados)")

# %%
HIRES_SEEDS = [42, 43, 44]
hires_runs = {}
for seed in HIRES_SEEDS:
    t0 = time.time()
    r, _ = run_classifier(f"hires224_corrected_s{seed}", CORRECTED_CFG, seed, split="stratified", data_source=X_all_hires)
    hires_runs[seed] = r
    print(f"[hires224, seed {seed}] recall COVID = {r['test']['covid_recall']:.3f} | macro-F1 = {r['test']['macro_f1']:.3f} "
          f"| época {r['best_epoch']}/{r['epochs_run']} | {time.time() - t0:.1f}s | VRAM pico {r['peak_vram_mb']} MB")

hires_recalls = np.array([hires_runs[s]["test"]["covid_recall"] for s in HIRES_SEEDS])
low_recalls = np.array([RESULTS[f"sweep_m0_s{s}"]["test"]["covid_recall"] for s in HIRES_SEEDS])
HIRES_VS_LOWRES = {
    "seeds": HIRES_SEEDS,
    "recall_covid_64px": low_recalls.tolist(), "recall_covid_224px": hires_recalls.tolist(),
    "mean_64px": float(low_recalls.mean()), "std_64px": float(low_recalls.std(ddof=1)),
    "mean_224px": float(hires_recalls.mean()), "std_224px": float(hires_recalls.std(ddof=1)),
    "mean_delta": float((hires_recalls - low_recalls).mean()),
}
print(f"\nRecall COVID — 64px: {low_recalls.tolist()} (média {HIRES_VS_LOWRES['mean_64px']:.3f} ± {HIRES_VS_LOWRES['std_64px']:.3f}) | "
      f"224px: {hires_recalls.tolist()} (média {HIRES_VS_LOWRES['mean_224px']:.3f} ± {HIRES_VS_LOWRES['std_224px']:.3f}) | "
      f"delta médio {HIRES_VS_LOWRES['mean_delta']:+.3f}")

# %%
# fig: hires_vs_lowres
fig, ax = plt.subplots(figsize=(5.5, 4.2))
x = np.arange(len(HIRES_SEEDS))
ax.bar(x - 0.19, low_recalls, 0.38, color=C_RED, label="64 px (já reportado)")
ax.bar(x + 0.19, hires_recalls, 0.38, color=C_BLUE, label="224 px nativo (novo)")
ax.axhline(0.90, color=C_GREEN, linestyle=":", label="meta recall 0,90 (Aula 7)")
ax.axhline(HIRES_VS_LOWRES["mean_64px"], color=C_RED, linestyle="--", alpha=0.5)
ax.axhline(HIRES_VS_LOWRES["mean_224px"], color=C_BLUE, linestyle="--", alpha=0.5)
ax.set_xticks(x, [f"seed {s}" for s in HIRES_SEEDS]); ax.set_ylim(0, 1.05)
ax.set_ylabel("recall COVID (teste)"); ax.set_title("Recall de COVID: 64px vs. 224px nativo, mesmo split/seeds", fontweight="bold", fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
savefig("hires_vs_lowres")
plt.show()

# %% [markdown]
# <!-- ANALISE: preencher depois da execução real no Colab (T4/Pro). Cobrir: (1) o delta médio de recall foi na direção
# esperada (224px > 64px) e de que magnitude — aproximou da meta 0,90 ou o gargalo era outra coisa (fonte, tamanho de
# amostra, dificuldade genuína dos casos)?; (2) tempo/VRAM real de treino em 224px nativo vs 64px (o pipeline já usa
# CLS_INPUT=224 hoje, então o custo do classificador em si não deveria mudar muito — só o recarregamento das imagens
# e o uso de memória do X_all_hires; conferir contra TIMES['hires_reload_s'] e peak_vram_mb de cada run); (3) se o
# ganho for pequeno/nulo, diferenciar duas leituras: a hipótese de resolução não era o gargalo principal (aponta para
# os atalhos de fonte, item (2) da priorização acima, ou para o tamanho de amostra) vs. o downsample de 224px na EDA
# (BICUBIC) ainda perde detalhe que só apareceria em resolução maior (299px nativo do dataset) -->

# %% [markdown]
# ## 10. Métricas finais (JSON)

# %%
def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return str(o)

def strip_preds(r):
    """Resultado de classificador sem as listas de predições por imagem (ficam nos .pt do Drive)."""
    out = {k: v for k, v in r.items() if k not in ("history",)}
    for part in ("val", "test"):
        out[part] = {k: v for k, v in r[part].items() if k not in ("y_pred", "p_covid")}
    out["history_last"] = r["history"][-1]
    return out

def gan_summary(name):
    g = GAN[name]["result"]
    h = pd.DataFrame(g["history"])
    return {"config": g["run"], "train_time_s": g["train_time_s"], "iterations": g["iterations"], "best_kid_epoch": g["best_epoch"],
            "peak_vram_mb": g["peak_vram_mb"], "peak_vram_reserved_mb": g["peak_vram_reserved_mb"],
            "final_epoch": {k: round(float(v), 4) for k, v in h.iloc[-1].items()},
            "frac_epochs_D_G_z_below_0.05": round(float((h.D_G_z1 < 0.05).mean()), 3),
            "frac_epochs_loss_D_below_0.1": round(float((h.loss_D < 0.1).mean()), 3),
            "max_loss_G": round(float(h.loss_G.max()), 3), "nonfinite_iters": int(h.nonfinite_iters.sum()),
            "evals": [{k: round(float(v), 5) for k, v in e.items()} for e in g["evals"]],
            "final_metrics": {"G_best": GAN_FINAL[f"{name}/G_best"], "G_final": GAN_FINAL[f"{name}/G_final"]}}

metrics = {
    "activity": "A4.1",
    "dataset": "tawsifurrahman/covid19-radiography-database (COVID, Normal, Viral Pneumonia; Lung_Opacity ignorada)",
    "classes": CLASS_NAMES,
    "data": {"files_per_class": counts_raw.to_dict(), "unique_md5_per_class": counts_dedup.to_dict(),
             "exact_duplicates": n_dup_exact, "md5_cross_class": len(conflicting),
             "phash_threshold": PHASH_THRESHOLD, "phash_rejections": int(len(rej_df)),
             "phash_rejections_by_class": rej_df.cls.value_counts().to_dict() if len(rej_df) else {},
             "phash_rejections_cross_class": int((rej_df.cls != rej_df.match_cls).sum()) if len(rej_df) else 0,
             "partitions": split_summary, "image_size": IMG, "classifier_input": CLS_INPUT,
             "sources_dev_test": SOURCES, "brightness_dev": BRIGHTNESS, "patient_id_available": False},
    "accuracy_paradox_all_normal": pred_all_normal,
    "baseline_group": strip_preds(baseline),
    "corrected": strip_preds(corrected),
    "hires_224": {"image_size": IMG_CLS_HIRES, "reload_time_s": TIMES.get("hires_reload_s"),
                  "runs": {seed: strip_preds(hires_runs[seed]) for seed in HIRES_SEEDS},
                  "vs_64px": HIRES_VS_LOWRES},
    "gan": {"common_config": GAN_CFG, "feature_extractor": FEAT_NAME, "lpips": LPIPS_NAME, "real_diversity_covid": REAL_DIV,
            "kid_floor_train_vs_ref": KID_TRAIN_VS_REF, "runs": {name: gan_summary(name) for name in GAN},
            "generator_for_augmentation": GAN_TAG},
    "control_real_vs_synthetic": CONTROL,
    "sweep": {"multipliers": SWEEP_MULTIPLIERS, "seeds": SWEEP_SEEDS, "n_covid_train_real": N_COVID_TRAIN,
              "per_seed": sweep_df.round(4).to_dict(orient="records"),
              "mean_std": {f"{m}x": {c: {"mean": round(float(agg.loc[m, (c, 'mean')]), 4), "std": round(float(agg.loc[m, (c, 'std')]), 4)}
                                     for c in metric_cols} for m in agg.index},
              "paired_stats": STATS, "peak_vram_mb": SWEEP_PEAK_VRAM},
    "external_test_sample_size": SAMPLE_SIZE,
    "config": {"baseline": BASELINE_CFG, "corrected": CORRECTED_CFG, "augmentation": AUG, "run_A": RUN_A, "run_B": RUN_B,
               "use_diffaug": USE_DIFFAUG, "amp_fp16": USE_AMP, "seed": SEED, "force_retrain": FORCE_RETRAIN},
    "times_s": TIMES,
    "figures": FIGURES,
    "peak_vram_notebook_mb": max([r.get("peak_vram_mb") or 0 for r in RESULTS.values()] +
                                 [g["result"].get("peak_vram_mb") or 0 for g in GAN.values()]) or None,
    **ram_stats(),
    "notebook_time_s": round(time.time() - NOTEBOOK_T0, 1),
    "notebook_time_scope": "da 1a célula de código (pip) até esta célula; com checkpoints no Drive, a reexecução pula os treinos",
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__},
}
(OUT_DIR / "A4_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=jsonable))
print(json.dumps(metrics, indent=2, ensure_ascii=False, default=jsonable))
