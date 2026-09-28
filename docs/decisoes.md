# Decisões, pendências e uso de IA

## Decisões (ADR curto)
| # | Data | Decisão | Motivo |
|---|---|---|---|
| D1 | 27/09/2026 | A1 usa NEU Surface Defects (6 classes, 1.800 imgs) | Domínio industrial; dataset pequeno expõe a diferença ViT do zero vs pré-treinado vs CNN |
| D2 | 27/09/2026 | A4.1 usa cGAN condicional | Base pronta na Aula 7; treino leve na T4; instabilidade fácil de evidenciar |
| D3 | 27/09/2026 | A4.1 replica 840/240/120 + test set fixo separado | Fidelidade ao cenário do enunciado |
| D4 | 27/09/2026 | Dados via kagglehub + Colab Secrets | Reprodutível pelo avaliador com o próprio token |
| D5 | 27/09/2026 | Fonte local (percent .py) → GitHub público → Colab via colab-mcp | Versionamento + execução na T4 |
| D6 | 27/09/2026 | Relatório em Markdown + pandoc | Edição por seção pelos agentes |
| D7 | 27/09/2026 | Time de agentes custom, padrão Supervisor | Conexão única do colab-mcp; consistência de convenções |
| D8 | 27/09/2026 | Gate 0: **aceitas as 20 recomendações** de `docs/base_conhecimento/README.md` (seção "Dúvidas consolidadas") | Aprovado por Gilmar; revisões pontuais nos gates |
| D9 | 27/09/2026 | A2: corpus = 300 anúncios (20 categorias) + ~300–400 imagens originais dos usuários (sem `*_th_*`), estratificadas por usuário e POS/NEG → ~600–700 imgs | ADS-16 só tem 300 anúncios; enunciado exige ≥500 |
| D10 | 27/09/2026 | A2: exemplos do ADS-16 aparecem no notebook e no PDF (uso acadêmico, entregue só no Moodle), com a citação exigida pela licença; notebooks executados fora do repo público | Enunciado exige visualização; licença proíbe redistribuição |

## Datasets (slugs Kaggle)
| Atividade | Slug | Status |
|---|---|---|
| A1 | `kaustubhdikshit/neu-surface-defect-database` | ✔ 27/09 — 1.800 imgs, 6 classes × 300; já dividido em `NEU-DET/train/images/<classe>` (240/classe) e `NEU-DET/validation/images/<classe>` (60/classe); 28 MB |
| A2 | `groffo/ads16-dataset` | ✔ 27/09 — 1,5 GB; ver estrutura abaixo |
| A3 | `pavansanagapati/images-dataset` | ✔ 27/09 — 7 classes em `data/<classe>`: bike 365, cars 420, cats 202, dogs 202, flowers 210, horses 202, human 202 (1.803). **Cópia duplicada em `data/data/`**, que deve ser ignorada |
| A4.1 | `tawsifurrahman/covid19-radiography-database` | ✔ 27/09 — `COVID-19_Radiography_Dataset/<classe>/images`: COVID 3.616, Normal 10.192, Viral Pneumonia 1.345, Lung_Opacity 6.012 (+ máscaras pulmonares). "Pneumonia" do cenário = **Viral Pneumonia** |

### Estrutura do ADS-16 (smoke test)
- `ADS16_Benchmark_part{1,2}/.../Ads/Ads/<1..20>/<n>.png`: **300 anúncios em 20 categorias × 15**. Os nomes estão em `U*-RT.csv` (Cat0–Cat19: Clothing & Shoes, Automotive, Baby Products, Health & Beauty, Media (BMVD), Consumer Electronics, Console & Video Games, DIY & Tools, Garden & Outdoor living, Grocery, Kitchen & Home, Betting, Jewellery & Watches, Musical Instruments, Office Products, Pet Supplies, Computer Software, Sports & Outdoors, Toys & Games, Dating Sites).
- `.../Corpus/Corpus/U<id>/U<id>-IM-{POS,NEG}/`: imagens favoritas/não favoritas de 120 usuários (~5+5 cada, ~1.200 originais), mais **1.173 miniaturas `*_th_*`** duplicadas. CSVs `IM-POS/NEG` trazem uma descrição textual de cada imagem (útil para validar o CLIP).
- CSVs `INF` têm dados pessoais (parcialmente ocultos): **não usar**.
- ⚠ O enunciado fala em "16 categorias"; o dataset tem **20** categorias de anúncios. O nome ADS-16 vem do ano (EMPIRE 2016).
- ⚠ **Licença**: proíbe redistribuir as imagens e exige citar Roffo & Vinciarelli (EMPIRE 2016) com a frase "The research in this paper use the ADS-16 database". Notebooks executados com imagens do ADS-16 **não vão para o repo público**.

## Pendências
- (nenhuma)

## Uso de IA
Ferramenta: Claude Code (Anthropic, modelos Claude Opus/Sonnet), com agentes custom definidos em `agents/`.

| Data | Agente/ferramenta | O que foi gerado/assistido | Verificação humana |
|---|---|---|---|
| 27/09/2026 | Claude Code (orquestrador) | Plano do projeto, definição dos agentes, scripts `py2ipynb.py` e `extract_figures.py` | Revisado e aprovado por Gilmar |
| 27/09/2026 | Claude Code (redator-relatorio) | Rascunho da seção A4.2 (`relatorio/secoes/A4_2_trafego.md`): 6 problemas, riscos do TL ImageNet → fluxo, tabela-resumo e referências; referência Meegle não acessível (conteúdo via JS), não citada no conteúdo | Pendente de revisão por Gilmar |
| 27/09/2026 | Claude Code (`construtor-notebook`) | Notebook A3 (`notebooks/src/A3_cnn_kaggle.py` → `.ipynb`): EDA com varredura de corrompidas/modos/duplicatas, split 70/15/15, justificativa EfficientNet-B0 + benchmark de VRAM FE vs FT, treino só do head, avaliação e rascunhos das análises 1.3/1.4 | Checagem de sintaxe (`py2ipynb.py`) e teste local das células de varredura/split com dados sintéticos; execução no Colab T4 pendente |
| 27/09/2026 | Claude Code (`analista-resultados`) | Análises do notebook A3 (EDA, escolha do modelo, curvas, teste com IC de Wilson, risco de *shortcut* formato/resolução ↔ classe, 1.3 com 4 estratégias por classe, 1.4 com números do benchmark), tabela de requisitos do cabeçalho com valores medidos e rascunho de `relatorio/secoes/A3.md`, a partir de `resultados/A3/` e das figuras | Pendente de revisão por Gilmar; números conferidos contra `metrics.json`/`outputs.json`; IC de Wilson recalculados em Python; observações sobre os erros baseadas na leitura visual das figuras |
| 27/09/2026 | Claude Code (`construtor-notebook`) | Notebook A1 (`notebooks/src/A1_vision_transformers.py` → `.ipynb`): EDA do NEU, split 70/15/15, pipeline na GPU com augmentation D4 + jitter, SDPA/MHA (heads independentes)/bloco Pre-LN/PatchEmbedding/ViT do zero com testes de equivalência, demo de PE por permutação, loop comum com AMP e checkpoint/retomada no Drive, ViT do zero vs ViT-B/16 in21k vs ResNet-18, mapas de atenção (1 head + rollout), distância de atenção, teste de patches embaralhados, rascunhos 2.2/2.5/3.4/3.5/3.6 | Testes dos módulos e execução ponta a ponta em CPU local (dataset sintético, ViT HF minúsculo no lugar do ViT-B, 2 épocas), incluindo retomada após desconexão simulada; execução no Colab T4 pendente |
| 27/09/2026 | Claude Code (`construtor-notebook`) | A3 v2: auditoria de quase-duplicatas (pHash, Hamming ≤ 4) e acc no teste limpo, fração de alfa, RAM/VRAM reservada/tempos medidos, figura de erros com a entrada real do modelo, EDA no JSON, ajustes de texto pedidos pelo revisor | Teste local das células novas com dados sintéticos; re-execução no Colab T4 pendente |
| 27/09/2026 | Claude Code (`analista-resultados`) | A3 v2: análises dos 5 marcadores novos (cabeçalho com RAM/VRAM reservada/tempos medidos, alfa 100% opaco, quase-duplicatas pHash, teste limpo, custo do treino), números do benchmark/treino atualizados, leitura da nova figura de erros (crop tirou olhos e orelhas do gato; bicicleta não foi cortada) e revisão de `relatorio/secoes/A3.md` com os apontamentos do revisor | Pendente de revisão por Gilmar; números conferidos contra `resultados/A3/metrics.json` e `outputs.json`; causas dos erros mantidas como hipótese |
| 27/09/2026 | Claude Code (`analista-resultados`) | A1: seção 11 (discussão, itens 11.1–11.5) fechada, amarrando os argumentos teóricos (BERT vs ViT, DeiT/Swin, quando ViT supera CNN, escolha de arquitetura, próximos passos) aos números medidos nas seções 9 (avaliação: 266/270 vs 270/270 vs 270/270, McNemar p=0,125) e 10 (atenção: distância média, entropia, patches embaralhados); `relatorio/secoes/A1.md` escrito a partir do notebook e das figuras | Pendente de revisão por Gilmar; números conferidos contra `resultados/A1/A1_comparison.csv`; leituras de atenção e causas de erro mantidas como hipótese |
| 28/09/2026 | Claude Code (orquestrador) | A1: adaptado para detectar CUDA/MPS/CPU automaticamente (execução local em Apple Silicon como alternativa ao Colab); execução completa confirmada na T4 do Colab (`A1_vision_transformers (1).ipynb` baixado por Gilmar) — 67/67 células executadas sem erro, JSON final e figuras conferidos e idênticos aos números já escritos em `relatorio/secoes/A1.md` | Notebook executado e baixado por Gilmar; JSON (`A1_metrics.json` embutido na saída) e 3 figuras (matrizes de confusão, curvas de treino) conferidos byte a byte contra o relatório pelo Claude Code |
