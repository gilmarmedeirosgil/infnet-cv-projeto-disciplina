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

## Datasets (slugs Kaggle)
| Atividade | Slug | Status |
|---|---|---|
| A1 | `kaustubhdikshit/neu-surface-defect-database` | a validar no smoke test |
| A2 | `groffo/ads16-dataset` | a validar no smoke test (estrutura e nº de imagens) |
| A3 | `pavansanagapati/images-dataset` | a validar no smoke test |
| A4.1 | `tawsifurrahman/covid19-radiography-database` | a validar no smoke test |

## Pendências
- Confirmar estrutura do ADS-16 (o enunciado pede ≥500 imagens em 16 categorias).

## Uso de IA
Ferramenta: Claude Code (Anthropic, modelos Claude Opus/Sonnet), com agentes custom definidos em `agents/`.

| Data | Agente/ferramenta | O que foi gerado/assistido | Verificação humana |
|---|---|---|---|
| 27/09/2026 | Claude Code (orquestrador) | Plano do projeto, definição dos agentes, scripts `py2ipynb.py` e `extract_figures.py` | Revisado e aprovado por Gilmar |
