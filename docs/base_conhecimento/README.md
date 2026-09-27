# Base de conhecimento — índice

Gerada na Etapa 0.1 por 3 instâncias do agente `pesquisador-aulas` a partir de `Aula-1` … `Aula-8` (notebooks, `.md`, slides e capítulos em PDF; vídeos não lidos). Índices de célula são 0-based.

| Arquivo | Atividades | Aulas lidas | Linhas |
|---|---|---|---|
| [A3_A2_cnn_clip.md](A3_A2_cnn_clip.md) | A3 (CNN transfer learning), A2 (CLIP/ADS-16) | Aula-1, Aula-6 (+ Aula-3 pontual para BERT) | 186 |
| [A1_transformers_vit.md](A1_transformers_vit.md) | A1 (atenção, ViT do zero, fine-tuning, DeiT/Swin) | Aula-2 … Aula-5 | 242 |
| [A4_gans.md](A4_gans.md) | A4.1 (cGAN raio-X), A4.2 (tráfego) | Aula-7, Aula-8 | 229 |

## Achados que mudam o plano
- **A1:** nenhum notebook do professor tem ViT do zero. Montar a partir de Aula-2 (células 12–18: SDPA, FFN, EncoderLayer Pre-LN) + esqueleto do Géron cap. 16. O MHA da Aula-2 usa projeção fundida: para cumprir a rubrica 2.1 **literalmente**, as heads devem ser explícitas.
- **A1:** extração de atenção segue dois padrões do professor: rollout (Aula-4, célula 25) e 1 head CLS→patches (DINO, células 13/17), ambos com `attn_implementation="eager"`.
- **A2:** cossenos CLIP ficam ~0,24–0,29 (margem ~0,05), então um threshold absoluto fixo não funciona; é preciso calibrar. O ADS-16 não aparece em nenhuma aula.
- **A3:** o núcleo da rubrica 1.1 está em `Aula-1/aula_01_cnn_architectures.ipynb` células 14 e 16. Split estratificado, accuracy por classe e matriz de confusão não estão nas aulas (usar scikit-learn).
- **A4.1:** a cGAN da Aula-7 é binária, com 1 seed, sem validação e sem métricas de GAN, então a evidência da rubrica 5.3 precisa ser construída. A DCGAN do professor mostra D(G(z)) ≈ 0,01 (discriminador dominando), o que serve de exemplo para a Run A.
- **A4.2:** quase sem material nas aulas; transpor os "5 problemas técnicos" do caso médico e a CycleGAN como ideia de adaptação de domínio.

## Dúvidas consolidadas para o Gilmar (Gate 0)
Recomendação do orquestrador entre parênteses.

**A3**
1. Modelo: EfficientNet-B0 ou ResNet-50? (EfficientNet-B0: está no slide da aula e é mais leve na T4.)
2. Augmentation no único treino ou só na análise escrita da 3.2? (Treino com `weights.transforms()` puro, como na aula; a augmentation fica na análise 3.2, que é o que o enunciado pede.)

**A2**

3. CLIP `patch32` (aula) ou `patch16`? (patch16: melhor recuperação e cabe na T4.)
4. Método de threshold? (Percentil da distribuição + margem sobre prompt neutro, validado por inspeção de uma amostra.)
5. Idioma das consultas? (Inglês no modelo, traduzido nas tabelas do relatório.)

**A1**

6. MHA com heads explícitas (`ModuleList`) + teste de equivalência? (Sim, literal à rubrica.)
7. Checkpoint pré-treinado principal? (`google/vit-base-patch16-224-in21k`: pré-treino supervisionado puro, casa com o texto 2.5; DeiT-small como extra se houver tempo.)
8. Trainer HF ou loop manual? (Loop manual igual ao do ViT do zero, com tempo e VRAM medidos igual.)
9. Resolução do ViT do zero? (224 com patch 16 = 196 tokens, igual ao pré-treinado; se ficar lento, 128.)
10. CutMix no ViT do zero? (Não no principal: pode colar defeitos de classes diferentes na mesma textura; citar como alternativa.)
11. Mapas de atenção: 1 head e/ou rollout? (1 head + rollout em ambos os modelos.)
12. Extras (Swin-tiny, backbone congelado)? (Swin-tiny só se sobrar GPU; reforça a 3.4 com dados.)

**A4.1**

13. Resolução da cGAN: 64 ou 128 px? (64 px nas Runs A/B.)
14. Resolução do classificador? (A mesma da GAN, para evitar o atalho "imagem borrada = sintética = COVID"; mais um teste de controle real vs sintético.)
15. Flip horizontal em raio-X? (Não: inverte a lateralidade anatômica.)
16. Treinar a cGAN nas 3 classes e gerar só COVID? (Sim.)
17. Quantidade de sintéticos? (Pequeno sweep: 0, 1× e 3× o nº de COVID reais no treino.)
18. Métricas da 5.3: KID + diversidade LPIPS, com FID comparativo? (Sim.)
19. Mitigação da Run B: SN no D + label smoothing + TTUR (tudo do slide 9), DiffAugment como extra? (Sim.)
20. Trocar `Sigmoid+BCELoss` por `BCEWithLogitsLoss`? (Sim, justificando.)
