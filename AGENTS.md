# AGENTS.md

Instruções para agentes de IA que trabalham neste repositório. Projeto de ML
para prever turnover de funcionários (classificação binária).

## Setup e comandos

Gerenciador de pacotes: **uv** (não use `pip`).

```bash
uv sync                      # instala dependências
uv run train.py              # treina o modelo e gera artifacts/turnover_pipeline.joblib
uv run streamlit run app.py  # app Streamlit
uv run main.py <csv>         # CLI de scoring

uv run ruff check .          # lint
uv run ruff format .         # formatação
uv run python -m pytest      # testes (use `python -m`; `pytest` direto é bloqueado neste ambiente)
```

## Arquitetura

Pipeline modularizado em `src/turnover/`, todas as classes no padrão `fit`/`transform`:

```
FeatureEngineer → FeatureSelector → Preprocessor → ModelWrapper
        (orquestrados por TurnoverPipeline)
```

- `config.py` — constantes, hiperparâmetros (`BEST_PARAMS`), `BEST_THRESHOLD`, `DROP_COLUMNS`, caminhos. Fonte única de verdade.
- `schema.py` / `data.py` — contrato de colunas e carga/validação.
- `pipeline.py` — `TurnoverPipeline`, serializado inteiro com joblib.
- `evaluate.py` — `ml_metrics` e escolha de threshold.
- `predict.py` — scoring; carrega o artefato (com fallback heurístico).
- `train.py` (raiz) — treina em treino+val, avalia no teste, salva o artefato.
- `src/notebooks/` — ciclos 01-04 (CRISP-DM) + planos.

Modelo final: **Regressão Logística** (escolhida pelo maior Recall + interpretabilidade).

## Convenções de ML (inegociáveis)

- **Nenhuma decisão ou estatística olha a validação/teste.** Tudo (feature
  engineering de grupo, scaler, encoder, seleção de features) é aprendido
  **só no treino** (`fit`) e reaplicado via `transform`.
- O **conjunto de teste fica lacrado** até a avaliação final (uma única vez).
- O modelo final re-treina em **treino + validação** combinados.
- **Recall é a métrica de negócio prioritária** (o erro caro é não detectar
  quem vai sair). AUC é a métrica técnica principal.
- Desbalanceamento: `class_weight='balanced'` (sklearn/LightGBM) ou
  `scale_pos_weight` (XGBoost) — nunca os dois juntos.

## Convenções de git

- Trabalhe na branch `dev`. Não faça commit direto na `main`.
- **Conventional Commits** (o projeto usa commitizen): `feat:`, `fix:`, `chore:`, `docs:`.
- **Nunca versione o artefato** `artifacts/*.joblib` — é regenerado por `train.py`.
- Notebooks passam por `nbstripout` no pre-commit (outputs são removidos).

## Pegadinhas conhecidas

- O alvo `Attrition` vira `0/1` já no início do pipeline (`Yes`=1, `No`=0).
  Use `pd.api.types.is_numeric_dtype` para detectar — o pandas pode ler texto
  como dtype `string` (Arrow), não só `object`.
- `JobLevel` varia de **0 a 5** (6 níveis), não 1 a 5.
- Carregar `.joblib` executa código na desserialização: carregue **apenas** o
  artefato gerado pelo próprio projeto, nunca arquivos de terceiros.
- Determinismo: `SEED=42` em splits e modelo. Não altere sem motivo.
