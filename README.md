# Projeto Turnover

Modelo de classificação de **turnover (attrition)** com um app Streamlit que recebe
um CSV de colaboradores e devolve o **Score de Rotatividade**. O contexto de negócio
e a metodologia (CRISP-DM) estão em [`docs/`](./docs).

## Rodando localmente

```bash
uv sync
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg

# App web
uv run streamlit run app.py

# CLI (score a partir de um CSV)
uv run main.py data/HR-Employee-Attrition-balanced.csv
```

## Qualidade

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```

## CI/CD

Esteira em GitHub Actions (lint, testes, padrão de commit, build Docker) e deploy via
Streamlit Community Cloud. Detalhes e passos manuais em
[`CICD/GUIA_CICD.MD`](./CICD/GUIA_CICD.MD).

## Estrutura

- `app.py` — entrypoint do app Streamlit
- `main.py` — CLI de scoring
- `src/turnover/` — código reutilizável (`schema`, `data`, `predict`)
- `tests/` — testes de contrato de dados e do pipeline
- `docs/` — problema de negócio, CRISP-DM, ROI, roadmap
