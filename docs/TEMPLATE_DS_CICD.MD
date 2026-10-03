# Template de Projeto de Ciência de Dados — Git, CI/CD e Testes

Guia **replicável** para iniciar projetos de Ciência de Dados com um padrão
consistente de versionamento (Git), integração/entrega contínua (CI/CD) e testes.

Como usar: copie os blocos marcados com 📋, troque os `<placeholders>` e remova o
que não se aplicar. Cada seção explica *o porquê* para você decidir o que adaptar.

> Convenções deste documento:
> - `<nome-do-projeto>` — slug do projeto (ex.: `projeto-turnover`)
> - `<owner>/<repo>` — dono/repositório no GitHub
> - `<pkg>` — nome do pacote Python (ex.: `turnover`)

---

## 0. Decisões a tomar por projeto

Antes de copiar os arquivos, decida:

| Decisão | Opções comuns | Impacto |
| --- | --- | --- |
| Gerenciador de pacotes | `uv` (recomendado), poetry, pip+venv | Comandos e lockfile |
| Versão do Python | 3.11 / 3.12 | `requires-python`, matriz de CI |
| Alvo/tipo do modelo | classificação, regressão, clustering | Métrica de avaliação |
| Métrica principal | recall, precision, F1, AUC, RMSE... | Critério de "modelo bom" |
| Entrega (deploy) | Streamlit Cloud, API (FastAPI), batch, Docker | Workflow de CD |
| Registro de imagem | ghcr.io, Docker Hub, ECR/GAR | Workflow Docker |
| Versionamento de dados/modelo | DVC, MLflow, Git LFS, só artifacts | Rastreabilidade |

> Dica de DS: a **métrica principal** vem do problema de negócio, não da conveniência
> técnica. Ex.: em evasão/turnover, falso negativo costuma ser o erro mais caro, então
> **recall** tende a pesar mais que acurácia.

---

## 1. Git

### 1.1 Estratégia de branches

```
feature/*  ──PR──▶  dev  ──PR──▶  main
                    │              │
                    │              └─ produção (deploy)
                    └─ integração / desenvolvimento
```

- `main` = produção. Protegida. **Só aceita PR de `dev`.**
- `dev` = integração. Protegida (ao menos CI verde).
- `feature/*`, `fix/*`, `exp/*` = trabalho do dia a dia (ramos curtos).

> Para projetos de DS, considere um prefixo `exp/` para branches de
> experimentação (notebooks, tentativas de modelo) que podem não virar PR.

### 1.2 Conventional Commits (commitizen)

Padrão de mensagem: `tipo(escopo): descrição`.

Tipos: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`, `perf`, `build`.
Exemplos de DS: `feat(features): adiciona razão salário/mediana`,
`fix(data): corrige parse de datas`, `perf(model): tuning via Bayesian Search`.

📋 **`.gitignore`** (base para DS):

```gitignore
# Ambiente
.env
.venv
__pycache__/
*.pyc

# Dados e artefatos (versionar via DVC/LFS, não no Git)
data/
*.csv
*.parquet
models/
*.pkl
*.joblib

# Notebooks (saídas) e caches
.ipynb_checkpoints/
.pytest_cache/
.ruff_cache/
mlruns/
```

> **Dados fora do Git.** Datasets e modelos binários incham o repositório e viram
> um pesadelo de histórico. Ignore-os e versione com **DVC** ou **Git LFS**, ou
> registre modelos no **MLflow**. Guarde só um `data/README.md` explicando a origem.

### 1.3 Branch Protection (passo manual no GitHub)

Repositório → **Settings → Branches → Add branch ruleset**:

1. Pattern `main`:
   - Require a pull request before merging
   - Require status checks to pass → selecione os checks de CI (ex.: `quality`)
   - Require branches to be up to date before merging
2. Pattern `dev`: exigir o check de CI.

> A regra "main só de dev" só é **garantida** com Branch Protection. Um job de CI
> ajuda a falhar cedo, mas não bloqueia push direto sozinho.

---

## 2. Estrutura de projeto

```
<nome-do-projeto>/
├── app.py                     # Entrypoint do app (Streamlit/API), se houver
├── main.py                    # CLI (treino, scoring batch, etc.)
├── pyproject.toml             # Deps + configs de ferramentas
├── src/<pkg>/
│   ├── __init__.py
│   ├── schema.py              # Contrato de dados (colunas, tipos, alvo)
│   ├── data.py                # Carga + validação do dataset
│   ├── features.py            # Engenharia de features
│   ├── model.py               # Treino / carga / serialização do modelo
│   └── predict.py             # Inferência (interface estável)
├── notebooks/                 # Exploração (chamam código de src/, não o contrário)
├── tests/                     # Testes (contrato de dados, pipeline, unitários)
├── data/                      # Dados (ignorado no Git; versionado via DVC/LFS)
├── models/                    # Artefatos de modelo (ignorado no Git)
├── Dockerfile
├── .dockerignore
├── .pre-commit-config.yaml
└── .github/workflows/
    ├── ci.yml
    └── docker.yml             # se usar imagem
```

> **Princípio-chave:** o notebook explora, mas a lógica reutilizável mora em
> `src/<pkg>/`. Assim o mesmo código roda no notebook, no app, no CLI e nos testes —
> e os testes conseguem importá-lo. Separe **treino** (gera o modelo) de
> **inferência** (consome o modelo); são ciclos de vida diferentes.

📋 **`pyproject.toml`** (base, com `uv`):

```toml
[project]
name = "<nome-do-projeto>"
version = "0.1.0"
description = "<descrição>"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "pandas",
    "scikit-learn>=1.5",
    # app: "streamlit>=1.38"  |  api: "fastapi", "uvicorn"
]

[dependency-groups]
dev = ["ruff>=0.6", "pytest>=8.0", "pytest-cov", "pre-commit>=3.8", "commitizen>=3.29"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/<pkg>"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-q"

[tool.commitizen]
name = "cz_conventional_commits"
version = "0.1.0"
version_files = ["pyproject.toml:version"]
tag_format = "v$version"
update_changelog_on_bump = true
```

---

## 3. Testes

Em DS, teste o que costuma quebrar silenciosamente: **os dados e o pipeline**.
Não é só testar funções puras — é garantir que a entrada tem o formato esperado e
que o fluxo de ponta a ponta não explode.

Camadas recomendadas:

1. **Contrato de dados** — o dataset tem as colunas/tipos esperados? Rejeita vazio?
2. **Pipeline / smoke** — carregar → validar → (features) → predizer roda sem erro
   e devolve a saída no formato combinado?
3. **Unitários de transformação** — cada feature/transformação faz o que promete.
4. **(Opcional) Qualidade do modelo** — a métrica principal fica acima de um piso
   num dataset de referência (teste de regressão de modelo).

📋 **`tests/test_contract.py`** (exemplo adaptável):

```python
import pandas as pd
import pytest

from <pkg>.data import validate, DataContractError
from <pkg>.schema import REQUIRED_COLUMNS

def test_rejeita_vazio():
    with pytest.raises(DataContractError):
        validate(pd.DataFrame())

def test_rejeita_colunas_ausentes():
    with pytest.raises(DataContractError):
        validate(pd.DataFrame([{"Age": 30}]))
```

📋 **`tests/test_pipeline.py`** (smoke):

```python
import pandas as pd
from <pkg>.predict import score, SCORE_COLUMN
from <pkg>.schema import REQUIRED_COLUMNS

def test_score_formato_saida():
    df = pd.DataFrame([dict.fromkeys(REQUIRED_COLUMNS, 1)])
    out = score(df)
    assert SCORE_COLUMN in out.columns
    assert out[SCORE_COLUMN].between(0, 1).all()
```

> **Dica:** para validação de dados mais robusta, considere `pandera` ou
> `great-expectations` em vez de checagens manuais, à medida que o schema cresce.

### Rodar localmente

```bash
uv run pytest
uv run pytest --cov=src --cov-report=term-missing   # com cobertura
```

---

## 4. Qualidade de código (pre-commit)

📋 **`.pre-commit-config.yaml`**:

```yaml
default_install_hook_types: [pre-commit, commit-msg]
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=1024]
  - repo: https://github.com/commitizen-tools/commitizen
    rev: v3.29.1
    hooks:
      - id: commitizen
        stages: [commit-msg]
```

Instalar (uma vez por clone):

```bash
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

> Para notebooks, considere adicionar `nbstripout` (remove saídas pesadas dos
> `.ipynb` antes do commit) — mantém o diff limpo e o repositório leve.

---

## 5. CI (GitHub Actions)

📋 **`.github/workflows/ci.yml`**:

```yaml
name: CI

on:
  push:
    branches: [dev, main]
  pull_request:
    branches: [dev, main]

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  branch-policy:
    if: github.event_name == 'pull_request' && github.base_ref == 'main'
    runs-on: ubuntu-latest
    steps:
      - name: PR para main só pode vir de dev
        run: |
          if [ "${{ github.head_ref }}" != "dev" ]; then
            echo "::error::PRs para 'main' só a partir de 'dev' (origem: '${{ github.head_ref }}')."
            exit 1
          fi

  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: astral-sh/setup-uv@v3
        with: { enable-cache: true }
      - run: uv python install 3.12
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pytest --cov=src --cov-report=term-missing

  commit-messages:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen
      - run: uv run cz check --rev-range ${{ github.event.pull_request.base.sha }}..${{ github.event.pull_request.head.sha }}
```

> O job `quality` roda os **testes unitários** (`pytest`) no runner Linux a cada
> push/PR. Para tornar o merge dependente disso, marque `quality` como *required
> status check* na Branch Protection (seção 1.3).

---

## 6. CD / Deploy

Escolha **um** dos caminhos conforme a decisão da seção 0.

### 6.a Streamlit Community Cloud (deploy automático, sem Actions)

O Streamlit Cloud redeploya sozinho a cada push na branch configurada.

1. <https://share.streamlit.io> → login com GitHub → **New app**
2. Branch: `main` · Main file path: `app.py`
3. Deploy. Dependências vêm do `pyproject.toml`/`uv.lock` (ou `requirements.txt`).

> Nesse caminho o Actions **não** faz o deploy. A imagem Docker (6.c) fica opcional,
> para execução local/portabilidade.

### 6.b API (FastAPI) em container

Build + push da imagem (seção 6.c) e deploy no destino (ECS, Cloud Run, VPS...).
O passo de deploy depende do provedor — adicione um job que autentica e atualiza o
serviço.

📋 **`.github/workflows/docker.yml`** (6.c — build/push no ghcr.io):

```yaml
name: Docker Build & Push

on:
  push:
    branches: [dev, main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=raw,value=dev,enable=${{ github.ref == 'refs/heads/dev' }}
            type=raw,value=latest,enable=${{ github.ref == 'refs/heads/main' }}
            type=sha,prefix=,format=short
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

📋 **`Dockerfile`** (base com uv; ajuste o `CMD` ao app/API):

```dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev
COPY . .
RUN uv sync --frozen --no-dev
EXPOSE 8501
# Streamlit:
CMD ["uv", "run", "--no-dev", "streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
# API (alternativa):
# CMD ["uv", "run", "--no-dev", "uvicorn", "<pkg>.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

> Permissão para publicar no ghcr.io: Settings → Actions → General → Workflow
> permissions → **Read and write**.

---

## 7. Rastreabilidade de experimentos e dados (específico de DS)

Opcional, mas o que diferencia um projeto de DS "de verdade":

- **MLflow** — registra parâmetros, métricas (a métrica principal!) e artefatos de
  cada treino. Vira seu histórico de decisão de modelo.
- **DVC** — versiona `data/` e `models/` fora do Git, com `dvc.yaml` descrevendo o
  pipeline (reprodutível com `dvc repro`).
- **Model card** — um `models/README.md` com: dados de treino, métrica alcançada,
  limitações e data. Facilita auditoria e handover.

---

## 8. Checklist de bootstrap de um projeto novo

```
[ ] git init + branches main e dev
[ ] copiar: pyproject.toml, .gitignore, .pre-commit-config.yaml, Dockerfile, .dockerignore
[ ] copiar: .github/workflows/ci.yml (+ docker.yml se usar imagem)
[ ] criar src/<pkg>/ (schema, data, features, model, predict) e tests/
[ ] uv sync && uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
[ ] definir a métrica principal a partir do problema de negócio
[ ] configurar Branch Protection (main só de dev; CI obrigatório)
[ ] ajustar Workflow permissions (se publicar imagem)
[ ] configurar deploy (Streamlit Cloud / API / batch)
[ ] (opcional) DVC/MLflow para dados e experimentos
```
