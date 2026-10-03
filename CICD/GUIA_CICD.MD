# Guia de CI/CD — Projeto Turnover

Este guia descreve o padrão de projeto e a esteira de CI/CD, além dos passos
**manuais** que a automação não cobre (configuração do GitHub e do Streamlit Cloud).

As diretrizes de origem estão em [`INSTRUCOES.MD`](./INSTRUCOES.MD).

---

## 1. Visão geral

| Camada | Ferramenta | Onde |
| --- | --- | --- |
| Padrão de commit | Commitizen (Conventional Commits) | local + CI |
| Lint / Formatação | Ruff | pre-commit + CI |
| Testes | Pytest | CI |
| Build de imagem | Docker → ghcr.io | GitHub Actions |
| Deploy do app | Streamlit Community Cloud | automático no push para `main` |

### Fluxo de branches

```
feature/*  ──PR──▶  dev  ──PR──▶  main
                    │              │
                    │              └─ Streamlit Cloud redeploya (produção)
                    └─ ambiente de desenvolvimento
```

- `main` = produção. **Só aceita PR vindo de `dev`.**
- `dev` = integração/desenvolvimento.

---

## 2. Estrutura do projeto

```
projeto-turnover/
├── app.py                  # Entrypoint do app Streamlit (deploy no Streamlit Cloud)
├── main.py                 # CLI: gera score a partir de um CSV
├── src/turnover/
│   ├── schema.py           # Contrato de dados (colunas esperadas / alvo)
│   ├── data.py             # Carregamento e validação do CSV
│   └── predict.py          # Geração do score (placeholder até o modelo treinado)
├── tests/                  # Testes (data-contract + smoke do pipeline)
├── Dockerfile              # Imagem do app (execução local/portabilidade)
├── .pre-commit-config.yaml # Hooks locais (ruff + commitizen)
└── .github/workflows/
    ├── ci.yml              # Lint, testes, commits, regra de branch
    └── docker.yml          # Build/push da imagem no ghcr.io
```

> **Nota sobre o modelo:** `predict.score()` é hoje um placeholder determinístico
> com a mesma assinatura do modelo final (`score(df) -> df` com a coluna
> `Score_Rotatividade`). Ao treinar o modelo (ex.: Random Forest / XGBoost com foco
> em **recall**, conforme `docs/PROBLEMA_NEGOCIO.MD`), serialize-o e carregue-o
> dentro de `predict.score()` sem mudar a assinatura. App, testes e esteira seguem
> funcionando.

---

## 3. Setup local (uma vez)

```bash
# Instala dependências (inclui as de desenvolvimento)
uv sync

# Instala os hooks de pre-commit e commit-msg
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

### Como commitar (Conventional Commits)

Use o commitizen para montar a mensagem no padrão:

```bash
uv run cz commit
```

Formato: `tipo(escopo): descrição` — ex.: `feat(predict): adiciona modelo XGBoost`.
Tipos comuns: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`.

### Rodar qualidade localmente

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```

> **Ambiente Windows (App Control / WDAC):** se a máquina bloquear a execução de
> binários baixados pelo uv (ex.: DLLs do pandas, `ruff.exe`), rode lint/testes
> dentro do Docker ou deixe a verificação para o CI (runner Linux), onde não há
> esse bloqueio.

---

## 4. Esteira de CI (`.github/workflows/ci.yml`)

Dispara em **push** e **pull request** para `dev` e `main`.

| Job | O que faz |
| --- | --- |
| `branch-policy` | Em PR para `main`, falha se a origem **não** for `dev`. |
| `quality` | `uv sync --frozen` → `ruff check` → `ruff format --check` → `pytest`. |
| `commit-messages` | Em PR, valida os commits do range com `cz check`. |

## 5. Build Docker (`.github/workflows/docker.yml`)

Dispara em **push** para `dev` e `main` (gatilho por branch, conforme as diretrizes):

- push em `dev`  → imagem publicada com a tag `dev`
- push em `main` → imagem publicada com as tags `latest` e o SHA curto

Imagens em: `ghcr.io/<owner>/<repo>`.

Rodar a imagem localmente:

```bash
docker build -t turnover-app .
docker run --rm -p 8501:8501 turnover-app
# abrir http://localhost:8501
```

---

## 6. Passos MANUAIS no GitHub (a esteira não faz)

### 6.1 Branch Protection — garantir "main só aceita merge de dev"

O job `branch-policy` do CI já faz o PR falhar quando a origem não é `dev`, mas o
**bloqueio efetivo** depende de configurar a proteção da branch:

1. Repositório → **Settings → Branches → Add branch ruleset** (ou *Branch protection rule*).
2. Branch name pattern: `main`.
3. Habilite:
   - **Require a pull request before merging** (bloqueia push direto na `main`).
   - **Require status checks to pass** → selecione os checks `branch-policy` e `quality`.
   - **Require branches to be up to date before merging**.
4. (Opcional) Repita para `dev` exigindo o check `quality`.

> Sem essa configuração, a regra "`main` só de `dev`" fica apenas no CI (que pode
> ser contornado por push direto). A proteção de branch é o que a torna obrigatória.

### 6.2 Permissões do GitHub Actions (para publicar no ghcr.io)

- **Settings → Actions → General → Workflow permissions**: deixe em
  **Read and write permissions** (o `docker.yml` usa `GITHUB_TOKEN` com `packages: write`).
- Após o primeiro push da imagem, em **Packages**, confirme a visibilidade desejada
  (privada/pública) do pacote.

---

## 7. Deploy no Streamlit Community Cloud

O Streamlit Cloud faz o deploy e o **redeploy automático** a cada push na branch
configurada — não é feito via GitHub Actions.

1. Acesse <https://share.streamlit.io> e faça login com o GitHub.
2. **New app** → selecione o repositório.
3. Configure:
   - **Branch:** `main` (produção).
   - **Main file path:** `app.py`.
4. **Deploy**. A cada push em `main`, o app é republicado automaticamente.

> **Dependências:** o Streamlit Cloud instala a partir do `pyproject.toml`/`uv.lock`
> (ou `requirements.txt`, se preferir gerar um). Mantenha as deps de runtime
> (`pandas`, `scikit-learn`, `streamlit`) no bloco `[project].dependencies`.

---

## 8. Resumo do que já está pronto no repositório

- [x] `pyproject.toml` com deps, dev-deps e configs de ruff/pytest/commitizen
- [x] Código modular em `src/turnover/` + `app.py` + testes
- [x] `Dockerfile` + `.dockerignore`
- [x] `.pre-commit-config.yaml`
- [x] Workflows de CI e de build Docker

## 9. O que ainda depende de você (manual)

- [ ] Criar a branch `dev` e definir `main` como protegida (seção 6.1)
- [ ] Ajustar Workflow permissions para publicar no ghcr.io (seção 6.2)
- [ ] Conectar o repositório no Streamlit Community Cloud (seção 7)
- [ ] Treinar e plugar o modelo de ML em `predict.score()`
