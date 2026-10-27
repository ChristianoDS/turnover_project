# 📉 Previsão de Turnover de Funcionários

Sistema de *machine learning* que prevê a **probabilidade de um colaborador deixar a empresa** (turnover / attrition), permitindo que o RH aja de forma **proativa** na retenção de talentos. O projeto entrega um modelo treinado, um pacote Python modularizado e um app Streamlit para simulação individual, em lote e de ROI.

---

## 🎯 O Problema de Negócio

A **InovaTech Solutions** (empresa de tecnologia, +1.400 colaboradores) enfrentou uma alta na rotatividade, principalmente em cargos técnicos e de gestão. A perda de talentos para a concorrência vinha corroendo a margem de lucro e atrasando entregas.

A rotatividade é um dos problemas mais **caros** de uma organização:

- **Custo de substituição:** entre **50% e 200% do salário anual** do colaborador (rescisão, recrutamento, seleção, onboarding).
- **Perda de conhecimento:** o talento leva consigo o *know-how* e o histórico dos projetos.
- **Efeito dominó:** equipes sobrecarregadas pela saída de colegas tendem a gerar novas saídas.

**Missão:** ajudar o RH a **reduzir a rotatividade**, identificando antecipadamente quem está em risco de sair.

### Por que a métrica-chave é o Recall

Este é um problema de **classificação binária** (o funcionário vai sair ou não). A métrica de negócio priorizada é o **Recall**, porque o **erro mais caro** é o modelo dizer que um funcionário vai **ficar** quando na verdade ele vai **sair** (falso negativo): perde-se a chance de retê-lo e arca-se com todo o custo de substituição. É preferível investigar alguns falsos alarmes (precision menor) a deixar escapar quem realmente vai embora.

---

## 💰 Retorno Financeiro Estimado

Análise sobre a base histórica (1.761 colaboradores), com custo de substituição ponderado por senioridade (júnior 40%, pleno 100%, sênior 150%, gestão 200% do salário anual):

| Indicador | Valor |
| --- | --- |
| Total de colaboradores | 1.761 |
| Saídas no período | 528 (**29,98%**) |
| Folha salarial anual total | R$ 132.937.080 |
| **Custo total estimado do turnover** | **R$ 37.495.244** |
| **Custo médio por saída** | **R$ 71.014** |

**Potencial de economia:** cada funcionário em risco **identificado e retido** evita ~R$ 71 mil em custo de substituição. Mesmo recuperando uma fração dos que sairiam, o retorno supera com folga o custo de operar o modelo — o app traz uma aba de **Simulação de ROI** que calcula economia bruta, líquida e ROI anual em função do recall do modelo e da taxa de retenção do RH.

---

## 🧪 Modelos Testados

Seguindo a metodologia **CRISP-DM**, o desenvolvimento foi organizado em ciclos incrementais. Na fase de modelagem foram comparados cinco modelos (métricas de **validação**, antes do fine-tuning):

| Modelo | AUC | Recall | Precision | F1 | Observação |
| --- | --- | --- | --- | --- | --- |
| Baseline (aleatório) | ~0,49 | 0,31 | 0,29 | 0,30 | piso de referência |
| **Regressão Logística** | 0,889 | **0,81** | 0,67 | 0,73 | linear, interpretável, maior recall |
| Random Forest | 0,902 | 0,72 | 0,80 | 0,76 | bagging |
| XGBoost | 0,890 | 0,71 | 0,80 | 0,75 | boosting sequencial |
| LightGBM | 0,881 | 0,69 | 0,79 | 0,74 | boosting leaf-wise |

> A Random Forest teve a maior AUC, mas a **Regressão Logística entregou o maior Recall (0,81)** — a métrica de negócio priorizada — além de ser a mais interpretável. Por isso foi a escolhida.

### Fine-Tuning e modelo final

Os hiperparâmetros foram otimizados com **validação cruzada estratificada (StratifiedKFold)** comparando três estratégias de busca: **Random Search**, **Grid Search** e **Bayesian Search (Optuna/TPE)**.

O modelo escolhido para produção foi a **Regressão Logística** — por entregar o **maior Recall** (a métrica de negócio prioritária), além da **interpretabilidade** (fator importante em decisões de RH) e da simplicidade de manutenção. Após a feature selection (de 64 para 26 features) e o re-treino em treino + validação, o desempenho no **conjunto de teste** (intocado até o fim) foi:

| Métrica (teste) | Valor |
| --- | --- |
| AUC | **0,862** |
| Recall | **0,75** |
| Precision | 0,60 |
| F1 | 0,67 |
| Accuracy | 0,78 |

O **limiar de decisão** é configurável (`BEST_THRESHOLD`), permitindo ajustar o trade-off recall × precision conforme a tolerância do negócio.

---

## 🔬 Metodologia (CRISP-DM em 4 ciclos)

O projeto evoluiu em ciclos, cada um documentado em um plano (`src/notebooks/CICLO_0X_PLAN.MD`) e um notebook:

- **Ciclo 01 — Baseline:** modelo base com todas as colunas, sem EDA nem tuning.
- **Ciclo 02 — EDA + Feature Selection:** análise univariada/bivariada/multivariada, engenharia de features (razões/taxas), seleção por filtro, multicolinearidade e **Boruta**.
- **Ciclo 03 — Fine-Tuning:** otimização de hiperparâmetros (Random / Grid / Bayesian Search) com cross-validation estratificado.
- **Ciclo 04 — Produtização:** pipeline modularizado em classes, treino final e serialização do modelo.

Decisão metodológica transversal: **toda estatística é aprendida apenas no treino** (feature engineering, scaler, encoder, seleção), evitando *data leakage*. O conjunto de **teste permanece lacrado** até a avaliação final.

---

## 🏗️ Arquitetura

O pipeline foi modularizado em classes (`src/turnover/`), todas no padrão `fit`/`transform`:

```
FeatureEngineer → FeatureSelector → Preprocessor → ModelWrapper
        (orquestrados por TurnoverPipeline)
```

| Componente | Responsabilidade |
| --- | --- |
| `FeatureEngineer` | cria features (razões/taxas); aprende estatísticas de grupo só no treino |
| `FeatureSelector` | remove as colunas descartadas na seleção dos Ciclos 02/03 |
| `Preprocessor` | `RobustScaler` (numéricas) + `OneHotEncoder` (categóricas) |
| `ModelWrapper` | Regressão Logística com `class_weight='balanced'` + save/load |
| `TurnoverPipeline` | orquestra tudo; serializa o pipeline inteiro com `joblib` |

Treino e produção compartilham o **mesmo código**: o `train.py` gera o artefato `.joblib` e o app / `predict.py` o consomem.

---

## 🚀 Como usar

### Pré-requisitos

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes)

### Instalação

```bash
git clone https://github.com/ChristianoDS/turnover_project.git
cd turnover_project
uv sync
```

### Treinar o modelo

Treina em treino + validação, avalia no teste e gera o artefato `artifacts/turnover_pipeline.joblib`:

```bash
uv run train.py
```

> O artefato não é versionado — cada ambiente o regenera. O treino é determinístico (`SEED=42`), então o modelo sai idêntico em qualquer máquina.

### Rodar o app (Simulador de Rotatividade)

```bash
uv run streamlit run app.py
```

O app tem **4 abas**:

1. **Simulação Individual** — formulário de um colaborador → probabilidade de saída.
2. **Simulação em Lote** — amostra aleatória do dataset com previsões.
3. **Simulação de ROI** — economia e ROI da ação de retenção.
4. **Upload CSV** — score em massa a partir de um arquivo enviado.

### CLI (scoring de um CSV)

```bash
uv run main.py data/HR-Employee-Attrition-balanced.csv
```

---

## ✅ Qualidade

```bash
uv run ruff check .      # lint
uv run ruff format .     # formatação
uv run python -m pytest  # testes
```

## 🔁 CI/CD

Esteira em GitHub Actions (lint, testes, padrão de commit, build Docker) com deploy via Streamlit Community Cloud. Detalhes em [`CICD/GUIA_CICD.MD`](./CICD/GUIA_CICD.MD).

---

## 📁 Estrutura do projeto

```
├── app.py                      # app Streamlit (4 abas)
├── train.py                    # treino do modelo final + serialização
├── main.py                     # CLI de scoring
├── artifacts/                  # modelo treinado (.joblib, não versionado)
├── data/                       # dataset
├── src/turnover/               # pacote do pipeline
│   ├── config.py               # constantes, hiperparâmetros, caminhos
│   ├── schema.py / data.py     # contrato e carga/validação de dados
│   ├── feature_engineering.py  # FeatureEngineer
│   ├── feature_selection.py    # FeatureSelector
│   ├── preprocessing.py        # Preprocessor
│   ├── model.py                # ModelWrapper
│   ├── pipeline.py             # TurnoverPipeline
│   ├── evaluate.py             # métricas e escolha de threshold
│   └── predict.py              # scoring (carrega o artefato)
├── src/notebooks/              # ciclos 01-04 + planos (CRISP-DM)
├── tests/                      # testes de contrato e do pipeline
└── docs/                       # problema de negócio, CRISP-DM, ROI, roadmap
```

## 📚 Documentação complementar

- [`docs/PROBLEMA_NEGOCIO.MD`](./docs/PROBLEMA_NEGOCIO.MD) — contexto e missão
- [`docs/ESTIMATIVA_ROI.MD`](./docs/ESTIMATIVA_ROI.MD) — números do retorno financeiro
- [`docs/ROADMAP_SOLUCAO.MD`](./docs/ROADMAP_SOLUCAO.MD) — plano da solução
- [`docs/CRISP-DM/`](./docs/CRISP-DM) — conceitos de cada etapa

---

## 📊 Dados

Base pública **IBM HR Analytics Employee Attrition** (35 variáveis: satisfação, remuneração, equilíbrio vida/trabalho, histórico de carreira), balanceada para o estudo.
Fonte: [Kaggle — IBM HR Analytics](https://www.kaggle.com/code/faressayah/ibm-hr-analytics-employee-attrition-performance)
