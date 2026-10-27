# Ciclo 03 — Plano de Execução

Problema: classificação binária de turnover (`Attrition`: `Yes`/`No` → `1`/`0`).

O Ciclo 03 parte de tudo que foi feito no Ciclo 02 (ver `CICLO_02_PLAN.MD`) e
adiciona a frente nova do ciclo: **fine-tuning de hiperparâmetros** com
**validação cruzada K-Fold estratificada**. O objetivo é sair dos modelos com
configuração padrão (Ciclo 02) para modelos **otimizados**, comparando três
estratégias de busca de hiperparâmetros: Random Search, Grid Search e Bayesian
Search (via Optuna).

> Regra transversal do ciclo: **toda decisão e toda estatística saem apenas do
> treino**. A validação cruzada é feita **dentro do treino**; o conjunto de
> validação (`X_val`) serve para medir o desempenho final do modelo otimizado, e
> o teste permanece intocado até o ciclo final.

---

## 1. Imports e Configuração

- Herda tudo do Ciclo 02 (`pandas`, `numpy`, `matplotlib`, `seaborn`,
  `scipy.stats`, modelos, seletores de features).
- **Novos imports do ciclo:**
    - `from sklearn.model_selection import StratifiedKFold, cross_val_score`
    - `from sklearn.model_selection import RandomizedSearchCV, GridSearchCV`
    - `import optuna`
- Supressão de warnings para manter a saída limpa.

## 2. Loading Data

- Carrega o CSV via caminho ancorado na raiz do projeto (`project_root()`).

## 3. Data Split

- **3.0 Train + Test** — separa o teste primeiro (`stratify`). Teste lacrado.
- **3.1 Train + Val** — converte `Attrition` para `0/1` e separa treino/validação
  (`stratify`).

## 4. Data Cleaning

- Inspeção de tipos e nulos (dataset chega sem NA).

## 5. Feature Engineering

- Mesmas features do Ciclo 02 (razões/taxas), com o cuidado de leakage:
  features linha a linha aplicadas em treino e val; estatísticas de grupo
  (ex.: `IncomeRelativeToLevel`) aprendidas **no treino**.

## 6. EDA (Exploratory Data Analysis)

Herdada do Ciclo 02 (pode ser mantida resumida, já que o foco do ciclo é o
tuning):

- **6.1 Univariada** — numéricas (KS, overlap, AUC, outliers IQR) e categóricas
  (Qui-quadrado, Cramér's V, AUC).
- **6.2 Bivariada** — numéricas e categóricas.
- **6.3 Multivariada** — correlação (Pearson + Cramér's V), correlação com o
  alvo e multicolinearidade (|corr| ≥ 0.75).

## 7. Feature Selection

- **7.1 Filter method** — `VarianceThreshold` + consolidação das listas de baixa
  discriminação em `drop_vars`.
- **7.2 Boruta** — seleção com `BorutaPy` sobre Random Forest.
- **7.3 RFE** — Recursive Feature Elimination como segundo seletor.

## 8. Data Preparation (pós-seleção)

- **8.1 Scaler** — `RobustScaler` ajustado no treino, aplicado em val/teste.
- **8.2 Encoders** — `OneHotEncoder` ajustado no treino, aplicado em val/teste.
  Alinhar colunas de val pelas do treino (`X_val = X_val[X_train.columns]`).

## 9. Machine Learning Modeling (Baseline do ciclo)

- Treina os modelos com configuração **padrão** (como no Ciclo 02): Baseline,
  Logistic Regression, Random Forest, XGBoost e LightGBM.
- Avaliação com `ml_metrics` em treino e validação. Esses números são a
  **referência pré-tuning**: o objetivo do ciclo é superá-los.
- Tratamento de desbalanceamento consistente (`class_weight='balanced'` /
  `scale_pos_weight`).

## 10. Fine-Tuning de Hiperparâmetros (foco do Ciclo 03)

### 10.1 Estratégia de validação cruzada

- **StratifiedKFold** (ex.: `n_splits=5`, `shuffle=True`, `random_state=42`)
  como esquema de CV compartilhado por **todas** as buscas, para comparação
  justa. O `Stratified` preserva a proporção de turnover em cada fold
  (importante em base desbalanceada).
- Toda a busca é feita **dentro do treino** (`X_train`), via CV. A validação
  (`X_val`) só é tocada para medir o modelo final de cada estratégia.
- **Métrica de otimização:** AUC (`scoring='roc_auc'`) como principal, com
  atenção ao Recall (objetivo de negócio). Definir a métrica-alvo de forma
  idêntica nas três buscas.

### 10.2 Escolha do modelo a tunar

- Selecionar o(s) modelo(s) mais promissor(es) do passo 9 (tipicamente XGBoost
  e/ou LightGBM, que lideram em dados tabulares). O mesmo estimador base e o
  mesmo espaço de busca devem ser usados nas três estratégias, para a
  comparação ser honesta.

### 10.3 Random Search (`RandomizedSearchCV`)

- Amostra aleatória de combinações de hiperparâmetros dentro do espaço definido.
- Parâmetros: `n_iter` (orçamento de avaliações), `scoring`, `cv` (o
  StratifiedKFold), `random_state`, `n_jobs=-1`.
- Rápido e bom para varrer um espaço amplo com orçamento limitado.

### 10.4 Grid Search (`GridSearchCV`)

- Busca exaustiva em todas as combinações de uma grade (grid) de valores.
- Garante cobertura completa do grid, mas custo cresce exponencialmente com o
  número de parâmetros — manter o grid **enxuto** (poucos valores por
  hiperparâmetro) para ser viável.

### 10.5 Bayesian Search (Optuna)

- `optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))`.
- Função `objective(trial)` que sugere hiperparâmetros (`trial.suggest_*`),
  treina via `cross_val_score` com o StratifiedKFold e retorna a métrica média.
- TPE (Tree-structured Parzen Estimator) aprende com os trials anteriores,
  tendendo a encontrar bons pontos com menos avaliações que o Random Search.
- Usar o **mesmo orçamento** (`n_trials` = `n_iter` do Random Search) para
  comparação justa.

### 10.6 Comparação das três estratégias

- Tabela comparando, para cada estratégia: melhores hiperparâmetros, AUC médio
  de CV (otimista) e **AUC/Recall na validação** (`X_val`) — este último é o
  número honesto para decidir.
- Retreinar cada modelo vencedor com os `best_params` **mais** os parâmetros
  fixos (ex.: `class_weight`/`scale_pos_weight`), para o modelo final não perder
  o tratamento de desbalanceamento.
- Discussão: CV é estimativa otimista (viés de seleção ao pegar o melhor de N
  trials); a validação reflete melhor o desempenho em dados novos.

## 11. Model Evaluation — Best Threshold

- Ajuste do limiar de decisão (0.5 é arbitrário) via curva precision-recall,
  escolhendo o corte que entrega o recall-alvo do negócio sem derrubar demais a
  precision — aplicado ao melhor modelo otimizado do passo 10.

---

## Pendências / Pontos de Atenção

- **Comparação justa das buscas:** mesmo estimador base, mesmo espaço de busca,
  mesmo `cv` (StratifiedKFold compartilhado), mesmo orçamento e mesma métrica nas
  três estratégias. Sem isso, a comparação não é conclusiva.
- **Retreino com parâmetros fixos:** ao retreinar com `best_params`, somar os
  parâmetros fixos que não entraram no espaço de busca (ex.:
  `class_weight='balanced'`), senão o modelo final difere do avaliado no CV.
- **CV vs Validação:** o `best_value` do CV é otimista; comparar as estratégias
  pela métrica na **validação**, não pelo CV.
- **Custo do Grid Search:** manter o grid pequeno; usar o Random/Bayesian para
  espaços grandes.
- **Modularização (meta do ciclo final):** encapsular pré-processamento e tuning
  em funções `fit`/`transform` para aplicar sem leakage em treino/val/teste.
