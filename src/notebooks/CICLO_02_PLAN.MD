# Ciclo 02 — Plano de Execução

Problema: classificação binária de turnover (`Attrition`: `Yes`/`No` → `1`/`0`).

O Ciclo 02 parte de tudo que foi feito no Ciclo 01 (ver `CICLO_1_PLAN.MD`) e
adiciona duas frentes novas: **Feature Engineering** e **EDA + Feature
Selection**. O objetivo é entender melhor os dados e reduzir o conjunto de
features para o que realmente tem poder preditivo, antes de treinar os modelos.

> Regra transversal do ciclo: **toda decisão e toda estatística saem apenas do
> treino**. A validação só é usada para medir desempenho, nunca para decidir
> features, scaler ou imputação. O teste permanece intocado até o ciclo final.

---

## 1. Imports e Configuração

- Bibliotecas: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy.stats`.
- Modelos: `LogisticRegression`, `RandomForestClassifier`, `xgboost`, `lightgbm`.
- Seleção de features: `VarianceThreshold`, `BorutaPy`, RFE.
- Supressão de warnings para manter a saída limpa.

## 2. Loading Data

- Carrega o CSV via caminho ancorado na raiz do projeto (`project_root()` busca
  o `pyproject.toml`), evitando caminhos relativos frágeis.

## 3. Data Split

- **3.0 Train + Test** — separa o teste primeiro (`stratify` no alvo). O teste
  fica lacrado até o ciclo final.
- **3.1 Train + Val** — converte `Attrition` para `0/1` e separa treino e
  validação (`stratify`), para o desenvolvimento do ciclo.

## 4. Data Cleaning

- Inspeção de tipos e nulos. O dataset chega sem NA; caso houvesse, o
  tratamento (ex.: imputação por mediana) seria aprendido **no treino** e
  aplicado em val/teste.

## 5. Feature Engineering

Criação de novas features, priorizando **razões e taxas** (mais informativas que
valores absolutos). Dois tipos, com cuidados diferentes:

- **Linha a linha (seguras):** dependem só da própria linha — aplicar igualmente
  em treino e validação. Exemplos: `CareerStability`, `LoyaltyRatio`,
  `CommuteImpact`, `ManagerTrust`, `TotalSatisfactionScore`, `OverTimeRisk`,
  `AgeGroup`, `Stability_Ratio`.
- **Estatística de grupo (atenção a leakage):** `IncomeRelativeToLevel` usa a
  mediana de renda por `JobLevel`. Essa mediana deve ser **aprendida no treino**
  e aplicada na validação — nunca recalculada na validação.

## 6. EDA (Exploratory Data Analysis)

### 6.1 Univariada

- **6.1.1 Numéricas** — por feature, consolida em um único DataFrame
  (`df_train_analysis`):
    - **Distribuição por classe:** média e desvio-padrão para `No` e `Yes`.
    - **Separação entre classes:** KS Statistic (Kolmogorov-Smirnov) + p-valor,
      AUC de uma Logistic Regression univariada e coeficiente de **overlap**
      (área de sobreposição das KDEs das duas classes).
    - **Outliers (método IQR, 1.5×IQR):** contagem (`N Outliers`) e percentual
      (`% Outliers`) por feature — etapa **diagnóstica**. Não há remoção de
      outliers; a escolha do `RobustScaler` (seção 8) é justamente a resposta a
      eles, por ser resistente a valores extremos.
    - **Classificação automática** de poder discriminativo (`classify`:
      🟢 Alta ≥ 0.65 / 🟡 Média ≥ 0.55 / 🔴 Baixa).
  Gera `low_discrimination_train_num_univars`.
- **6.1.2 Categóricas** — por feature: Qui-quadrado, Cramér's V e AUC de uma
  Logistic Regression univariada (com `LabelEncoder`). Mesma classificação de
  discriminação. Gera `low_discrimination_train_cat_univars`.

### 6.2 Bivariada

- **6.2.1 Numéricas** — tabela de métricas (KS, p-valor, ponto-biserial, AUC) +
  gráficos (boxplot, violino, probability plot por faixa, KDE por classe). Gera
  `low_discrimination_train_num_bivars`.
- **6.2.2 Categóricas** — taxa de attrition por categoria, stacked bar 100% e
  heatmap de contingência. Gera `low_discrimination_train_cat_bivars`.

### 6.3 Multivariada

- Matriz de correlação (Pearson para numéricas; Cramér's V para categóricas),
  visualizada em heatmap.
- Tabela de correlação de cada feature com o **alvo**.
- Tabela de **multicolinearidade** entre features explicativas: pares com
  |correlação| ≥ 0.75. Para cada par, a feature candidata a remoção é a de
  **menor correlação com o alvo** (menor poder preditivo).

## 7. Feature Selection

Objetivo: eliminar features com baixo sinal ou redundantes, reduzindo ruído e
dimensionalidade.

- **7.1 Filter method** — `VarianceThreshold` para pegar variância quasi-zero, e
  consolidação de todas as listas de baixa discriminação (univariada +
  bivariada, numéricas + categóricas) em `drop_vars` (com remoção de duplicatas
  via `set`).
- **7.2 Boruta** — seleção por `BorutaPy` sobre um Random Forest, que compara
  cada feature contra versões aleatórias (shadow features).
- **7.3 RFE** — Recursive Feature Elimination como segundo seletor, para
  cruzar/confirmar o resultado do Boruta.

> A decisão final de quais colunas remover deve ser aplicada **igualmente** em
> treino, validação e (no ciclo final) teste — usar `errors='ignore'` nos drops
> para robustez em reexecução.

## 8. Data Preparation (pós-seleção)

- **Normalização / Rescala** — `RobustScaler` (dados não-normais, com outliers).
  O scaler é ajustado (`fit`) só no treino e aplicado (`transform`) em val/teste.
  Escalar **todas** as colunas numéricas (`int64` e `float64`), já que a feature
  engineering cria colunas `float`.
- **Encoders** — `OneHotEncoder` nas categóricas (`handle_unknown='ignore'`,
  `drop='if_binary'`), ajustado no treino e aplicado em val/teste. Garantir que
  val/teste tenham **as mesmas colunas do treino, na mesma ordem**
  (`X_val = X_val[X_train.columns]`) para evitar descompasso de features.

## 9. Machine Learning Modeling

- Treina os modelos sobre o conjunto de features selecionado: Baseline
  (`DummyClassifier`), Logistic Regression, Random Forest, XGBoost e LightGBM.
- Avaliação com `ml_metrics` (Accuracy, F1, Precision, Recall, AUC) em treino e
  validação. Métrica de referência: **AUC** (robusta a desbalanceamento);
  atenção especial a **Recall** (objetivo de negócio: identificar quem vai sair).
- Tratamento de desbalanceamento de forma **consistente** entre os modelos
  (`class_weight='balanced'` no sklearn/LightGBM; `scale_pos_weight` no XGBoost).

## 10. Model Evaluation — Best Threshold

- Ajuste do limiar de decisão (padrão 0.5 é arbitrário) via curva
  precision-recall / análise de trade-off, escolhendo o corte que entrega o
  recall que o negócio precisa sem derrubar demais a precision.

---

## Pendências / Pontos de Atenção (dívida técnica do ciclo)

- **Numeração duplicada de seções:** há duas seções "7.0" (Feature Selection e
  Data Preparation) e duas "9.0" (Model Comparison e Best Threshold). Renumerar
  para 7 / 8 / 9 / 10 conforme este plano.
- **Consistência de drops:** garantir que todo drop de coluna e toda
  transformação sejam aplicados em treino **e** validação (bug recorrente de
  colunas presentes em um conjunto e não no outro).
- **Modularização (meta do ciclo final):** encapsular todo o pré-processamento
  (feature engineering, seleção, scaler, encoder) em funções `fit`/`transform`,
  para aplicar de forma idêntica e sem leakage em treino/val/teste.
