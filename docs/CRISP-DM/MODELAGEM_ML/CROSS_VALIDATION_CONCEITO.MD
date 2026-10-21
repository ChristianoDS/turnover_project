## O Problema que o Cross-Validation Resolve

Antes de aprender o que é Cross-Validation, é preciso entender o problema que ele resolve: **como estimar a performance real de um modelo em dados que ele nunca viu?**

A resposta ingnua seria: treinar o modelo em todos os dados e medir o erro no mesmo dataset. O problema é que qualquer modelo pode simplesmente **memorizar** os dados de treino — produzindo erro de treino baixo sem generalizar para dados novos. Isso é overfitting.

A resposta um pouco menos ingnua: separar os dados em treino e teste, treinar no treino, medir no teste. Funciona, mas tem um problema sutil: se o cientista de dados usa o conjunto de teste para tomar decisões — comparar modelos, ajustar hiperparâmetros, selecionar features — o conjunto de teste deixa de ser um estimador imparcial. Ele se torna **contaminado** pelo processo de modelagem.

### O dilema fundamental

```
Se avalio no treino:     performance inflada (overfitting)
Se avalio no teste:      performance real, mas uso o teste para decidir
                         → teste contaminado → performance inflada também

Solução: precisamos de uma estimativa de performance que não
          contamine o conjunto de teste final.

Cross-Validation resolve isso usando os próprios dados de treino
para simular múltiplas avaliações em dados "novos".
```

## 2. O que é Cross-Validation?

Cross-Validation (CV) é uma técnica de avaliação que divide o conjunto de treino em múltiplos subconjuntos, treinando e avaliando o modelo repetidamente em partições diferentes. O resultado é uma **estimativa estatisticamente mais confiável** da performance de generalização do que uma única avaliação train/test.

### O que o CV fornece?

- Uma **média de performance** mais robusta do que uma única avaliação
- Um **desvio padrão** que quantifica a incerteza da estimativa
- Uma forma de **comparar modelos** de forma justa e sem contaminar o teste
- Detecção de **instabilidade**: se o desvio padrão é alto, o modelo é sensível à partição dos dados

### O que o CV NÃO é

- Não substitui o conjunto de teste final — ele estima a performance, o teste confirma
- Não resolve data leakage se o pré-processamento for feito antes do CV
- Não garante generalização se o dataset tiver distribuição temporal ou geográfica especial

## 3. K-Fold Cross-Validation

O K-Fold é o método mais utilizado. O dataset de treino é dividido em **K partições (folds) de tamanho igual**. Em cada uma das K iterações, um fold diferente é usado como validação e os K-1 folds restantes como treino. A performance final é a média das K avaliações.

### Como funciona com K=5

```
Dataset de treino: [F1] [F2] [F3] [F4] [F5]

Fold 1:  Treino=[F2,F3,F4,F5]  Validação=[F1]  → AUC=0.821
Fold 2:  Treino=[F1,F3,F4,F5]  Validação=[F2]  → AUC=0.834
Fold 3:  Treino=[F1,F2,F4,F5]  Validação=[F3]  → AUC=0.819
Fold 4:  Treino=[F1,F2,F3,F5]  Validação=[F4]  → AUC=0.828
Fold 5:  Treino=[F1,F2,F3,F4]  Validação=[F5]  → AUC=0.815

AUC média:  0.823
Desvio padrão: 0.007  → modelo estável
```

### Implementação

```python
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

# Método simples
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='roc_auc', n_jobs=-1)

print(f"AUC por fold:    {scores.round(4)}")
print(f"AUC média:       {scores.mean():.4f}")
print(f"Desvio padrão:   {scores.std():.4f}")
print(f"Intervalo 95%%:   [{scores.mean()-2*scores.std():.4f}, {scores.mean()+2*scores.std():.4f}]")

# Método completo: múltiplas métricas e tempo de treino
result = cross_validate(
    model, X_train, y_train,
    cv=kf,
    scoring={'auc': 'roc_auc', 'f1': 'f1'},
    return_train_score=True,
    n_jobs=-1
)
print(f"\nAUC treino média:      {result['train_auc'].mean():.4f}")
print(f"AUC validação média:   {result['test_auc'].mean():.4f}")
# AUC treino >> AUC val → sinal de overfitting
```

### Como escolher K?

- **K=5:** boa para datasets com N > 5.000. Equilíbrio entre variância da estimativa e custo computacional
- **K=10:** mais preciso que K=5, mas 2x mais lento. Recomendado para datasets médios
- **K=N (LOOCV):** Leave-One-Out — estimativa quase sem viés, mas extremamente lento. Reservado para datasets muito pequenos (N < 100)

## 4. Stratified K-Fold

O K-Fold padrão divide os dados aleatoriamente sem garantia de que a proporção de cada classe seja preservada em cada fold. Em datasets desbalanceados, isso pode gerar folds com proporções de classes muito diferentes — tornando a avaliação instável.

O **Stratified K-Fold** garante que cada fold tenha a mesma proporção de cada classe do dataset original.

```python
from sklearn.model_selection import StratifiedKFold

# Sempre prefira StratifiedKFold para classificação
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    model, X_train, y_train,
    cv=skf,               # estratificado
    scoring='roc_auc',
    n_jobs=-1
)

print(f"AUC média (Stratified K-Fold): {scores.mean():.4f} ± {scores.std():.4f}")

# Verificar proporção de classes por fold
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    taxa_treino = y_train.iloc[train_idx].mean()
    taxa_val    = y_train.iloc[val_idx].mean()
    print(f"Fold {fold+1}: taxa treino={taxa_treino:.3f} | taxa val={taxa_val:.3f}")
```

**Regra prática:** para classificação, sempre use `StratifiedKFold`. Para regressão, use `KFold` padrão.

## 5. Time Series Cross-Validation

O K-Fold e o Stratified K-Fold assumem que as observações são **independentes e identicamente distribuídas (i.i.d.)**. Dados temporais violam essa suposição: o passado influencia o futuro, e usar dados futuros para prever o passado durante a avaliação é **data leakage temporal**.

O **TimeSeriesSplit** (ou Walk-Forward Validation) resolve isso: o conjunto de treino de cada fold só contém dados **anteriores** ao fold de validação, nunca posteriores.

```
TimeSeriesSplit com K=5:

Fold 1:  Treino=[t₁...t₂₀₀]       Validação=[t₂₀₁...t₂‵₀]
Fold 2:  Treino=[t₁...t₂‵₀]       Validação=[t₂‵₁...t₃₀₀]
Fold 3:  Treino=[t₁...t₃₀₀]       Validação=[t₃₀₁...t₃‵₀]
Fold 4:  Treino=[t₁...t₃‵₀]       Validação=[t₃‵₁...t₄₀₀]
Fold 5:  Treino=[t₁...t₄₀₀]       Validação=[t₄₀₁...t₅₀₀]

O treino sempre cresce — nunca há dados futuros no treino.
```

```python
from sklearn.model_selection import TimeSeriesSplit

tss = TimeSeriesSplit(n_splits=5)

scores = cross_val_score(
    model, X_train, y_train,
    cv=tss,
    scoring='roc_auc',
    n_jobs=1    # sequential obrigatório para dados temporais
)

print(f"AUC por fold (temporal): {scores.round(4)}")
print(f"AUC média:  {scores.mean():.4f} ± {scores.std():.4f}")
```

## 6. CV dentro do Pipeline: Evitando Data Leakage

Um erro crítico e comum: aplicar pré-processamento (normalização, encoding, imputação, feature selection) no dataset **antes** do cross-validation. Isso vaza informação dos folds de validação para o treino, produzindo estimativas de performance artificialmente otimistas.

```
ERRO comum — Data Leakage:
  1. Normalizar X_train inteiro com scaler.fit_transform(X_train)
  2. Rodar cross_val_score(model, X_train_scaled, y_train, cv=5)
  → A normalização usou estatísticas de TODOS os folds incluindo validação
  → AUC estimada é otimista

CORRETO — Pipeline integrado:
  O scaler deve ser fit apenas no treino de cada fold
  e apenas aplicado (transform) no fold de validação
  → Usar sklearn Pipeline resolve isso automaticamente
```

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

# Pipeline: cada etapa é ajustada apenas no treino de cada fold
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    use_label_encoder=False,
                    eval_metric='auc',
                    random_state=42,
                    n_jobs=-1
                ))
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)

print(f"AUC média (pipeline sem leakage): {scores.mean():.4f} ± {scores.std():.4f}")
```

## 7. CV para Seleção de Modelos e Relatório Final

A aplicação mais importante do Cross-Validation no CRISP é comparar diferentes algoritmos de forma justa antes de escolher o modelo final.

```python
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'XGBoost':             XGBClassifier(n_estimators=300, learning_rate=0.05,
                               use_label_encoder=False, eval_metric='auc',
                               random_state=42, n_jobs=-1),
    'LightGBM':            LGBMClassifier(n_estimators=300, learning_rate=0.05,
                               random_state=42, verbose=-1, n_jobs=-1)
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    cv_result = cross_validate(
        model, X_train, y_train,
        cv=skf,
        scoring='roc_auc',
        return_train_score=True,
        n_jobs=-1
    )
    results.append({
        'Modelo':           name,
        'AUC Val Média':    cv_result['test_score'].mean(),
        'AUC Val Std':      cv_result['test_score'].std(),
        'AUC Treino Média': cv_result['train_score'].mean(),
        'Gap Treino-Val':   cv_result['train_score'].mean() - cv_result['test_score'].mean()
    })

df_results = pd.DataFrame(results).sort_values('AUC Val Média', ascending=False)
df_results = df_results.round(4)
print(df_results.to_string(index=False))

# Gap > 0.05 indica overfitting — revisar regularização
```

## 8. Comparativo dos Métodos de CV

| Método | Quando usar | Vantagem | Cuidado |
| --- | --- | --- | --- |
| K-Fold | Regressão, dados i.i.d. | Simples e eficiente | Não preserva proporção de classes |
| Stratified K-Fold | Classificação (padrão) | Preserva proporção de classes | Não funciona para séries temporais |
| TimeSeriesSplit | Séries temporais | Sem leakage temporal | Não paralelizável entre folds |
| LOOCV (K=N) | Datasets muito pequenos | Mínimo viés | Extremamente lento |
| GroupKFold | Dados com grupos (pacientes, lojas) | Evita vazamento entre grupos | Requer coluna de grupo |

### Regra prática para o CRISP

- **Classificação:** sempre `StratifiedKFold(n_splits=5)`
- **Regressão:** `KFold(n_splits=5)`
- **Séries temporais:** `TimeSeriesSplit(n_splits=5)`
- **Pré-processamento:** sempre dentro de um `Pipeline` — nunca antes do CV
- **Comparação de modelos:** usar `cross_validate` com `return_train_score=True` para monitorar overfitting
- **Conjunto de teste final:** avaliado apenas **uma vez**, no modelo final treinado — nunca durante o processo de seleção