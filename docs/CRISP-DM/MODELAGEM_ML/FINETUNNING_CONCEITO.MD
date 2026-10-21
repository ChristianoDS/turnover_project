## O que é Fine-Tuning?

O Fine-Tuning é o processo de encontrar a combinação ótima de **hiperparâmetros** de um modelo de Machine Learning — aqueles valores que não são aprendidos durante o treinamento, mas sim definidos pelo cientista de dados **antes** do treino começar.

### Parâmetros vs Hiperparâmetros

- **Parâmetros** são aprendidos pelo modelo a partir dos dados: coeficientes da Regressão Logística, pesos de uma Rede Neural, valores das folhas na XGBoost
- **Hiperparâmetros** são definidos pelo cientista de dados antes do treino: learning rate, número de árvores, profundidade máxima, regularização L1/L2, num_leaves

### Por que Fine-Tuning é necessário?

O mesmo algoritmo pode ter performances radicalmente diferentes dependendo dos hiperparâmetros escolhidos. Um modelo mal ajustado pode:

- Fazer **underfitting** (modelo muito simples — bias alto)
- Fazer **overfitting** (modelo muito complexo — variância alta)
- Treinar desnecessariamente lento sem ganho de performance

### O problema do espaço de busca

O conjunto de todos os possíveis valores de hiperparâmetros forma um **espaço de busca** de alta dimensão. Testar todas as combinações possíveis é computacionalmente inviável — por isso existem estratégias inteligentes de busca.

```python
# Exemplo de espaço de busca para XGBoost
# 4 x 3 x 5 x 4 x 4 x 4 x 4 = 15.360 combinações possíveis

param_space = {
    'n_estimators':    [100, 500, 1000, 2000],
    'learning_rate':   [0.01, 0.05, 0.1],
    'max_depth':       [3, 4, 5, 6, 8],
    'min_child_weight':[1, 3, 5, 10],
    'subsample':       [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree':[0.6, 0.7, 0.8, 0.9],
    'reg_lambda':      [0.1, 1.0, 5.0, 10.0],
}

# Testar todas: 15.360 modelos x 5-fold CV = 76.800 treinamentos
# Com 1 min por treino = 53 dias. Impraticável.
```

## 3. Técnica 1: Grid Search

O Grid Search é a abordagem mais simples e sistemática: define-se uma grade (**grid**) discreta de valores para cada hiperparâmetro e **testa-se exaustivamente todas as combinações possíveis**. Nenhuma combinação é pulada.

### Como funciona

Imagine dois hiperparâmetros: `max_depth` com 3 opções e `learning_rate` com 3 opções. O Grid Search forma uma grade 3×3 = 9 pontos e treina o modelo em cada um deles com cross-validation. O ponto com melhor métrica de validação é escolhido.

### Vantagens

- Garantia de encontrar o **melhor valor dentro da grade definida**
- Resultado completamente reproduzível e determinístico
- Fácil de entender e interpretar

### Desvantagens

- **Escala exponencialmente** com o número de hiperparâmetros e valores: K hiperparâmetros com V valores cada = V^K combinações
- Inviável para espaços de busca grandes
- Desperdiça recursos testando combinações ruins que poderiam ser evitadas

### Quando usar

- Poucos hiperparâmetros (2–3) e espaço pequeno
- Quando é necessário garantir a busca exaustiva
- Refinamento fino após uma busca aleatória identificar regiões promissoras

### Implementação

```python
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Espaço de busca PEQUENO — grid search só funciona assim
param_grid = {
    'max_depth':    [3, 5, 7],
    'learning_rate':[0.01, 0.05, 0.1],
    'n_estimators': [100, 300, 500]
}
# 3 x 3 x 3 = 27 combinações x 5-fold = 135 treinamentos

model = XGBClassifier(
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    refit=True,      # retreina o melhor modelo no dataset completo
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Melhores parâmetros: {grid_search.best_params_}")
print(f"Melhor AUC (CV):     {grid_search.best_score_:.4f}")
print(f"AUC Validação:       {roc_auc_score(y_val, grid_search.best_estimator_.predict_proba(X_val)[:,1]):.4f}")
```

## 4. Técnica 2: Random Search

O Random Search abandona a ideia de testar todas as combinações e em vez disso **sorteia aleatoriamente N combinações** do espaço de busca. Cada hiperparâmetro pode ser definido como uma distribuição de probabilidade — não apenas uma lista discreta — permitindo explorar valores contínuos.

### A intuição por trás do Random Search

Na prática, poucos hiperparâmetros realmente importam para a performance final — o restante tem efeito marginal. O Grid Search desperdiça recursos testando todas as combinações dos parâmetros irrelevantes. O Random Search, ao amostrar aleatoriamente, naturalmente cobre uma faixa maior de valores dos parâmetros importantes com o mesmo orçamento computacional.

### Vantagens

- **Mais eficiente que Grid Search** para espaços grandes — encontra boas soluções com muito menos avaliações
- Suporta distribuições contínuas (ex: `loguniform`, `uniform`) — não apenas valores discretos
- Fácil de aumentar o orçamento: basta aumentar `n_iter`
- Paralelizável: cada avaliação é independente

### Desvantagens

- Não garante encontrar o ótimo global (nem o melhor dentro de qualquer grade)
- Resultados variam entre execuções (sem `random_state` fixo)
- Não aprende com avaliações anteriores — não há inteligência na busca

### Quando usar

- Espaços de busca grandes (muitos hiperparâmetros)
- Orçamento computacional limitado
- Primeira exploração do espaço antes de um refinamento mais fino
- Quando não se sabe quais hiperparâmetros são mais importantes

### Implementação

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform, randint
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Espaço de busca com distribuições contínuas
param_dist = {
    'n_estimators':      randint(100, 2000),         # inteiro uniforme
    'learning_rate':     loguniform(0.005, 0.3),     # log-uniforme (ideal para LR)
    'max_depth':         randint(3, 10),
    'min_child_weight':  randint(1, 20),
    'subsample':         uniform(0.5, 0.5),          # uniforme entre 0.5 e 1.0
    'colsample_bytree':  uniform(0.5, 0.5),
    'reg_lambda':        loguniform(0.01, 10),
    'reg_alpha':         loguniform(0.001, 1)
}

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=100,          # número de combinações sorteadas
    scoring='roc_auc',
    cv=5,
    refit=True,
    random_state=42,
    verbose=1,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print(f"Melhores parâmetros: {random_search.best_params_}")
print(f"Melhor AUC (CV):     {random_search.best_score_:.4f}")
print(f"AUC Validação:       {roc_auc_score(y_val, random_search.best_estimator_.predict_proba(X_val)[:,1]):.4f}")
```