## Fase 1: O Treinamento

A Random Forest é um algoritmo de **aprendizado por conjunto (ensemble)** que combina múltiplas Árvores de Decisão para produzir previsões mais robustas e generalizáveis do que qualquer árvore individual conseguiria sozinha.

O nome revela a essência: "Random" porque introduz aleatoriedade em duas dimensões durante o treinamento; "Forest" porque o resultado é uma floresta de árvores independentes que votam juntas.

O princípio central é o da **Sabedoria das Multidões**: muitos modelos fracos e independentes, quando combinados, produzem um modelo forte. Cada árvore individual comete erros diferentes — ao agregar suas previsões, os erros se cancelam e o acerto se amplifica.

### O que entra no modelo?

- **Entradas (X):** variáveis preditoras numéricas e/ou categóricas (encodadas). Random Forest é invariante à escala — **não precisa de normalização**
- **Saída esperada (y):** variável resposta — binária ou multiclasse (classificação) ou contínua (regressão)
- **Hiperparâmetros principais:** n_estimators (nº de árvores), max_features (features por split), max_depth (profundidade), min_samples_split

### O que o modelo aprende?

Cada árvore aprende um conjunto de **regras de divisão (splits)** baseadas nas features. Cada split seleciona a feature e o valor de corte que melhor separa as classes no subconjunto de dados daquela árvore. O conjunto de splits de uma árvore forma sua estrutura — e estruturas diferentes emergem graças à aleatoriedade injetada durante o treinamento.

### O que o modelo produz?

- **Classificação:** probabilidade de cada classe via votação proporcional entre as árvores. A classe com mais votos é a predição final
- **Regressão:** média das predições contínuas de todas as árvores
- **Feature Importance:** ranking de importância de cada variável, calculado como contribuição média na redução de impureza ao longo de todas as árvores

### Funcionamento geral

O treinamento consiste em construir N árvores independentes em paralelo. Cada árvore recebe um subconjunto diferente dos dados (Bootstrap) e, em cada nó de divisão, enxerga apenas um subconjunto aleatório das features (Feature Randomness). Após o treinamento, a previsão de uma nova observação passa por todas as árvores e o resultado é agregado — votação para classificação, média para regressão.

## 2. Fase 2: Processo de Aprendizado

O aprendizado da Random Forest acontece em 5 etapas que se repetem para cada uma das N árvores da floresta. Entender a sequência é entender o que torna o algoritmo simultaneamente robusto e diverso.

### 2.1. Etapa 1 — Bootstrap Sampling

**O que é:** para cada árvore a ser construída, o algoritmo sorteia com reposição um subconjunto do dataset original com o mesmo tamanho N. Esse processo se chama Bootstrap.

**Entrada:** dataset completo com N observações e M features

**Saída:** N subsets bootstrap diferentes — um por árvore. Cada subset contém ~63.2% das observações únicas originais (as demais são repetições)

**Próxima etapa:** cada árvore é construída exclusivamente sobre seu subset bootstrap — nunca vê os dados das outras árvores

### 2.2. Etapa 2 — Seleção Aleatória de Features em Cada Nó

**O que é:** ao construir cada nó de divisão de cada árvore, o algoritmo não avalia todas as M features disponíveis. Em vez disso, sorteia aleatoriamente um subconjunto de k features (onde k < M) e só busca o melhor split dentro dessas k features.

**Entrada:** nó atual da árvore + conjunto completo de M features

**Saída:** subconjunto de k features candidatas ao split nesse nó

**Próxima etapa:** dentre as k features sorteadas, encontrar o melhor ponto de corte

### 2.3. Etapa 3 — Seleção do Melhor Split (Critério de Impureza)

**O que é:** para cada feature candidata (dentre as k sorteadas), o algoritmo testa todos os valores de corte possíveis e escolhe o par (feature, valor) que mais reduz a impureza dos dados naquele nó.

**Entrada:** k features candidatas + dados do nó atual

**Saída:** par (feature*, threshold*) que maximiza a redução de impureza

**Próxima etapa:** dividir o nó em dois nós filhos com base no split escolhido

### 2.4. Etapa 4 — Crescimento Recursivo da Árvore

**O que é:** o processo de divisão (etapas 2 e 3) se repete recursivamente em cada nó filho até que um critério de parada seja atingido — profundidade máxima, número mínimo de amostras, ou impureza zero.

**Entrada:** nós filhos gerados pelo split anterior

**Saída:** árvore completamente crescida com folhas que contêm as predições (classe majoritária ou média)

**Próxima etapa:** a árvore está treinada. O ciclo (etapas 1–4) recomeça para a próxima árvore da floresta

### 2.5. Etapa 5 — Agregação das Predições (Voting / Averaging)

**O que é:** ao receber uma nova observação, ela percorre todas as N árvores da floresta. Cada árvore produz sua predição independente. O resultado final é a agregação dessas N predições.

**Entrada:** predições individuais das N árvores para uma observação

**Saída:** classe com mais votos (classificação) ou média dos valores (regressão)

**Próxima etapa:** fim — a Random Forest está treinada e pronta para predição

## 3. Fase 3: Detalhamento das Etapas

Agora veremos as fórmulas concretas de cada etapa e como elas transformam a entrada na saída.

### 3.1. Etapa 1 — Bootstrap Sampling: Fórmula e Execução

**Fórmula:**

```
Para cada árvore t = 1, 2, ..., T:
  D_t = Bootstrap(D, N)
      = sortear N amostras de D com reposição

Propriedade fundamental:
  P(observação i não é selecionada) = (1 - 1/N)^N → 1/e ≈ 36.8%
  ∴ cada bootstrap contém ~63.2% de amostras únicas

Amostras não selecionadas = Out-Of-Bag (OOB)
  → usadas para validação interna sem necessidade de holdout
```

**Como é executado:**

1. Para cada uma das T árvores, gera um índice de N posições sorteadas com reposição de {0, 1, ..., N-1}
2. Os índices repetidos aumentam o peso de certas observações naquela árvore
3. Os índices nunca selecionados formam o conjunto OOB daquela árvore
4. O OOB permite estimar o erro de generalização sem separar um conjunto de validação

```python
import numpy as np

def bootstrap_sample(X, y, random_state=None):
    rng = np.random.RandomState(random_state)
    N = len(X)
    indices = rng.choice(N, size=N, replace=True)   # com reposição
    oob_indices = np.setdiff1d(np.arange(N), indices)
    return X[indices], y[indices], oob_indices

# Exemplo
X_boot, y_boot, oob_idx = bootstrap_sample(X_train.values, y_train.values, random_state=42)
print(f"Amostras únicas no bootstrap: {len(np.unique(oob_idx))} / {len(X_train)}")
print(f"OOB size: {len(oob_idx)} ({100*len(oob_idx)/len(X_train):.1f}%)")
```

### 3.2. Etapa 2 — Feature Randomness: Fórmula e Execução

**Fórmula:**

```
Em cada nó de divisão:
  F_candidatas = sortear k features de {f₁, f₂, ..., f_M} sem reposição

Valores padrão de k (max_features):
  Classificação: k = √M   (raiz quadrada do total de features)
  Regressão:     k = M/3  (um terço do total de features)

Efeito sobre a correlação entre árvores:
  k = M   → árvores correlacionadas (sem aleatoriedade → Bagging puro)
  k = √M  → decorrelação ótima para a maioria dos problemas
  k = 1   → máxima aleatoriedade, árvores muito diversas mas fracas
```

**Como é executado:**

1. Em cada nó, sorteia k índices de features sem reposição
2. Apenas essas k features são avaliadas para o split
3. O sorteio é independente em cada nó — a mesma feature pode ser sorteada em nós diferentes
4. Isso garante que features poderosas não dominem todas as árvores

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# max_features controla k
M = X_train.shape[1]
k_sqrt = int(np.sqrt(M))   # padrão para classificação

model = RandomForestClassifier(
    n_estimators=200,
    max_features=k_sqrt,    # ou 'sqrt' (string equivalente)
    random_state=42
)
print(f"Total features (M): {M} | Features por nó (k): {k_sqrt}")
```

### 3.3. Etapa 3 — Critério de Impureza: Fórmula e Execução

**Critério de Classificação — Índice de Gini:**

```
Gini(nó) = 1 - Σ_c p_c²

onde:
  p_c = proporção de amostras da classe c no nó
  Σ vai sobre todas as C classes

Gini = 0   → nó puro (apenas uma classe)
Gini = 0.5 → nó totalmente impuro (2 classes, 50/50)

Redução de Gini (Information Gain):
  ΔGini = Gini(pai) - [N_esq/N * Gini(esq) + N_dir/N * Gini(dir)]

O split escolhido é o que maximiza ΔGini.
```

**Critério de Regressão — MSE / Variância:**

```
Impureza(nó) = Var(y_nó) = (1/N) Σ (yᵢ - ȳ)²

Redução de variância:
  ΔVar = Var(pai) - [N_esq/N * Var(esq) + N_dir/N * Var(dir)]
```

**Como é executado:**

1. Para cada uma das k features candidatas, ordena os valores únicos
2. Testa cada valor único como threshold de corte
3. Para cada corte, calcula a impureza dos dois nós filhos resultantes
4. Calcula a redução de impureza ponderada
5. Seleciona o par (feature, threshold) com máxima redução

```python
import numpy as np

def gini_impurity(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

def best_split(X_node, y_node, k_features):
    best_gain, best_feat, best_thresh = -np.inf, None, None
    M = X_node.shape[1]
    # Sortear k features
    feat_indices = np.random.choice(M, size=k_features, replace=False)

    gini_pai = gini_impurity(y_node)
    N = len(y_node)

    for feat in feat_indices:
        thresholds = np.unique(X_node[:, feat])
        for thresh in thresholds:
            mask = X_node[:, feat] <= thresh
            y_esq, y_dir = y_node[mask], y_node[~mask]
            if len(y_esq) == 0 or len(y_dir) == 0:
                continue
            gini_split = (len(y_esq)/N)*gini_impurity(y_esq) + \
                         (len(y_dir)/N)*gini_impurity(y_dir)
            gain = gini_pai - gini_split
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat, thresh
    return best_feat, best_thresh, best_gain
```

### 3.4. Etapa 4 — Crescimento Recursivo: Fórmula e Execução

**Critérios de parada:**

```
A recursão para quando qualquer condição é satisfeita:

  1. profundidade atual ≥ max_depth
  2. amostras no nó ≤ min_samples_split
  3. amostras em qualquer folha < min_samples_leaf
  4. Gini(nó) = 0  →  nó já é puro
  5. Nenhum split melhora a impureza

Se nenhuma condição for atingida → dividir e chamar recursivamente
Se alguma condição for atingida  → transformar em folha
  Folha (classificação): classe majoritária das amostras no nó
  Folha (regressão):     média das amostras no nó
```

**Como é executado:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Treinamento completo
model = RandomForestClassifier(
    n_estimators=300,       # T árvores
    max_features='sqrt',    # k = √M features por nó
    max_depth=None,         # cresce até folhas puras (default)
    min_samples_split=2,    # mín. amostras para dividir um nó
    min_samples_leaf=1,     # mín. amostras por folha
    bootstrap=True,         # ativa o Bootstrap Sampling
    oob_score=True,         # calcula erro OOB automaticamente
    n_jobs=-1,              # paraleliza nas CPUs disponíveis
    random_state=42
)

model.fit(X_train, y_train)

# OOB Score — estimativa de generalização sem validação separada
print(f"OOB Score (AUC proxy): {model.oob_score_:.4f}")

# Performance no conjunto de validação
y_proba = model.predict_proba(X_val)[:, 1]
print(f"AUC Validação: {roc_auc_score(y_val, y_proba):.4f}")
```

### 3.5. Etapa 5 — Agregação: Fórmula e Execução

**Fórmula — Classificação (Soft Voting):**

```
P(y=c | X) = (1/T) Σ_{t=1}^{T} p_t(y=c | X)

onde:
  T        = número de árvores
  p_t(y=c) = proporção de amostras da classe c na folha
             em que X cai na árvore t

Classe predita = argmax_c P(y=c | X)
```

**Fórmula — Regressão (Averaging):**

```
ŷ = (1/T) Σ_{t=1}^{T} ŷ_t(X)

onde ŷ_t(X) é a média das amostras na folha da árvore t
```

**Feature Importance — Impurity-based:**

```
FI(f) = (1/T) Σ_t Σ_{nós n que usam f} [N_n/N * ΔImpureza(n)]

onde:
  N_n = amostras que passam pelo nó n
  N   = total de amostras de treino
  ΔImpureza(n) = redução de Gini/MSE no nó n
```

```python
import pandas as pd
import numpy as np

# Probabilidades agregadas (soft voting)
y_proba = model.predict_proba(X_val)[:, 1]
y_pred  = model.predict(X_val)

# Feature Importance
fi = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("Top 10 features mais importantes:")
print(fi.head(10).round(4))

# Threshold de seleção: features acima da média
selected = fi[fi >= fi.mean()].index.tolist()
print(f"\nFeatures selecionadas: {len(selected)}")
```

## 4. Fase 4: Aprofundando a Matemática

Nesta fase exploramos cada ferramenta matemática em profundidade — o que é, por que existe e como contribui para a força do algoritmo.

### 4.1. Bootstrap Sampling — Por que a reamostragem com reposição cria diversidade?

O Bootstrap é uma técnica estatística criada por Bradley Efron em 1979 para estimar a distribuição amostral de uma estatística sem fazer suposições sobre a distribuição dos dados.

**Por que ele funciona para criar diversidade:** cada subset bootstrap é diferente porque o sorteio é aleatório. Algumas observações aparecem 0 vezes em uma árvore e 3 vezes em outra. Isso significa que cada árvore "aprende" uma visão ligeiramente diferente da distribuição dos dados — produzindo erros não-correlacionados entre as árvores.

**A propriedade dos ~63.2%:** a probabilidade de uma observação específica NÃO ser sorteada em uma extração é (1 - 1/N). Em N extrações independentes: (1 - 1/N)^N → e^(-1) ≈ 0.368. Portanto, ~36.8% das observações únicas ficam de fora de cada bootstrap — formando o conjunto OOB.

**O OOB como validação gratuita:**

```
Para cada observação i:
  OOB_trees(i) = árvores que NÃO usaram i no treino
                 (~36.8% das T árvores)

OOB prediction(i) = agregação apenas das OOB_trees(i)

OOB Score = métrica calculada sobre todas as observações
            usando apenas suas OOB predictions

Benefício: é uma estimativa quase equivalente à validação
cruzada k-fold, sem custo computacional adicional.
```

### 4.2. Feature Randomness — Por que sortear features reduz o erro?

Este é o ingrediente que transforma Bagging em Random Forest. Sem ele, todas as árvores tenderiam a usar as mesmas features poderosas nos primeiros splits — produzindo árvores altamente correlacionadas.

**O problema da correlação entre árvores:** considere T árvores correlacionadas com correlação ρ entre si e variância σ² cada. A variância da média é:

```
Var(média de T árvores correlacionadas) = ρσ² + (1-ρ)σ²/T

Quando ρ → 1 (árvores idênticas): Var ≈ σ²  → sem ganho!
Quando ρ → 0 (árvores independentes): Var ≈ σ²/T → erro cai com √T

A Feature Randomness reduz ρ:
  Ao esconder features poderosas em alguns nós, força as
  árvores a explorar features alternativas — tornando-as
  estruturalmente diferentes umas das outras.
```

**O trade-off de k (max_features):**

- k grande → cada árvore é mais forte individualmente, mas as árvores são mais correlacionadas
- k pequeno → cada árvore é mais fraca, mas as árvores são menos correlacionadas
- k = √M equilibra esses dois efeitos para a maioria dos problemas de classificação

### 4.3. Índice de Gini — O que ele mede e por que é preferido à Entropia?

O Índice de Gini mede a probabilidade de classificar incorretamente uma observação sorteada aleatoriamente de um nó se ela fosse rotulada conforme a distribuição de classes do nó.

**Interpretação geométrica:**

```
Gini(nó) = 1 - Σ p_c²

Para 2 classes (p e 1-p):
  Gini = 1 - (p² + (1-p)²) = 2p(1-p)

  Gini = 0.0  quando p=0 ou p=1  → nó puro
  Gini = 0.5  quando p=0.5       → máxima impureza

Comparação com Entropia:
  Entropia = -Σ p_c * log₂(p_c)
  Entropia máx = 1.0 (base 2, 2 classes)

  Gini é computacionalmente mais barato (sem log)
  Ambos produzem splits similares na prática
  Gini tende a isolar a classe mais frequente
  Entropia tende a criar árvores mais balanceadas
```

**Por que a redução de Gini ponderada pelo tamanho do nó:** um split que produz dois nós puros mas desbalanceados (ex: 995 vs 5 amostras) é menos valioso do que um split que produz dois nós igualmente puros e equilibrados. A ponderação por N_filho/N_pai captura isso.

### 4.4. Bagging — O princípio matemático por trás da redução de variância

O Bootstrap Aggregating (Bagging), proposto por Leo Breiman em 1996, é o mecanismo estatístico que fundamenta a Random Forest.

**O problema que o Bagging resolve:** árvores de decisão individuais têm **alta variância** — são sensíveis aos dados de treino e tendem ao overfitting. Se o conjunto de treino mudar ligeiramente, a árvore resultante pode ser completamente diferente.

**Como o Bagging reduz a variância:**

```
Seja f̂_t(x) a predição da árvore t. O Bagging produz:
  f̂_bag(x) = (1/T) Σ f̂_t(x)

Se as árvores fossem independentes (ρ=0):
  Var(f̂_bag) = σ²/T → variância cai linearmente com T

Na prática (ρ > 0):
  Var(f̂_bag) = ρσ² + (1-ρ)σ²/T
             = ρσ² + σ²(1-ρ)/T

O primeiro termo (ρσ²) é irredutível pelo número de árvores T.
Só a Feature Randomness (que reduz ρ) consegue diminuí-lo.

Consequência prática:
  - Aumentar T além de ~200-500 tem retorno marginal decrescente
  - Reduzir ρ (aumentando aleatoriedade) frequentemente ajuda mais
    do que aumentar T indefinidamente
```

### 4.5. Feature Importance — O que ela mede e quais suas limitações?

A Feature Importance baseada em impureza (MDI — Mean Decrease Impurity) é calculada durante o treinamento como subproduto gratuito da construção das árvores.

**O que ela mede:** o quanto, em média, uma feature contribui para reduzir a impureza dos nós onde ela é usada, ponderado pela fração de amostras que passa por esses nós, acumulado ao longo de todas as T árvores.

**Limitações importantes:**

```
1. Viés para alta cardinalidade:
   Features numéricas contínuas têm mais valores únicos
   para testar como threshold → mais chances de aparecer
   em splits por puro acaso → recebem importância inflada

2. Correlação entre features:
   Quando duas features são correlacionadas, a importância
   se divide arbitrariamente entre elas — nenhuma aparece
   como importante mesmo que ambas sejam relevantes

3. Medida de treino, não de generalização:
   MDI mede importância nos dados de TREINO — pode superestimar
   features que causam overfitting

Alternativa mais robusta: Permutation Importance
  Para cada feature f:
    1. Embaralha os valores de f no conjunto de validação
    2. Mede a queda na performance do modelo
    3. Importância = queda média na AUC/accuracy
  → Mede impacto real na performance, não no treino
```

```python
from sklearn.inspection import permutation_importance

# Permutation Importance — mais confiável para seleção de features
result = permutation_importance(
    model, X_val, y_val,
    n_repeats=10,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)

perm_fi = pd.Series(
    result.importances_mean,
    index=X_train.columns
).sort_values(ascending=False)

print("Permutation Importance (top 10):")
print(perm_fi.head(10).round(4))
```

### 4.6. Hiperparâmetros críticos e como ajustá-los

| Hiperparâmetro | O que controla | Efeito de aumentar | Valor padrão |
| --- | --- | --- | --- |
| n_estimators | Número de árvores | Reduz variância (até platô) | 100 |
| max_features | Features por nó (k) | Aumenta correlação entre árvores | 'sqrt' |
| max_depth | Profundidade máxima | Aumenta bias se muito baixo | None |
| min_samples_split | Mín. amostras para dividir | Reduz overfitting | 2 |
| min_samples_leaf | Mín. amostras por folha | Suaviza fronteira de decisão | 1 |
| bootstrap | Ativa Bootstrap | Desativar = Bagging puro | True |
| class_weight | Peso das classes | Corrige desbalanceamento | None |

```python
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Grid de busca
param_dist = {
    'n_estimators':      [100, 200, 300, 500],
    'max_features':      ['sqrt', 'log2', 0.3, 0.5],
    'max_depth':         [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'class_weight':      [None, 'balanced']
}

rf = RandomForestClassifier(n_jobs=-1, random_state=42)

search = RandomizedSearchCV(
    rf, param_dist,
    n_iter=50,
    scoring='roc_auc',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
search.fit(X_train, y_train)

print(f"Melhores hiperparâmetros: {search.best_params_}")
print(f"Melhor AUC CV: {search.best_score_:.4f}")

best_model = search.best_estimator_
```