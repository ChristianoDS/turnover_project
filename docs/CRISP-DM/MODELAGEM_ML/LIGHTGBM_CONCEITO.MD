## Fase 1: O Treinamento

A LightGBM (Light Gradient Boosting Machine) é um algoritmo de **ensemble sequencial baseado em Gradient Boosting**, desenvolvido pela Microsoft em 2017. Assim como a XGBoost, ela constrói árvores em sequência, onde cada nova árvore aprende a corrigir os erros residuais das anteriores. A diferença está em **como** ela constrói cada árvore.

A XGBoost e o Gradient Boosting tradicional crescem árvores **por nível** (level-wise): a cada passo, dividem todos os nós do mesmo nível antes de ir ao próximo. A LightGBM cresce árvores **por folha** (leaf-wise): a cada passo, divide apenas o nó que proporciona a maior redução de loss — independente do nível. Isso produz árvores mais assimtéricas, mais profundas em alguns ramos, e geralmente mais precisas com menos iterações.

Além disso, a LightGBM introduz duas inovações próprias que a tornam dramaticamente mais rápida em datasets grandes: **GOSS** (Gradient-based One-Side Sampling) e **EFB** (Exclusive Feature Bundling).

### O que entra no modelo?

- **Entradas (X):** variáveis preditoras numéricas e/ou categóricas. LightGBM suporta features categóricas nativamente — sem necessidade de OneHot Encoding. Invariante à escala — **não precisa de normalização**
- **Saída esperada (y):** variável resposta — binária ou multiclasse (classificação) ou contínua (regressão)
- **Hiperparâmetros principais:** n_estimators, learning_rate, num_leaves, max_depth, min_child_samples, subsample, colsample_bytree, reg_alpha (L1), reg_lambda (L2)

### O que o modelo aprende?

Assim como a XGBoost, a LightGBM aprende um **ensemble aditivo de árvores de decisão**. Cada árvore aprende a prever os **gradientes negativos** da função de perda em relação às previsões acumuladas. A diferença estrutural está na forma como as árvores são construídas internamente: leaf-wise ao invés de level-wise.

### O que o modelo produz?

- **Classificação binária:** probabilidade via Sigmoid: P(y=1 | X)
- **Classificação multiclasse:** probabilidades por classe via Softmax
- **Regressão:** valor contínuo predito diretamente
- **Feature Importance:** split (número de vezes que a feature foi usada) ou gain (redução total de loss)

### Funcionamento geral

O treinamento começa com uma previsão inicial constante. Em cada iteração: (1) calcula-se gradientes e hessianos para todas as observações, (2) aplica-se GOSS para amostrar apenas as observações mais informativas, (3) aplica-se EFB para reduzir a dimensionalidade das features, (4) constrói-se uma árvore leaf-wise usando histogramas binários, (5) adiciona-se a árvore ao ensemble com peso η. Esse ciclo se repete T vezes.

## 2. Fase 2: Processo de Aprendizado

O aprendizado da LightGBM acontece em 6 etapas por iteração. As etapas 1, 2, 5 e 6 são compartilhadas com a XGBoost. As etapas 3 e 4 são as inovações exclusivas da LightGBM.

### 2.1. Etapa 1 — Previsão Inicial (Base Score)

**O que é:** antes de qualquer árvore, o modelo define uma previsão constante inicial para todas as observações — o ponto de partida do ensemble.

**Entrada:** variável resposta y do conjunto de treino

**Saída:** valor constante ŷ₀ aplicado a todas as N observações

**Próxima etapa:** calcular os gradientes e hessianos a partir de ŷ₀ e y

### 2.2. Etapa 2 — Cálculo dos Gradientes e Hessianos

**O que é:** para cada observação, calcula-se a primeira derivada (gradiente g) e a segunda derivada (hessiano h) da função de perda em relação à previsão atual. Esses valores representam o erro e a curvatura do erro de cada observação.

**Entrada:** previsões acumuladas ŷ e rótulos reais y

**Saída:** vetores g = (g₁,...,gₙ) e h = (h₁,...,hₙ)

**Próxima etapa:** os gradientes alimentam o GOSS para seleção das observações mais relevantes

### 2.3. Etapa 3 — GOSS: Amostragem Baseada em Gradiente

**O que é:** o GOSS (Gradient-based One-Side Sampling) é a primeira inovação exclusiva da LightGBM. Em vez de usar todas as N observações para construir cada árvore, o GOSS seleciona apenas um subconjunto informativo: mantendo **todas** as observaões com gradientes altos (grandes erros) e apenas uma amostra aleatória das observações com gradientes baixos (pequenos erros).

**Entrada:** vetores g de todas as N observações + hiperparâmetros a (fração top) e b (fração aleatória)

**Saída:** subset reduzido de observações com pesos ajustados para manter a distribuição original

**Próxima etapa:** esse subset alimenta o EFB e a construção da árvore

### 2.4. Etapa 4 — EFB: Empacotamento Exclusivo de Features

**O que é:** o EFB (Exclusive Feature Bundling) é a segunda inovação exclusiva da LightGBM. Ele identifica features que raramente são simultaneamente não-zero (features "exclusivas" entre si) e as agrupa em um único bundle — reduzindo o número efetivo de features sem perda significativa de informação.

**Entrada:** matriz de features X com M colunas

**Saída:** matriz reduzida com K < M colunas (bundles de features)

**Próxima etapa:** a matriz reduzida é usada para construir a árvore leaf-wise

### 2.5. Etapa 5 — Construção Leaf-Wise com Histogramas

**O que é:** a árvore é construída usando a estratégia leaf-wise (Best-First Tree Growth): a cada passo, divide-se apenas a folha que proporciona o maior ganho de loss, independentemente do nível. Para encontrar os splits eficientemente, a LightGBM usa histogramas binários em vez de varrer todos os valores únicos.

**Entrada:** gradientes g e hessianos h do subset GOSS + features bundled do EFB

**Saída:** estrutura completa da árvore com splits e valores das folhas

**Próxima etapa:** a árvore é adicionada ao ensemble com peso η

### 2.6. Etapa 6 — Atualização do Ensemble

**O que é:** a nova árvore é incorporada ao ensemble com peso controlado pela learning rate η. As previsões acumuladas são atualizadas.

**Entrada:** previsões acumuladas ŷ_{t-1} + nova árvore f_t + learning rate η

**Saída:** previsões atualizadas ŷ_t = ŷ_{t-1} + η · f_t(X)

**Próxima etapa:** o ciclo reinicia na Etapa 2 — até T árvores ou early stopping

## 3. Fase 3: Detalhamento das Etapas

Agora veremos as fórmulas concretas de cada etapa, com foco nas inovações exclusivas da LightGBM.

### 3.1. Etapa 1/2 — Base Score e Gradientes: Fórmula e Execução

**Fórmulas (idênticas à XGBoost):**

```
Base score (classificação binária):
  ŷ₀ = log( ȳ / (1 - ȳ) )   → log-odds da taxa de evento

Gradiente e Hessiano (Binary Cross-Entropy):
  pᵢ = sigmoid(ŷᵢ) = 1 / (1 + e^(-ŷᵢ))
  gᵢ  = pᵢ - yᵢ
  hᵢ  = pᵢ · (1 - pᵢ)
```

```python
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# Verificação da taxa de evento e base score
y_mean = y_train.mean()
base_score = np.log(y_mean / (1 - y_mean))
print(f"Taxa de evento: {y_mean:.4f} | Base score: {base_score:.4f}")
```

### 3.2. Etapa 3 — GOSS: Fórmula e Execução

**Fórmula:**

```
Dados: N observações com gradientes |gᵢ|
Hiperparâmetros: a (fração top), b (fração aleatória)

Passo 1 — Selecionar top-a:
  Ordenar observaões por |gᵢ| de forma decrescente
  A = {top a·N observações com maior |gᵢ|}

Passo 2 — Amostrar b das restantes:
  B = {b·(1-a)·N observações aleatórias do restante}

Passo 3 — Ajuste de peso:
  Observações em A → peso = 1.0
  Observações em B → peso = (1-a)/b
  O peso (1-a)/b amplifica as amostras de B
  para compensar sua sub-representação

Dataset efetivo: A ∪ B com |A ∪ B| << N
  Redução típica: de N para ~20-30% de N

Garantia teórica:
  A distribuição dos gradientes é preservada
  com alta probabilidade (bound de concentração)
```

**Como é executado:**

```python
def goss_sampling(g, a=0.2, b=0.1, random_state=42):
    """Gradient-based One-Side Sampling"""
    rng = np.random.RandomState(random_state)
    N = len(g)
    abs_g = np.abs(g)

    # Passo 1: top-a por gradiente
    n_top = int(N * a)
    top_idx = np.argsort(abs_g)[::-1][:n_top]  # maiores |g|

    # Passo 2: amostragem aleatória do restante
    rest_idx = np.argsort(abs_g)[::-1][n_top:]
    n_rand = int(len(rest_idx) * b)
    rand_idx = rng.choice(rest_idx, size=n_rand, replace=False)

    # Passo 3: pesos de compensação
    selected_idx = np.concatenate([top_idx, rand_idx])
    weights = np.ones(len(selected_idx))
    weights[n_top:] = (1 - a) / b   # amplifica amostras aleatórias

    print(f"GOSS: {N} → {len(selected_idx)} observações "
          f"({100*len(selected_idx)/N:.1f}%)")
    return selected_idx, weights
```

### 3.3. Etapa 4 — EFB: Fórmula e Execução

**Fórmula:**

```
Objetivo: encontrar K bundles de features tal que K << M

Condição de exclusividade:
  Features fᵢ e fⱼ são excluivas se:
  conflict(fᵢ, fⱼ) = |{x : fᵢ(x) ≠ 0 AND fⱼ(x) ≠ 0}| / N < γ
  onde γ é a taxa de conflito tolerada (padrão: 0)

Algoritmo de agrupamento:
  1. Construir grafo G onde nós = features
     e arestas = pares com conflict > γ
  2. Problema equivale ao Minimum Graph Coloring:
     features sem arestas (exclusivas) → mesmo bundle
     features com arestas (conflitantes) → bundles diferentes
  3. Greedy approximation ordena nós por grau (descrescente)
     e atribui ao menor bundle disponível sem conflito

Merge de features no bundle:
  Para fᵢ, fⱼ no mesmo bundle, com offsets:
  f_bundle(x) = fᵢ(x) + offset_j  se fⱼ(x) ≠ 0
              = fᵢ(x)              se fⱼ(x) = 0
  O offset garante que os ranges não se sobreponham

Eficiência:
  Dados esparsos (ex: após OneHot): 90%+ das features
  são exclusivas → redução de 10x ou mais em M
```

```python
import numpy as np

def efb_bundle(X, max_conflict_rate=0.0):
    """
    Simplified EFB: detecta features esparso-exclusivas
    e calcula taxa de conflito entre pares
    """
    M = X.shape[1]
    nonzero_mask = (X != 0)  # True onde feature é ativa

    conflict_matrix = np.zeros((M, M))
    for i in range(M):
        for j in range(i+1, M):
            both_nonzero = (nonzero_mask[:, i] & nonzero_mask[:, j]).sum()
            conflict_matrix[i, j] = both_nonzero / len(X)

    # Features que podem ser agrupadas (sem conflito)
    can_bundle = conflict_matrix <= max_conflict_rate
    n_potential_bundles = can_bundle.sum() // 2
    print(f"Pares exclusivos (sem conflito): {n_potential_bundles}")
    return conflict_matrix
```

### 3.4. Etapa 5 — Histogramas e Leaf-Wise: Fórmula e Execução

**Histograma binário — algoritmo de split:**

```
Para cada feature f com valores contínuos:
  1. Discretizar em B bins (padrão B=255)
     bin(x) = floor(x / (max-min) * B)

  2. Para cada bin b, acumular:
     G[b] = Σ gᵢ  para observações em bin b
     H[b] = Σ hᵢ  para observações em bin b

  3. Para cada threshold candidato t = 1,...,B-1:
     G_L = Σ_{b=0}^{t} G[b]     G_R = G_total - G_L
     H_L = Σ_{b=0}^{t} H[b]     H_R = H_total - H_L

     Gain(t) = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) - G_total²/(H_total+λ)] - γ

  4. Escolher t* = argmax Gain(t)

Complexidade:
  Level-wise (XGBoost): O(N · M)  por split
  LightGBM histograma: O(B · M)  por split  →  N >> B → muito mais rápido

Otimização extra — Histogram Subtraction:
  Se nó pai e filho esquerdo estão calculados:
  H_filho_direito = H_pai - H_filho_esquerdo
  → evita recalcular histograma do filho maior
```

**Leaf-wise vs Level-wise:**

```
Level-wise (XGBoost, GBDT clássico):
  Iteração 1: divide todos os nós do nível 0 (raiz)
  Iteração 2: divide todos os nós do nível 1
  Iteração 3: divide todos os nós do nível 2
  ...
  Produz árvores simétricas e balanceadas

Leaf-wise (LightGBM):
  Iteração 1: divide a folha com maior Gain (ex: raiz)
  Iteração 2: divide a folha com maior Gain do conjunto atual
  Iteração 3: divide a folha com maior Gain do conjunto atual
  ...
  Produz árvores assimétricas — alguns ramos muito mais profundos

Consequência:
  Mesma redução de loss com menos folhas (menos splits)
  Maior risco de overfitting sem controle de num_leaves
  num_leaves é o hiperparâmetro crítico da LightGBM
  (substitui max_depth como controle de complexidade)
```

```python
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# Dataset LightGBM nativo (mais eficiente)
dtrain = lgb.Dataset(X_train, label=y_train)
dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

params = {
    'objective':        'binary',
    'metric':           'auc',
    'boosting_type':    'gbdt',         # leaf-wise por padrão
    'num_leaves':       31,             # controle principal de complexidade
    'max_depth':        -1,             # -1 = sem limite (usa num_leaves)
    'learning_rate':    0.05,
    'min_child_samples': 20,            # mín. amostras por folha
    'subsample':        0.8,            # GOSS é ativado com boosting_type='goss'
    'colsample_bytree': 0.8,
    'reg_alpha':        0.0,            # L1
    'reg_lambda':       1.0,            # L2
    'max_bin':          255,            # bins do histograma
    'verbose':          -1,
    'seed':             42
}

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=True),
    lgb.log_evaluation(period=100)
]

model = lgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    valid_sets=[dval],
    callbacks=callbacks
)

y_proba = model.predict(X_val)
print(f"\nMelhor iteração: {model.best_iteration}")
print(f"AUC Validação:   {roc_auc_score(y_val, y_proba):.4f}")
```

## 4. Fase 4: Aprofundando a Matemática

Nesta fase exploramos cada ferramenta matemática em profundidade — o que é, por que existe e como contribui para a velocidade e precisão da LightGBM.

### 4.1. GOSS — Por que gradientes grandes são mais informativos?

O GOSS parte de uma observação fundamental do Gradient Boosting: **a contribuição de cada observação para o aprendizado é proporcional à magnitude do seu gradiente**.

**Intuição:** observações com gradiente alto são aquelas onde o modelo atual está mais errado — elas carregam mais informação sobre o que precisa ser corrigido. Observações com gradiente baixo já estão bem ajustadas — sua contribuição marginal para o split é pequena.

```
Fundamento teórico — ganho do split no Gradient Boosting:
  Gain ∝ Σ_{i ∈ nó} gᵢ²  (proporcional ao quadrado dos gradientes)

Observações com |gᵢ| ≈ 0:
  gᵢ² ≈ 0  → contribuição desprezível para o Gain
  Removê-las impacta minimamente a qualidade do split

Observações com |gᵢ| grande:
  gᵢ² >> 0  → contribuição relevante para o Gain
  Devem ser preservadas sem exceção

O peso (1-a)/b das amostras aleatórias:
  Garante que Σ_{i ∈ B} gᵢ · wᵢ ≈ Σ_{i ∈ rest} gᵢ
  Ou seja: a distribuição dos gradientes é preservada
  em expectativa mesmo após a redução do dataset

Ganho prático:
  Redução de 70-80% do dataset sem perda significativa
  de qualidade → speedup proporcional na construção
  do histograma e no cálculo dos splits
```

### 4.2. EFB — Por que features esparssas podem ser fundidas?

O EFB explora uma característica comum em dados reais de alta dimensionalidade: **features mutuamente exclusivas**. Isso ocorre naturalmente após OneHot Encoding (apenas uma categoria é ativa por vez), em dados de texto (a maioria das features TF-IDF é zero), e em dados transacionais esparsos.

```
Problema de colorir grafos (NP-hard no geral):
  Nós = features (M no total)
  Arestas = conflito entre features (ambas não-zero juntas)
  Cores = bundles
  Objetivo: colorir com mínimo de cores (bundles)

Aproximação greedy em O(M²):
  1. Calcular grau de cada nó (número de conflitos)
  2. Ordenar nós por grau decrescente
  3. Para cada feature, atribuir ao menor bundle
     onde nenhum conflito é introduzido

Merge dentro do bundle — offset por ranges:
  feature_1: valores em [0, 10]  → offset = 0
  feature_2: valores em [0,  5]  → offset = 11 (acima do max de feat_1)
  feature_3: valores em [0,  8]  → offset = 17 (acima do max de feat_2)

  bundle(x) = feat_1(x) + 0         se feat_1 ativo
            = feat_2(x) + 11        se feat_2 ativo
            = feat_3(x) + 17        se feat_3 ativo
            = 0                     se nenhum ativo

O histograma do bundle distingue as contribuições
de cada feature original via os offsets — nenhuma
informação de split é perdida quando conflict=0

Ganho prático:
  Dados com 1000 features após OneHot Encoding
  → EFB reduz para ~100-200 bundles efetivos
  → speedup de 5-10x na construção do histograma
```

### 4.3. Histogramas Binários — Como bins discretos aceleram sem perder qualidade?

A busca pelo melhor split é a etapa mais custosa na construção de qualquer árvore. O algoritmo naive varre todos os N valores únicos de cada feature — O(N·M) por nó. Com histogramas, isso cai para O(B·M) onde B << N.

```
Custo computacional comparado:
  Dados: N=1.000.000 observações, M=100 features, B=255 bins

  Algoritmo exato (sort-based):  O(N · M) = 10^8 operações por nó
  LightGBM histograma:           O(B · M) = 2.55×10^4 operações por nó
  Speedup: ~4.000x por nó

Qualidade dos splits:
  B=255 é suficiente para capturar quase todos os splits relevantes
  em dados reais — a perda de precisão por discretização é
  compensada pelo volume de dados processado

Histogram Subtraction — otimização adicional:
  Ao dividir um nó em filho_esq e filho_dir:
  H_filho_dir = H_pai - H_filho_esq
  → só é necessário construir o histograma do filho MENOR
  → filho maior é obtido por subtração em O(B) — gratuitamente
  → combinado com leaf-wise: sempre construído para a folha
     com MENOS amostras, a mais cara evitada
```

### 4.4. Leaf-Wise — Por que crescer pela melhor folha reduz o erro mais rápido?

A estratégia leaf-wise (Best-First) é uma heurística gulosa: em cada passo, maximiza a redução imediata da loss. Isso a torna mais eficiente que level-wise para o mesmo número de folhas — mas impede o controle por profundidade.

```
Comparar duas estratégias com mesmo número de folhas (K=8):

Level-wise para profundidade 3:
  Nível 0: 1 split (obrigatório)
  Nível 1: 2 splits (obrigatório)
  Nível 2: 4 splits (obrigatório)
  Total: 7 splits, 8 folhas — mas nem todo split é útil

Leaf-wise com num_leaves=8:
  Split 1: maior Gain global
  Split 2: maior Gain entre folhas restantes
  ...
  Split 7: maior Gain entre folhas restantes
  Total: 7 splits, 8 folhas — cada split é o mais útil possível

Resultado:
  Loss após 8 folhas leaf-wise < loss após 8 folhas level-wise
  (provado empiricamente e com bound teórico de melhor case)

Risco de overfitting:
  Árvores assimétricas podem ser muito profundas em um ramo
  → num_leaves é o controle principal (não max_depth)
  → num_leaves=31 (padrão) = profundidade efetiva ~5
  → aumentar num_leaves aumenta capacidade E risco de overfit
```

### 4.5. Suporte Nativo a Features Categóricas — Como a LightGBM evita OneHot?

A LightGBM é um dos poucos algoritmos que trata features categóricas diretamente, sem necessidade de OneHot Encoding — usando um algoritmo de Fisher para encontrar o melhor agrupamento de categorias.

```
Problema: feature categórica com C categorias
  OneHot Encoding: cria C colunas binárias (EFB pode reduzir)
  LightGBM nativo: encontra o melhor subconjunto S ⊆ {1,...,C}
                   tal que Gain é maximizado

Algoritmo de Fisher (split binário ótimo):
  1. Para cada categoria c, calcular: score(c) = G_c / H_c
     onde G_c = Σ_{i: x_i=c} gᵢ  e  H_c = Σ_{i: x_i=c} hᵢ
  2. Ordenar categorias por score(c) de forma crescente
  3. Varrer todos os C-1 cortes nessa ordenação
  4. Escolher o corte com maior Gain

Resultado:
  Split binário: S_esq = {categorias com score < threshold}
                 S_dir = {categorias com score >= threshold}
  → cada nó divide as categorias em dois grupos
  → complexidade O(C log C) ao invés de O(2^C)

Benefício:
  Evita explosão dimensional do OneHot em alta cardinalidade
  Captura interações entre categorias diretamente
```

```python
# LightGBM com features categóricas nativas
import lightgbm as lgb

# Converter para int (LightGBM requer int para categóricas)
cat_cols = ['estado_civil', 'segmento', 'regiao']
for col in cat_cols:
    X_train[col] = X_train[col].astype('category').cat.codes
    X_val[col]   = X_val[col].astype('category').cat.codes

dtrain = lgb.Dataset(
    X_train, label=y_train,
    categorical_feature=cat_cols  # informa quais são categóricas
)
```

### 4.6. Hiperparâmetros críticos e estratégia de tuning

| Hiperparâmetro | O que controla | Efeito de aumentar | Padrão |
| --- | --- | --- | --- |
| num_leaves | Número máximo de folhas (complexidade) | Maior capacidade, mais overfitting | 31 |
| n_estimators | Número de árvores | Mais capacidade (usar early stopping) | 100 |
| learning_rate | Shrinkage (η) | Aprendizado mais lento, mais robusto | 0.1 |
| max_depth | Profundidade máxima (-1 = ilimitada) | Limita overfitting se num_leaves grande | -1 |
| min_child_samples | Mín. amostras por folha | Folhas mais conservadoras | 20 |
| subsample | Fração de linhas por árvore | Mais diversidade, menos overfitting | 1.0 |
| colsample_bytree | Fração de features por árvore | Mais diversidade | 1.0 |
| max_bin | Número de bins do histograma | Mais precisão, mais memória | 255 |
| reg_lambda | L2 sobre pesos das folhas | Pesos menores, mais estabilidade | 0.0 |
| reg_alpha | L1 sobre pesos das folhas | Esparsidade nos pesos | 0.0 |

```python
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

param_dist = {
    'n_estimators':       [500, 1000, 2000],
    'learning_rate':      [0.01, 0.05, 0.1],
    'num_leaves':         [15, 31, 63, 127],
    'max_depth':          [-1, 5, 8, 12],
    'min_child_samples':  [10, 20, 50, 100],
    'subsample':          [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree':   [0.6, 0.7, 0.8, 0.9],
    'reg_lambda':         [0.0, 0.1, 1.0, 5.0],
    'reg_alpha':          [0.0, 0.1, 0.5, 1.0],
    'max_bin':            [127, 255, 511]
}

lgbm = LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    metric='auc',
    n_jobs=-1,
    random_state=42,
    verbose=-1
)

search = RandomizedSearchCV(
    lgbm, param_dist,
    n_iter=60,
    scoring='roc_auc',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
search.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)

print(f"Melhores hiperparâmetros: {search.best_params_}")
print(f"Melhor AUC CV:  {search.best_score_:.4f}")
print(f"AUC Validação:  {roc_auc_score(y_val, search.best_estimator_.predict_proba(X_val)[:,1]):.4f}")
```

### 4.7. LightGBM vs XGBoost — Quando usar cada um?

| Critério | LightGBM | XGBoost |
| --- | --- | --- |
| Velocidade de treino | Muito rápida (histogramas + GOSS + EFB) | Mais lenta (sort-based exato) |
| Uso de memória | Baixo (histogramas binários) | Alto (sort de valores contínuos) |
| Performance em datasets pequenos | Risco de overfit mais alto | Mais estável |
| Performance em datasets grandes | Superior ou igual | Inferior em velocidade |
| Features categóricas | Suporte nativo (sem OneHot) | Requer encoding manual |
| Valores ausentes | Tratamento nativo | Tratamento nativo |
| Hiperparâmetro chave | num_leaves | max_depth |
| Paralelização | Data + feature parallel | Feature parallel |

**Recomendação prática no CRISP:**

1. **Datasets grandes (N > 100k ou M > 100):** prefira LightGBM — muito mais rápido sem perda de qualidade
2. **Datasets pequenos (N < 10k):** XGBoost tende a ser mais estável — LightGBM pode overfit com num_leaves alto
3. **Features categóricas de alta cardinalidade:** LightGBM ganha com suporte nativo
4. **Competições Kaggle e produção:** compare ambos com cross-validation — a diferença de AUC é geralmente pequena, a diferença de velocidade pode ser grande