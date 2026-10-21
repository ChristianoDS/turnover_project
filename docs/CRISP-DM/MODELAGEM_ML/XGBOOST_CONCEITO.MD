## Fase 1: O Treinamento

A XGBoost (eXtreme Gradient Boosting) é um algoritmo de **ensemble sequencial** baseado em Gradient Boosting. Diferente da Random Forest — que constrói árvores em paralelo e independentes — a XGBoost constrói árvores **em sequência**, onde cada nova árvore aprende a corrigir os erros residuais cometidos por todas as árvores anteriores.

A palavra "eXtreme" não é exagero: a XGBoost é uma implementação altamente otimizada do Gradient Boosting, com regularização embutida, tratamento nativo de valores ausentes, suporte a paralelização e uso eficiente de memória — tornando-a consistentemente um dos algoritmos mais competitivos em dados tabulares.

### O que entra no modelo?

- **Entradas (X):** variáveis preditoras numéricas e/ou categóricas (encodadas). XGBoost é invariante à escala — **não precisa de normalização**
- **Saída esperada (y):** variável resposta — binária ou multiclasse (classificação) ou contínua (regressão)
- **Hiperparâmetros principais:** n_estimators, learning_rate (eta), max_depth, subsample, colsample_bytree, reg_alpha (L1), reg_lambda (L2), min_child_weight

### O que o modelo aprende?

A XGBoost aprende um **ensemble aditivo de árvores de decisão** (chamadas de CART — Classification and Regression Trees). Cada árvore aprende a prever os **gradientes negativos** (pseudo-resíduos) da função de perda em relação às previsões acumuladas até aquele ponto. O modelo final é a soma ponderada de todas as árvores treinadas sequencialmente.

### O que o modelo produz?

- **Classificação binária:** score bruto transformado em probabilidade via função Sigmoid: P(y=1 | X)
- **Classificação multiclasse:** scores por classe via Softmax
- **Regressão:** valor contínuo predito diretamente
- **Feature Importance:** gain, cover, weight ou SHAP — múltiplas métricas de importância

### Funcionamento geral

O treinamento começa com uma previsão inicial constante (média do target para regressão, log-odds para classificação). Em cada iteração, calcula-se o erro do modelo atual, treina-se uma nova árvore para corrigir esse erro, e adiciona-se a árvore ao ensemble com peso controlado pela learning rate. Esse processo se repete T vezes — uma árvore por iteração.

## 2. Fase 2: Processo de Aprendizado

O aprendizado da XGBoost acontece em uma sequência de 6 etapas que se repetem a cada iteração (uma por árvore). A lógica é sempre a mesma: medir onde o modelo atual está erando e construir uma nova árvore para corrigir precisamente esses erros.

### 2.1. Etapa 1 — Previsão Inicial (Base Score)

**O que é:** antes de treinar qualquer árvore, o modelo define uma previsão inicial constante para todas as observações. É o ponto de partida do ensemble — o "chute inicial" antes de qualquer aprendizado.

**Entrada:** variável resposta y do conjunto de treino

**Saída:** valor constante ŷ₀ aplicado a todas as observações

**Próxima etapa:** com a previsão inicial definida, calcula-se o erro entre ŷ₀ e os valores reais y

### 2.2. Etapa 2 — Cálculo dos Gradientes e Hessianos

**O que é:** o modelo calcula, para cada observação, duas quantidades derivadas da função de perda: o **gradiente (g)** — primeira derivada da loss em relação à previsão atual — e o **hessiano (h)** — segunda derivada. Juntos, eles descrevem a curvatura do erro de cada observação.

**Entrada:** previsões acumuladas até a iteração atual ŷ e rótulos reais y

**Saída:** vetores g = (g₁, g₂, ..., gₙ) e h = (h₁, h₂, ..., hₙ) — um par (g, h) por observação

**Próxima etapa:** esses gradientes são o target que a nova árvore vai aprender a prever

### 2.3. Etapa 3 — Construção da Nova Árvore sobre os Gradientes

**O que é:** uma nova árvore CART é construída usando g e h como sinal de aprendizado. A árvore não aprende a prever y diretamente — aprende a prever os gradientes, ou seja, a direção e magnitude do erro atual do ensemble.

**Entrada:** gradientes g e hessianos h de todas as observações + features X

**Saída:** estrutura da árvore — conjunto de splits (feature, threshold) e valores nas folhas

**Próxima etapa:** os valores ótimos das folhas são calculados com base em g e h

### 2.4. Etapa 4 — Cálculo dos Valores Ótimos das Folhas

**O que é:** diferente das árvores da Random Forest (que usam a média das amostras na folha), a XGBoost calcula o valor ótimo de cada folha analiticamente, minimizando a função objetivo regularizada.

**Entrada:** gradientes g e hessianos h das amostras que caem em cada folha + parâmetro de regularização λ

**Saída:** valor ótimo w* para cada folha da árvore

**Próxima etapa:** a árvore completa (estrutura + valores de folha) é adicionada ao ensemble

### 2.5. Etapa 5 — Atualização do Ensemble

**O que é:** a nova árvore é incorporada ao ensemble com peso controlado pela learning rate η (eta). As previsões acumuladas são atualizadas somando a contribuição da nova árvore multiplicada por η.

**Entrada:** previsões acumuladas ŷ_{t-1} + nova árvore f_t + learning rate η

**Saída:** previsões atualizadas ŷ_t = ŷ_{t-1} + η · f_t(X)

**Próxima etapa:** o ciclo reinicia na Etapa 2 com as previsões atualizadas — até treinar T árvores

### 2.6. Etapa 6 — Seleção do Melhor Split (Critério de Ganho)

**O que é:** dentro da construção de cada árvore (Etapa 3), o algoritmo avalia cada possível split usando um critério de ganho baseado em g e h. O split escolhido é o que maximiza o ganho líquido após penalidade de regularização.

**Entrada:** gradientes e hessianos do nó atual + todos os possíveis pares (feature, threshold)

**Saída:** melhor par (feature*, threshold*) para dividir o nó

**Próxima etapa:** divisão do nó em dois filhos; recursão continua ou nó vira folha

## 3. Fase 3: Detalhamento das Etapas

Agora veremos as fórmulas concretas de cada etapa e como elas transformam a entrada na saída.

### 3.1. Etapa 1 — Base Score: Fórmula e Execução

**Fórmula:**

```
Classificação binária:
  ŷ₀ = log( ȳ / (1 - ȳ) )   → log-odds da taxa de evento
  onde ȳ = proporção de positivos no treino

Regressão:
  ŷ₀ = ȳ   → média do target

Efeito: inicializa as previsões de todas as N observações
        com o mesmo valor constante antes da 1ª árvore
```

**Como é executado:**

```python
import numpy as np

# Classificação binária — base score manual
y_mean = y_train.mean()
base_score = np.log(y_mean / (1 - y_mean))  # log-odds
print(f"Taxa de evento: {y_mean:.4f}")
print(f"Base score (log-odds): {base_score:.4f}")

# No XGBoost sklearn API — base_score é ajustado automaticamente
import xgboost as xgb
model = xgb.XGBClassifier(base_score=0.5, random_state=42)
# base_score=0.5 equivale a log-odds=0.0 (50% de probabilidade inicial)
```

### 3.2. Etapa 2 — Gradientes e Hessianos: Fórmula e Execução

**Fórmulas por função de perda:**

```
Classificação Binária (Binary Cross-Entropy):
  Loss = -[ y·log(p) + (1-y)·log(1-p) ]   onde p = sigmoid(ŷ)

  Gradiente:  gᵢ = pᵢ - yᵢ
  Hessiano:   hᵢ = pᵢ · (1 - pᵢ)

Regressão (MSE):
  Loss = (ŷ - y)²

  Gradiente:  gᵢ = 2·(ŷᵢ - yᵢ)   → resíduo escalado
  Hessiano:   hᵢ = 2              → constante (MSE é quadrático)

Interpretação:
  gᵢ → magnitude e direção do erro para a observação i
  hᵢ → curvatura da loss — o quanto confiar no gradiente
       hᵢ alto → gradiente confiável, ajuste maior
       hᵢ baixo → gradiente incerto, ajuste mais conservador
```

**Como é executado:**

```python
def compute_gradients_hessians(y_true, y_pred_logit):
    """Para classificação binária com BCE"""
    p = 1 / (1 + np.exp(-y_pred_logit))  # sigmoid
    g = p - y_true                        # gradiente
    h = p * (1 - p)                       # hessiano
    return g, h

# Exemplo
y_pred_logit = np.full(len(y_train), base_score)  # iteração 0
g, h = compute_gradients_hessians(y_train.values, y_pred_logit)
print(f"Gradiente médio: {g.mean():.4f}")
print(f"Hessiano médio:  {h.mean():.4f}")
```

### 3.3. Etapa 3/6 — Critério de Ganho do Split: Fórmula e Execução

**Fórmula do Gain:**

```
Gain(split) = ½ · [ G_L²/(H_L + λ) + G_R²/(H_R + λ) - (G_L+G_R)²/(H_L+H_R + λ) ] - γ

onde:
  G_L = Σ gᵢ  para amostras no filho esquerdo
  G_R = Σ gᵢ  para amostras no filho direito
  H_L = Σ hᵢ  para amostras no filho esquerdo
  H_R = Σ hᵢ  para amostras no filho direito
  λ   = reg_lambda (regularização L2 — suaviza o denominador)
  γ   = min_split_loss (penalidade mínima para criar um split)

Interpretação:
  Gain > 0 → o split melhora o modelo líquido da regularização
  Gain ≤ 0 → o split não justifica a complexidade — nó vira folha
  γ controla a "poda" — splits com ganho < γ são descartados
```

**Como é executado:**

```python
def compute_gain(g_left, h_left, g_right, h_right, lam=1.0, gamma=0.0):
    score_left  = g_left**2  / (h_left  + lam)
    score_right = g_right**2 / (h_right + lam)
    score_total = (g_left + g_right)**2 / (h_left + h_right + lam)
    gain = 0.5 * (score_left + score_right - score_total) - gamma
    return gain

def best_split_xgb(X_node, g, h, lam=1.0, gamma=0.0):
    best_gain, best_feat, best_thresh = -np.inf, None, None
    n_features = X_node.shape[1]
    G_total, H_total = g.sum(), h.sum()

    for feat in range(n_features):
        sorted_idx = np.argsort(X_node[:, feat])
        G_left, H_left = 0.0, 0.0

        for i in sorted_idx[:-1]:   # varre todos os thresholds
            G_left += g[i]; H_left += h[i]
            G_right = G_total - G_left
            H_right = H_total - H_left
            gain = compute_gain(G_left, H_left, G_right, H_right, lam, gamma)
            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = (X_node[sorted_idx[i], feat] +
                               X_node[sorted_idx[i+1], feat]) / 2
    return best_feat, best_thresh, best_gain
```

### 3.4. Etapa 4 — Valor Ótimo das Folhas: Fórmula e Execução

**Fórmula:**

```
w* = - (Σ gᵢ) / (Σ hᵢ + λ)

onde:
  Σ gᵢ = soma dos gradientes das amostras na folha
  Σ hᵢ = soma dos hessianos das amostras na folha
  λ    = reg_lambda (regularização L2)

Derivação: minimiza a função objetivo quadrática aproximada
da loss em relação ao valor w da folha — resultado analítico
exato, sem necessidade de iteração adicional.

Efeito de λ:
  λ → 0: w* ≈ -G/H  (sem regularização)
  λ → ∞: w* → 0    (regularização forte — folhas próximas de zero)
```

**Como é executado:**

```python
def leaf_weight(g_leaf, h_leaf, lam=1.0):
    return -g_leaf.sum() / (h_leaf.sum() + lam)

# Exemplo: folha com 5 amostras
g_leaf = np.array([0.3, -0.1, 0.4, 0.2, -0.05])
h_leaf = np.array([0.2,  0.15, 0.18, 0.22, 0.1])

w_star = leaf_weight(g_leaf, h_leaf, lam=1.0)
print(f"Valor ótimo da folha (w*): {w_star:.4f}")
# Sem regularização: {-g_leaf.sum()/h_leaf.sum():.4f}
```

### 3.5. Etapa 5 — Atualização do Ensemble: Fórmula e Execução

**Fórmula:**

```
ŷ_t(xᵢ) = ŷ_{t-1}(xᵢ) + η · f_t(xᵢ)

onde:
  ŷ_{t-1} = previsão acumulada até a árvore t-1
  η (eta)  = learning_rate — fator de encolhimento (shrinkage)
  f_t(xᵢ) = valor da folha em que xᵢ cai na árvore t

Efeito do learning rate:
  η = 1.0 → sem shrinkage — cada árvore tem peso total
  η = 0.1 → shrinkage moderado — cada árvore contribui 10%
  η → 0   → passos minúsculos — precisa de mais árvores

Relação η × n_estimators:
  η menor → precisa de mais árvores para convergir
  Regra prática: η=0.01–0.1, n_estimators=500–5000
  com early stopping para encontrar T ótimo
```

**Como é executado:**

```python
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# XGBoost com Early Stopping — encontra T ótimo automaticamente
model = xgb.XGBClassifier(
    n_estimators=2000,       # máximo de árvores
    learning_rate=0.05,      # η
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,          # L2
    reg_alpha=0.0,           # L1
    min_child_weight=1,
    use_label_encoder=False,
    eval_metric='auc',
    early_stopping_rounds=50,  # para se AUC não melhora em 50 rodadas
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

print(f"Melhor iteração: {model.best_iteration}")
print(f"AUC Validação:   {roc_auc_score(y_val, model.predict_proba(X_val)[:,1]):.4f}")
```

## 4. Fase 4: Aprofundando a Matemática

Nesta fase exploramos cada ferramenta matemática em profundidade — o que é, por que existe e como contribui para a força do algoritmo.

### 4.1. Boosting vs Bagging — A diferença fundamental de filosofia

Embora ambos sejam métodos de ensemble, Boosting e Bagging têm filosofias completamente opostas para lidar com o erro.

**Bagging (Random Forest):**

- Treina modelos **em paralelo**, cada um em um subset aleatório dos dados
- Reduz **variância** — média de modelos independentes é mais estável
- Cada modelo ignora os erros dos outros — não há coordenação
- Funciona porque os erros se cancelam na agregação

**Boosting (XGBoost):**

- Treina modelos **em sequência** — cada um foca nos erros do anterior
- Reduz **bias** — o ensemble aprende progressivamente padrões difíceis
- Cada modelo é guiado pelos gradientes do modelo anterior
- Funciona porque o erro residual diminui a cada iteração

```
Decomposição do Erro de Generalização:
  Erro = Bias² + Variância + Ruído irredutível

  Bagging:  mantém bias, reduz variância
  Boosting: reduz bias, pode aumentar variância (risco de overfitting)

Por isso:
  XGBoost precisa de regularização (λ, α, subsample, early stopping)
  Random Forest é naturalmente mais resistente ao overfitting
```

### 4.2. Gradiente e Hessiano — Por que usar as duas derivadas?

O Gradient Boosting clássico (Friedman, 1999) usa apenas a primeira derivada (gradiente) para guiar o aprendizado. A XGBoost (Chen & Guestrin, 2016) usa também a **segunda derivada (hessiano)** — essa é uma das inovações centrais do algoritmo.

**O que o gradiente diz:** aponta a direção em que a loss aumenta. Seu negativo é a direção de descida — o que a árvore deve aprender a prever.

**O que o hessiano adiciona:** mede a curvatura da loss. Uma curvatura alta significa que a loss muda rapidamente nessa região — um ajuste conservador é mais seguro. Uma curvatura baixa permite um ajuste mais ousado.

```
Analogo à otimização de Newton-Raphson:
  Gradient Descent:  w_novo = w - η · g
  Newton-Raphson:    w_novo = w - g/h

  Newton usa h para escalar o gradiente:
  → passos menores onde a curvatura é alta (incerto)
  → passos maiores onde a curvatura é baixa (confiante)

A XGBoost faz exatamente isso no valor ótimo da folha:
  w* = -G/H  (análogo a Newton, com regularização λ)

Benefício prático:
  Convergência mais rápida — precisa de menos árvores
  para atingir a mesma qualidade que Gradient Boosting puro
```

### 4.3. Função Objetivo Regularizada — A inovação teórica da XGBoost

A XGBoost minimiza explicitamente uma função objetivo que combina a loss dos dados com a complexidade do modelo — isso a distingue do Gradient Boosting tradicional.

**Função objetivo completa:**

```
Obj = Σᵢ L(yᵢ, ŷᵢ) + Σ_t Ω(f_t)

onde Ω(f) é a penalidade de complexidade da árvore f:
  Ω(f) = γ·T + (λ/2)·Σⱼ wⱼ²

  T  = número de folhas da árvore
  wⱼ = valor da folha j
  γ  = penalidade por folha extra (min_split_loss)
  λ  = regularização L2 sobre os pesos das folhas (reg_lambda)

Efeito de cada componente:
  γ  → controla o número de folhas — árvores mais rasas
  λ  → encolhe os valores das folhas — previsões mais conservadoras
  α  → regularização L1: Ω += α·Σ|wⱼ| → zera folhas irrelevantes

Approximação de 2ª ordem (Taylor):
  L(yᵢ, ŷᵢ + f_t(xᵢ)) ≈ L(yᵢ, ŷᵢ) + gᵢ·f_t(xᵢ) + ½·hᵢ·f_t(xᵢ)²

  Esta aproximação permite minimizar analiticamente Obj
  → resultado: w* = -G/(H + λ)  e  Score do nó = -G²/(2(H+λ))
```

### 4.4. Shrinkage (Learning Rate) — Por que aprender devagar funciona melhor?

O shrinkage (η < 1) é um dos regularizadores mais importantes da XGBoost. Multiplicar cada árvore por η antes de adicioná-la ao ensemble "encolhe" sua contribuição, deixando espaço para que as próximas árvores refinem a solução.

**Intuição:** imagine construir uma escultura. Cada árvore é um golpe de cinzel. Com η=1, cada golpe é na força máxima — fácil errar. Com η=0.1, cada golpe é sutil — mais controle, mais precisão, mas precisa de mais golpes.

```
Efeito estatístico do Shrinkage:
  Cada árvore é multiplicada por η antes de ser somada.
  O ensemble após T árvores é:
  ŷ_T = ŷ₀ + η·f₁ + η·f₂ + ... + η·f_T

Por que funciona:
  1. Evita que uma única árvore domine o ensemble
  2. Aumenta a diversidade — cada árvore vê resíduos
     que ainda contêm estrutura para aprender
  3. Reduz overfitting ao custo de mais iterações

Regra prática:
  η ∈ [0.01, 0.1]  →  n_estimators ∈ [500, 5000]
  Use early stopping para encontrar T ótimo automaticamente
  Nunca ajuste η e n_estimators manualmente juntos
```

### 4.5. Subsampling — Como a aleatoriedade melhora o Boosting

A XGBoost herda da Stochastic Gradient Boosting (Friedman, 2002) o uso de subsampling para injetar aleatoriedade — reduzindo a correlação entre as árvores e funcionando como regularizador implícito.

**Dois níveis de subsampling:**

```
1. subsample (row sampling):
   Antes de construir cada árvore, sorteia uma fração
   das observações sem reposição
   → subsample=0.8: usa 80% das linhas por árvore
   → reduz overfitting, aumenta velocidade
   → análogo ao Bootstrap da RF, mas sem reposição

2. colsample_bytree (column sampling):
   Antes de construir cada árvore, sorteia uma fração
   das features disponíveis
   → colsample_bytree=0.8: usa 80% das colunas por árvore
   → análogo ao max_features da RF
   → reduz correlação entre árvores consecutivas

Outros níveis de column sampling:
   colsample_bylevel: re-sorteia features a cada nível da árvore
   colsample_bynode:  re-sorteia features a cada nó (mais agressivo)
```

### 4.6. Hiperparâmetros críticos e estratégia de tuning

| Hiperparâmetro | O que controla | Efeito de aumentar | Padrão |
| --- | --- | --- | --- |
| n_estimators | Número de árvores | Mais capacidade (risco overfitting) | 100 |
| learning_rate | Shrinkage (η) | Aprendizado mais devagar, mais robusto | 0.3 |
| max_depth | Profundidade das árvores | Maior capacidade, mais overfitting | 6 |
| min_child_weight | Σhᵢ mínimo por folha | Folhas mais conservadoras | 1 |
| subsample | Fração de linhas por árvore | Mais diversidade entre árvores | 1.0 |
| colsample_bytree | Fração de features por árvore | Mais diversidade, menos overfitting | 1.0 |
| reg_lambda | Regularização L2 dos pesos | Pesos de folha menores | 1.0 |
| reg_alpha | Regularização L1 dos pesos | Esparsidade nos pesos | 0.0 |
| gamma | Ganho mínimo para split | Árvores mais rasas e simples | 0.0 |

```python
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

# Grid de busca balanceado
param_dist = {
    'n_estimators':      [500, 1000, 2000],
    'learning_rate':     [0.01, 0.05, 0.1],
    'max_depth':         [3, 4, 5, 6],
    'min_child_weight':  [1, 3, 5, 10],
    'subsample':         [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree':  [0.6, 0.7, 0.8, 0.9],
    'reg_lambda':        [0.1, 1.0, 5.0, 10.0],
    'reg_alpha':         [0.0, 0.1, 0.5, 1.0],
    'gamma':             [0.0, 0.1, 0.5, 1.0]
}

xgb_model = xgb.XGBClassifier(
    eval_metric='auc',
    use_label_encoder=False,
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)

search = RandomizedSearchCV(
    xgb_model, param_dist,
    n_iter=60,
    scoring='roc_auc',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# fit_params para passar eval_set ao early stopping
search.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"Melhores hiperparâmetros: {search.best_params_}")
print(f"Melhor AUC CV:  {search.best_score_:.4f}")
print(f"AUC Validação:  {roc_auc_score(y_val, search.best_estimator_.predict_proba(X_val)[:,1]):.4f}")
```

### 4.7. XGBoost vs Random Forest — Quando usar cada um?

| Critério | XGBoost | Random Forest |
| --- | --- | --- |
| Paradigma | Boosting sequencial (reduz bias) | Bagging paralelo (reduz variância) |
| Performance típica | Superior em dados tabulares competitivos | Excelente baseline, mais robusto |
| Overfitting | Maior risco — precisa de regularização | Menor risco naturalmente |
| Velocidade de treino | Lento (sequencial) | Rápido (paralelo) |
| Tuning necessário | Intenso — muitos hiperparâmetros | Moderado — menos sensível |
| Dados desbalanceados | scale_pos_weight para ajuste | class_weight='balanced' |
| Valores ausentes | Tratamento nativo embutido | Requer imputação prévia |
| Early stopping | Sim — essencial para o tuning | Não aplicável |

**Recomendação prática no CRISP:**

1. Use Random Forest como **baseline rápido** — fácil de treinar, difícil de errar feio
2. Use XGBoost para **maximizar performance** — quando o baseline não é suficiente
3. Compare ambos com cross-validation antes de escolher o modelo final