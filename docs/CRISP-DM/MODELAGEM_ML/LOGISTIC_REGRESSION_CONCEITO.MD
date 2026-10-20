
## 1. Fase 1: O Treinamento

A Regressão Logística é um algoritmo de **classificação supervisionada** — apesar do nome conter "Regressão". Ela é usada para prever a probabilidade de um evento binário ocorrer: churn ou não churn, fraude ou não fraude, inadimplência ou adimplência.

O algoritmo aprende uma fronteira de decisão no espaço das features que separa as duas classes. Diferente de uma linha reta (como na regressão linear), essa fronteira é aprendida através de uma função que transforma qualquer valor numérico em uma probabilidade entre 0 e 1.

### O que entra no modelo?

- **Entradas (X):** variáveis preditoras numéricas, já normalizadas e encodadas
- **Saída esperada (y):** variável binária — 0 (não-evento) ou 1 (evento)

### O que o modelo aprende?

O modelo aprende um conjunto de **pesos (coeficientes)** — um para cada feature — que indicam o quanto cada variável aumenta ou reduz a probabilidade do evento. Também aprende um **bias (intercepto)**, que é o ponto de partida da probabilidade quando todas as features são zero.

### O que o modelo produz?

- Uma **probabilidade** entre 0 e 1 para cada observação: $P(y=1\| X)$
	- Uma **classe predita** (0 ou 1) com base em um limiar de corte (default = 0.5)
    
    ### Funcionamento geral
    
    O treinamento consiste em encontrar os valores de pesos que minimizam o erro entre as probabilidades previstas e os rótulos reais. Esse processo é iterativo: o modelo faz uma previsão, calcula o erro, e ajusta os pesos na direção que reduz esse erro. Isso se repete até a convergência.
    

## 2. Fase 2: Processo de Aprendizado

O aprendizado da Regressão Logística acontece em uma sequência de 5 etapas que se repetem a cada iteração do treinamento. Entender essa sequência é entender como qualquer modelo linear aprende.

### 2.1. Etapa 1 — Combinação Linear das Features

**O que é:** o modelo começa calculando uma combinação linear de todas as features de entrada — cada feature é multiplicada pelo seu peso correspondente e os resultados são somados junto com o bias.

**Entrada:** vetor de features X e vetor de pesos W (inicializados aleatoriamente)

**Saída:** um valor numérico z, chamado de log-odds ou logit — pode ser qualquer número real, de −∞ a +∞

**Próxima etapa:** esse valor z não é ainda uma probabilidade. Precisa ser transformado.

### 2.2. Etapa 2 — Função de Ativação Sigmoid

**O que é:** a função Sigmoid recebe o valor z da etapa anterior e o transforma em uma probabilidade entre 0 e 1. Ela "comprime" qualquer número real para o intervalo (0, 1).

**Entrada:** valor z (log-odds)

**Saída:** probabilidade ŷ = P(y=1 | X), um número entre 0 e 1

**Próxima etapa:** com a probabilidade estimada, o modelo pode calcular o erro em relação ao rótulo real.

### 2.3. Etapa 3 — Cálculo do Erro com Binary Cross-Entropy - Para converter a função sigmoid em concava (onde há minimos)

**O que é:** a função de perda mede o quão errada está a previsão do modelo. Para classificação binária, usamos a Binary Cross-Entropy (Log Loss), que penaliza fortemente previsões confiantes e erradas.

**Entrada:** probabilidade prevista ŷ e rótulo real y (0 ou 1)

**Saída:** um escalar L (loss) que indica o tamanho do erro — quanto maior, pior

**Próxima etapa:** com o erro calculado, precisamos saber em qual direção ajustar os pesos para reduzi-lo.

### 2.4. Etapa 4 — Cálculo do Gradiente

**O que é:** o gradiente é a derivada da função de perda em relação a cada peso. Ele indica a direção e a intensidade do ajuste necessário em cada peso para reduzir o erro.

**Entrada:** loss L, pesos W atuais e features X

**Saída:** vetor de gradientes ∂L/∂W — um valor por peso, indicando "quanto e para onde mover"

**Próxima etapa:** com os gradientes em mãos, os pesos são atualizados.

### 2.5. Etapa 5 — Atualização dos Pesos com Gradient Descent

**O que é:** o Gradient Descent usa os gradientes calculados para atualizar os pesos na direção que reduz a loss. O tamanho do passo é controlado pela **taxa de aprendizado (learning rate)**.

**Entrada:** pesos W atuais, gradientes ∂L/∂W e learning rate η

**Saída:** pesos W atualizados

**Próxima etapa:** o ciclo recomeça com os novos pesos — volta para a Etapa 1. Repete até convergir.

## 3. Fase 3: Detalhamento das Etapas

Agora veremos as fórmulas concretas de cada etapa e como elas transformam a entrada na saída.

### 3.1. Etapa 1 — Combinação Linear: Fórmula e Execução

**Fórmula:**

```
z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ

Ou em notação vetorial:
z = W · X + b

onde:
  X  = [x₁, x₂, ..., xₙ]  → vetor de features da observação
  W  = [w₁, w₂, ..., wₙ]  → vetor de pesos (aprendidos)
  b  = w₀                  → bias (intercepto)
  z  = log-odds            → saída escalar
```

**Como é executado:**

1. Para cada observação, pega-se o vetor X com os valores das n features
2. Cada xᵢ é multiplicado pelo peso wᵢ correspondente
3. Todos os produtos são somados junto com o bias b
4. O resultado z é um único número — positivo indica tendência à classe 1, negativo à classe 0

```python
import numpy as np

# Exemplo com 1 observação e 3 features
X = np.array([0.8, -0.3, 1.2])   # features normalizadas
W = np.array([0.5, -1.2, 0.9])   # pesos (inicialmente aleatórios)
b = 0.1                           # bias

z = np.dot(W, X) + b
print(f"z (log-odds) = {z:.4f}")
```

### 3.2. Etapa 2 — Função Sigmoid: Fórmula e Execução

**Fórmula:**

```
σ(z) = 1 / (1 + e^(-z))

onde:
  z    = log-odds (saída da etapa 1)
  e    = número de Euler ≈ 2.718
  σ(z) = probabilidade prevista ŷ ∈ (0, 1)

Propriedades:
  σ(0)   = 0.5   → incerteza máxima
  σ(+∞) → 1.0   → certeza na classe 1
  σ(-∞) → 0.0   → certeza na classe 0
```

**Como é executado:**

1. Recebe o valor z da combinação linear
2. Aplica a fórmula: divide 1 por (1 + e^(-z))
3. Para z=0 → probabilidade=0.5; para z=5 → probabilidade≈0.993; para z=-5 → probabilidade≈0.007
4. A saída ŷ é a probabilidade estimada de y=1

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

y_hat = sigmoid(z)
print(f"Probabilidade prevista ŷ = {y_hat:.4f}")

# Decisão com threshold padrão
y_pred = 1 if y_hat >= 0.5 else 0
print(f"Classe predita = {y_pred}")
```

### 3.3. Etapa 3 — Binary Cross-Entropy: Fórmula e Execução

**Fórmula (para uma observação):**

```
L(y, ŷ) = -[ y · log(ŷ) + (1-y) · log(1-ŷ) ]

Para N observações (loss média):
L = -(1/N) · Σᵢ [ yᵢ · log(ŷᵢ) + (1-yᵢ) · log(1-ŷᵢ) ]

onde:
  y  = rótulo real (0 ou 1)
  ŷ  = probabilidade prevista pelo modelo
  log = logaritmo natural
```

**Como é executado:**

1. Se y=1: a loss é -log(ŷ). Quanto mais próximo ŷ de 1, menor a loss
2. Se y=0: a loss é -log(1-ŷ). Quanto mais próximo ŷ de 0, menor a loss
3. A loss explode para +∞ quando o modelo prevê com certeza e erra (ex: ŷ→0 mas y=1)
4. A loss total é a média sobre todas as N observações do batch

```python
def binary_cross_entropy(y_true, y_hat):
    epsilon = 1e-15  # evita log(0)
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_hat) + (1 - y_true) * np.log(1 - y_hat))

y_true = np.array([1, 0, 1, 1, 0])
y_hat  = np.array([0.9, 0.1, 0.8, 0.4, 0.3])

loss = binary_cross_entropy(y_true, y_hat)
print(f"Binary Cross-Entropy Loss = {loss:.4f}")
```

### 3.4. Etapa 4 — Gradiente da Loss: Fórmula e Execução

**Fórmula do gradiente (derivada da BCE em relação aos pesos):**

```
∂L/∂W = (1/N) · Xᵀ · (ŷ - y)
∂L/∂b = (1/N) · Σ (ŷᵢ - yᵢ)

onde:
  ŷ - y  = vetor de erros (resíduos) por observação
  Xᵀ     = transposta da matriz de features
  N      = número de observações

Propriedade elegante: o gradiente da BCE com Sigmoid
se simplifica para (ŷ - y) · X — sem precisar de
cálculo encadeado complexo (chain rule se cancela).
```

**Como é executado:**

1. Calcula o vetor de erros: residuals = ŷ - y
2. Multiplica cada resíduo pelo vetor de features correspondente
3. Faz a média sobre todas as observações
4. O resultado é um vetor com o mesmo tamanho de W — um gradiente por peso

```python
def compute_gradients(X, y_true, y_hat):
    N = len(y_true)
    residuals = y_hat - y_true          # (ŷ - y)
    dW = (1/N) * X.T @ residuals        # gradiente dos pesos
    db = (1/N) * np.sum(residuals)      # gradiente do bias
    return dW, db
```

### 3.5. Etapa 5 — Gradient Descent: Fórmula e Execução

**Fórmula de atualização dos pesos:**

```
W_novo = W_atual - η · ∂L/∂W
b_novo = b_atual - η · ∂L/∂b

onde:
  η (eta)  = learning rate (taxa de aprendizado)
             controla o tamanho do passo de atualização
  ∂L/∂W   = gradiente dos pesos (direção do aumento da loss)
  O sinal negativo → move os pesos na direção oposta
             ao gradiente (descida do gradiente)
```

**Como é executado:**

1. Multiplica cada gradiente pela learning rate η
2. Subtrai o resultado dos pesos atuais
3. Os novos pesos substituem os antigos
4. O ciclo completo (etapas 1→5) se repete por N épocas

```python
def update_weights(W, b, dW, db, learning_rate=0.01):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

# Loop de treinamento completo
def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    n_features = X.shape[1]
    W = np.zeros(n_features)   # inicialização
    b = 0.0
    losses = []

    for epoch in range(epochs):
        # Etapa 1: Combinação linear
        z = X @ W + b
        # Etapa 2: Sigmoid
        y_hat = sigmoid(z)
        # Etapa 3: Loss
        loss = binary_cross_entropy(y, y_hat)
        losses.append(loss)
        # Etapa 4: Gradientes
        dW, db = compute_gradients(X, y, y_hat)
        # Etapa 5: Atualização
        W, b = update_weights(W, b, dW, db, lr)

        if epoch % 100 == 0:
            print(f"Época {epoch:4d} | Loss: {loss:.4f}")

    return W, b, losses
```

## 4. Fase 4: Aprofundando a Matemática

Nesta fase exploramos cada ferramenta matemática em profundidade — o que ela é, por que existe e como contribui para o aprendizado.

### 4.1. A Combinação Linear — Por que multiplicar features por pesos?

A combinação linear é a operação fundamental de qualquer modelo linear. Ela representa uma **hipótese sobre o mundo**: que o efeito de cada feature sobre o resultado é proporcional ao seu valor e pode ser somado ao efeito das demais.

**Para que serve:** transforma um vetor multidimensional (todas as features) em um único número escalar que resume a "evidência" acumulada para classificar a observação.

**O papel dos pesos (W):** cada peso wᵢ representa a contribuição marginal da feature xᵢ. Um peso positivo grande → a feature aumenta a probabilidade do evento. Um peso negativo grande → a feature reduz a probabilidade.

**O papel do bias (b):** o bias é o "ponto de partida" do modelo — a log-odds quando todas as features são zero. Sem o bias, o modelo seria forçado a passar pela origem, limitando sua expressividade.

**Limitação:** a combinação linear só captura relações lineares entre features e target. Relações não-lineares (ex: U-shaped) não são capturadas sem engenharia de features.

### 4.2. A Função Sigmoid — Por que não usar a saída linear diretamente?

A combinação linear z pode assumir qualquer valor de −∞ a +∞. Probabilidades, por definição, existem apenas entre 0 e 1. A Sigmoid é a "ponte" entre esses dois mundos.

**O que ela faz matematicamente:** a Sigmoid é uma função S (sigmoide) que mapeia qualquer número real para (0, 1). Ela é monótona crescente — valores maiores de z sempre produzem probabilidades maiores.

**Por que ela é ideal para classificação binária:**

- Derivada simples: σ'(z) = σ(z) · (1 - σ(z)) — facilita o cálculo dos gradientes
- Interpretação probabilística direta: a saída é literalmente P(y=1 | X)
- Saturação nos extremos: para valores muito positivos ou negativos de z, a Sigmoid "satura" em 1 ou 0 — o modelo expressa certeza

**A relação com log-odds:**

```
Se ŷ = σ(z), então:
  z = log( ŷ / (1-ŷ) )  →  log-odds (logit)

Isso significa que z representa o logaritmo da
razão de probabilidades (odds ratio) — uma medida
natural de "quanto mais provável é y=1 do que y=0".
```

### 4.3. Binary Cross-Entropy — Por que não usar o MSE?

A escolha da função de perda não é arbitrária. O MSE (Erro Quadrático Médio), padrão em regressão, é tecnicamente aplicável à classificação mas produz uma superfície de loss não-convexa com Sigmoid — dificultando a convergência do Gradient Descent.

**Por que a Cross-Entropy é melhor para classificação:**

1. **Convexidade:** a BCE com Sigmoid produz uma superfície convexa — existe um único mínimo global, garantindo convergência
2. **Penalidade assimétrica:** erros confiantes são penalizados muito mais que erros hesitantes. Se o modelo prevê ŷ=0.99 mas y=0, a loss é enorme. Isso força o modelo a ser cauteloso.
3. **Derivação do máximo de verossimilhança:** a BCE é equivalente a maximizar a verossimilhança (log-likelihood) dos dados sob o modelo — tem fundamento estatístico sólido

**Interpretação intuitiva da BCE:**

```
Quando y=1:
  Quer ŷ próximo de 1
  Loss = -log(ŷ): se ŷ=0.9  → loss=0.10
                  se ŷ=0.5  → loss=0.69
                  se ŷ=0.01 → loss=4.60  ← punição severa

Quando y=0:
  Quer ŷ próximo de 0
  Loss = -log(1-ŷ): se ŷ=0.1  → loss=0.10
                    se ŷ=0.5  → loss=0.69
                    se ŷ=0.99 → loss=4.60  ← punição severa
```

### 4.4. O Gradiente — O que é e como ele aponta a direção certa?

O gradiente é um conceito do cálculo vetorial. Para uma função de múltiplas variáveis (como a loss em função de todos os pesos), o gradiente é o vetor das derivadas parciais — ele aponta na direção de **máximo crescimento** da função.

**Por que subtraímos o gradiente:** queremos minimizar a loss, não maximizá-la. Por isso, movemos os pesos na direção **oposta** ao gradiente — daí o nome "descida do gradiente" (Gradient Descent).

**Intuição geométrica:** imagine a superfície de loss como uma paisagem montanhosa. O gradiente aponta morro acima. Subtraindo o gradiente, caminhamos morro abaixo, em direção ao vale (mínimo da loss).

**A elegância da Regressão Logística:** o gradiente da BCE com Sigmoid tem uma forma analítica extraordinariamente simples:

```
∂L/∂wᵢ = (1/N) · Σ (ŷⱼ - yⱼ) · xᵢⱼ

Isso é apenas: média dos resíduos ponderados pelas features.
A complexidade do cálculo encadeado (chain rule) se cancela
cancelamento que ocorre porque BCE e Sigmoid foram
escolhidas como par complementar.
```

### 4.5. Gradient Descent — Como a learning rate controla o aprendizado?

O Gradient Descent é o otimizador que guia o aprendizado. A learning rate η é seu único hiperparâmetro — e o mais crítico.

**O que acontece com η muito alto:**

- Os passos são grandes demais
- O modelo ultrapassa o mínimo e oscila
- A loss pode divergir (crescer indefinidamente)

**O que acontece com η muito baixo:**

- Os passos são minúsculos
- O treinamento é lento — muitas épocas para convergir
- Risco de ficar preso em mínimos locais (em modelos não-convexos)

**Variantes do Gradient Descent usadas na prática:**

| Variante | Calcula gradiente sobre | Velocidade | Estabilidade |
| --- | --- | --- | --- |
| Batch GD | Todo o dataset | Lento | Alta |
| Stochastic GD (SGD) | 1 observação | Rápido | Baixa (ruidoso) |
| Mini-batch GD | Subconjunto (batch) | Equilibrado | Equilibrada |

**Na prática com Scikit-Learn:** a Regressão Logística do sklearn usa por padrão o solver `lbfgs` (quasi-Newton) — mais eficiente que o GD puro para datasets de tamanho médio. Para datasets grandes, use `solver='saga'` com `max_iter` ajustado.

```python
from sklearn.linear_model import LogisticRegression

# Uso padrão
model = LogisticRegression(
    C=1.0,              # C = 1/λ → inverso da regularização L2
    solver='lbfgs',     # otimizador (quasi-Newton)
    max_iter=1000,      # máximo de iterações
    random_state=42
)
model.fit(X_train, y_train)

# Acessar os coeficientes aprendidos
print("Pesos (W):", model.coef_)
print("Bias (b):", model.intercept_)

# Probabilidades e predições
probs  = model.predict_proba(X_test)[:, 1]  # P(y=1)
preds  = model.predict(X_test)              # classe predita
```

### 4.6. Regularização L2 — O parâmetro C do sklearn

A Regressão Logística do sklearn inclui regularização L2 por padrão, controlada pelo parâmetro C. A regularização adiciona uma penalidade à loss que desencoraja pesos muito grandes, reduzindo overfitting.

**Fórmula da loss com regularização L2:**

```
L_reg = L_BCE + (λ/2) · Σ wᵢ²

No sklearn, C = 1/λ:
  C grande (ex: C=100) → λ pequeno → pouca regularização → modelo mais complexo
  C pequeno (ex: C=0.01) → λ grande → muita regularização → pesos forçados a zero
  C=1.0 é o default — ponto de equilíbrio
```

**Quando ajustar C:** use GridSearchCV ou a classe LogisticRegressionCV para encontrar o C ótimo por cross-validation.