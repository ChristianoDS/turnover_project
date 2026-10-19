## Rescaling com MinMaxScaler

O MinMaxScaler reescala a variável para o intervalo [0, 1] (ou qualquer intervalo definido). Diferente do StandardScaler, ele não assume nenhuma distribuição — apenas comprime ou expande os valores para caber no intervalo alvo.

É a escolha correta para distribuições não-normais, assimétricas (ex: renda, valor de transação) ou variáveis com limites naturais bem definidos.

### 1.1. A fórmula do MinMax

```
x_scaled = (x - x_min) / (x_max - x_min)

onde:
  x     = valor original
  x_min = valor mínimo da variável no conjunto de treino
  x_max = valor máximo da variável no conjunto de treino

Para intervalo customizado [a, b]:
x_scaled = a + (x - x_min) * (b - a) / (x_max - x_min)
```

Após a transformação:

- O valor mínimo do treino se torna 0
- O valor máximo do treino se torna 1
- Valores de teste fora do range do treino podem resultar em valores < 0 ou > 1

### 1.2. Implementação com Scikit-Learn

```python
from sklearn.preprocessing import MinMaxScaler

# Features com distribuição não-normal (assimétrica)
num_cols_skewed = ['transaction_value', 'days_since_last_purchase', 'debt_amount']

# Fit APENAS no treino
minmax = MinMaxScaler(feature_range=(0, 1))
X_train[num_cols_skewed] = minmax.fit_transform(X_train[num_cols_skewed])
X_test[num_cols_skewed]  = minmax.transform(X_test[num_cols_skewed])

# Verificar resultado
print(X_train[num_cols_skewed].describe().round(3))
# min ≈ 0.0, max ≈ 1.0
```

### 1.3. StandardScaler vs. MinMaxScaler

| Critério | StandardScaler | MinMaxScaler |
| --- | --- | --- |
| Distribuição ideal | Normal (Gaussiana) | Qualquer distribuição |
| Resultado | média=0, std=1 | valores entre [0, 1] |
| Sensível a outliers | Moderado | Alto — outliers distorcem o range |
| Preserva o formato | Sim | Sim |
| Indicado para | Regressão, SVM, Redes Neurais | KNN, imagens, dados normalizados |

**Atenção com outliers no MinMax:** um único valor extremo no treino comprime todos os outros em uma faixa estreita. Se o dataset tiver outliers, trate-os antes de aplicar o MinMaxScaler ou use o RobustScaler como alternativa.

Existe tambem o robust scaler.

## Robust Scaler
- Para lidar com outliers
formula = (Valor - Q1)/(Q3-Q1)