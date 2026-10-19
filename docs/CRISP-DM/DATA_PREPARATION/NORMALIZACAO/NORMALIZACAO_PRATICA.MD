## Normalização com StandardScaler

A normalização transforma a variável para que ela tenha **média 0** e **desvio padrão 1**. O resultado é chamado de Z-Score ou variável padronizada.

Esse processo é matematicamente correto quando a variável segue uma distribuição Normal, pois a média e o desvio padrão são estimativas estáveis e representativas da distribuição.

### 1.1. A fórmula do Z-Score

```
z = (x - μ) / σ

onde:
  x = valor original
  μ = média da variável no conjunto de treino
  σ = desvio padrão da variável no conjunto de treino
```

Após a transformação:

- Valores abaixo da média terão z < 0
- Valores acima da média terão z > 0
- A distribuição mantém seu formato — apenas a escala muda

### 1.2. Implementação com Scikit-Learn

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Separar treino e teste ANTES de qualquer transformação
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Identificar features numéricas com distribuição Normal
num_cols_normal = ['age', 'income', 'credit_score']  # exemplo

# Fit APENAS no treino — transform em treino e teste
scaler = StandardScaler()
X_train[num_cols_normal] = scaler.fit_transform(X_train[num_cols_normal])
X_test[num_cols_normal]  = scaler.transform(X_test[num_cols_normal])

# Verificar resultado
print(X_train[num_cols_normal].describe().round(3))
# média ≈ 0, std ≈ 1
```

### 1.3. Regra de ouro: fit no treino, transform no teste

Este é o erro mais comum na preparação de dados. O scaler deve aprender (fit) os parâmetros — média e desvio padrão — **exclusivamente no conjunto de treino**.

Aplicar fit_transform no dataset completo (antes do split) é **data leakage**: o modelo passa a conhecer informações do futuro durante o treinamento, e a performance estimada no teste não reflete a realidade.

```python
# ❌ ERRADO — data leakage
X_scaled = scaler.fit_transform(X)  # aprende com treino E teste juntos

# ✅ CORRETO — sem leakage
scaler.fit(X_train)                  # aprende APENAS com treino
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)  # aplica os parâmetros do treino
```