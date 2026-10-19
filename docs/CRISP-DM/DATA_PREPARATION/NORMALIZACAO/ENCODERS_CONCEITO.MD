## 1. Encoders Categóricos

Modelos de Machine Learning trabalham com números e não com texto. Variáveis categóricas como "estado civil", "segmento", "região" precisam ser convertidas para representações numéricas antes de entrar no modelo.

A escolha do encoder impacta diretamente a qualidade do modelo: um encoder errado pode introduzir relações ordinais falsas, criar dimensionalidade explosiva ou vazar o target durante o treino.

### 1.1. Label Encoder

Atribui um número inteiro para cada categoria (0, 1, 2, ..., n-1). É simples e eficiente, mas cria uma **ordem implícita** entre as categorias — o modelo pode interpretar que a categoria 2 é "maior" ou "melhor" que a categoria 1, mesmo sem nenhuma relação ordinal real.

**Quando usar:** exclusivamente para variáveis com **ordem natural** (ex: baixo, médio, alto) ou como pré-processamento para modelos baseados em árvores que não são afetados pela escala.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Fit no treino, transform em treino e teste
X_train['education'] = le.fit_transform(X_train['education'])
X_test['education']  = le.transform(X_test['education'])

print(dict(zip(le.classes_, le.transform(le.classes_))))
# {'high_school': 0, 'college': 1, 'graduate': 2}
```

⚠️ **Nunca use LabelEncoder em variáveis nominais** (sem ordem) com modelos lineares ou baseados em distância.

### 1.2. OneHot Encoder

Cria uma coluna binária (0 ou 1) para cada categoria da variável. Elimina a ordem implícita do Label Encoder — cada categoria se torna uma dimensão independente.

**Quando usar:** variáveis **nominais** (sem ordem) com **baixa cardinalidade** (poucos valores únicos, ex: gênero, estado civil, tipo de produto).

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Com pandas (simples)
dummies = pd.get_dummies(X_train[['marital_status']], drop_first=True)

# Com sklearn (recomendado para pipelines)
ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# Exemplo em um ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'),
     ['marital_status', 'region'])
], remainder='passthrough')

X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc  = preprocessor.transform(X_test)
```

⚠️ **Cuidado com alta cardinalidade:** uma variável com 500 categorias gera 499 novas colunas, tornando o modelo lento e propenso a overfitting. Prefira Target Encoder nesses casos.

### 1.3. Ordinal Encoder

Codifica variáveis com **ordem natural** definida explicitamente pelo cientista de dados. Diferente do LabelEncoder, o OrdinalEncoder permite especificar a ordem correta das categorias, garantindo que a numeração reflita a hierarquia real.

**Quando usar:** variáveis **ordinais** com ordem conhecida (ex: nível de escolaridade, faixa de risco, grau de satisfação).

```python
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# Definir a ordem correta das categorias
ordinal_cols = ['risk_level', 'education_level']
orderings = [
    ['low', 'medium', 'high'],                        # risk_level
    ['high_school', 'college', 'graduate', 'phd']    # education_level
]

ord_enc = OrdinalEncoder(
    categories=orderings,
    handle_unknown='use_encoded_value',
    unknown_value=np.nan
)

X_train[ordinal_cols] = ord_enc.fit_transform(X_train[ordinal_cols])
X_test[ordinal_cols]  = ord_enc.transform(X_test[ordinal_cols])

print(X_train[ordinal_cols].head())
# risk_level: low=0, medium=1, high=2
```

### 1.4. Target Encoder

Substitui cada categoria pela **taxa média do target** naquela categoria. É o encoder mais indicado para variáveis com **alta cardinalidade** (muitas categorias únicas, ex: CEP, modelo de produto, cidade).

É poderoso porque captura a relação direta entre a categoria e a variável resposta, mas exige cuidado com **data leakage**: o target rate deve ser calculado **apenas no treino**, com suavização para categorias raras.

Usado muito quando se tem muitas variações categóricas.

```python
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, smoothing=10):
        self.cols = cols
        self.smoothing = smoothing
        self.encoding_map = {}
        self.global_mean = None

    def fit(self, X, y):
        self.global_mean = y.mean()
        for col in self.cols:
            stats = pd.DataFrame({'cat': X[col], 'target': y})\
                      .groupby('cat')['target']\
                      .agg(['mean', 'count'])
            # Suavização: categorias raras puxadas para a média global
            smooth = (stats['count'] * stats['mean'] +
                      self.smoothing * self.global_mean) / \
                     (stats['count'] + self.smoothing)
            self.encoding_map[col] = smooth.to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.encoding_map[col])\
                           .fillna(self.global_mean)
        return X

# Uso
high_card_cols = ['city', 'product_model', 'zip_code']
te = TargetEncoder(cols=high_card_cols, smoothing=10)
X_train[high_card_cols] = te.fit_transform(X_train[high_card_cols], y_train)
X_test[high_card_cols]  = te.transform(X_test[high_card_cols])
```

⚠️ **Nunca faça fit do Target Encoder com dados de teste.** Isso vaza o target para o modelo e infla artificialmente a performance.

### 1.5. Comparativo dos Encoders

| Encoder | Tipo de variável | Cardinalidade | Cuidado principal |
| --- | --- | --- | --- |
| Label Encoder | Ordinal ou árvores | Qualquer | Cria ordem implícita |
| OneHot Encoder | Nominal | Baixa (< 15 cats) | Explosão de dimensionalidade |
| Ordinal Encoder | Ordinal com ordem definida | Baixa / Média | Ordem deve ser especificada |
| Target Encoder | Nominal de alta cardinalidade | Alta (> 15 cats) | Data leakage se mal aplicado |
| Embedding | Categóric | Altíssima | Um passo extra no processo de Padronização dos dados. |