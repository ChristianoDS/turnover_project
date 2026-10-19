## Wrapper Methods RFE

Os Wrapper Methods selecionam features treinando o modelo iterativamente — testam diferentes subconjuntos de variáveis e avaliam a performance de cada combinação. Capturam interações entre features, mas são computacionalmente caros.

### 1.1. RFE: Recursive Feature Elimination

O RFE treina o modelo, identifica as features menos importantes, remove-as e repete o processo recursivamente. A cada iteração, o conjunto de features encolhe até restar apenas o número desejado.

Diferente do Boruta, o RFE não tem um critério estatístico de parada — você define quantas features quer no final. Isso pode ser um problema se o número ideal for desconhecido.

**O processo do RFE:**

1. Treina o modelo com todas as features disponíveis
2. Calcula a importância de cada feature no modelo treinado
3. Remove as N features menos importantes
4. Repete até atingir o número de features desejado

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier

# Preparação simplificada para o Boruta (Encoding Temporário)
X = df03.drop(columns=['Attrition'])
y = df03['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

X_rfe = pd.get_dummies(X)

# RFE com número fixo de features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=10, step=1)
rfe.fit(X_rfe, y)

features_selecionadas = X_rfe.columns[rfe.support_].tolist()
print("Features selecionadas pelo RFE:", features_selecionadas)

# RFECV: encontra automaticamente o número ótimo via cross-validation
rfecv = RFECV(
    estimator=rf,
    step=1,
    cv=5,
    scoring='roc_auc',
    min_features_to_select=5
)
rfecv.fit(X_rfe, y)

print(f"Número ótimo de features: {rfecv.n_features_}")
print("Features:", X_rfe.columns[rfecv.support_].tolist())
```

**Dica:** use o RFECV (com cross-validation) sempre que possível, ele encontra automaticamente o número ideal de features pela performance de validação.