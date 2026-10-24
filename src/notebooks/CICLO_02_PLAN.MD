# Ciclo 02

## Para problemas de classificação binária
Apoós realizado o ciclo 1 (CICLO_01_PLNA.MD), o ciclo 02 deve contemplar:
- Tudo que já foi contemplado no ciclo 01;
- Feature Engineering - Criação de novas features, dando preferencia para aquelas com relação/taxas.
- EDA (
    - Univariada: 3 testes: Aplicar from scipy import stats, KS Statistic (Kolmogorov-Smirnov), 
    - Coeficiente de overlap.
    - Treinar uma logistic regression com somente uma variavel (from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# AUC Univariado
X = df2[['Age']]
y = LabelEncoder().fit_transform(df2['Attrition'])

model = LogisticRegression(max_iter=500)
model.fit(X, y)
y_prob = model.predict_proba(X)[:, 1]

auc = roc_auc_score(y, y_prob)
print(f"\nAUC Univariado para Age: {auc:.4f}")
print(f"Interpretacao: {'Descriminativo' if auc > 0.55 else 'Fraco'}"))

O ideal seria um df com as colunas Feature	Mean (No)	Mean (Yes)	Std (No)	Std (Yes)	KS Stat	KS p-value	AUC Univariado	Overlap Coef.	N Outliers	% Outliers	Discriminação e a analise de cada linha de variavel numerica.

com # --- Classificação automática por poder discriminativo ---
def classify(row):
    if row['AUC Univariado'] >= 0.65:
        return '🟢 Alta'
    elif row['AUC Univariado'] >= 0.55:
        return '🟡 Média'
    else:
        return '🔴 Baixa'

df_train_analysis['Discriminação'] = df_train_analysis.apply(classify, axis=1)

Para as variáveis categoricas univariada, adotar Chi square, kramer-v e Roc Auc de regressao logistica. # --- 3. AUC Univariado ---
X_enc = LabelEncoder().fit_transform(series).reshape(-1, 1)
try:
    model = LogisticRegression(max_iter=300).fit(X_enc, y)
    auc   = roc_auc_score(y, model.predict_proba(X_enc)[:, 1])
    auc   = max(auc, 1 - auc)
except Exception:
    auc = np.nan


    - Bivariada:
    - Multivariada: Analise heatmap # Matriz de Correlação Mista: Pearson + Cramér's V 
# Cramer's V para dados categóricos
)

Trazer tabela de correlação das variáveis com a variavel alvo e trazer dataframe de multicolinearidade entre as variáveis explicativas e lista de todas as colunas com possivel remoção.
- Feature Selection com Boruta e mais um seletor