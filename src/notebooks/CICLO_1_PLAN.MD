# PLANEJAMENTO RASCUNHO DE CICLO 01

## PROBLEMAS DE CLASSIFICAÇÃO BINÁRIA
- A ideia do ciclo inicial é ter uma validação rápida dos modelos de machine learning.
- Essa etapa é sugerido conter os seguintes passos:
    - Carregamento dos dados
    - Split dos dados em treino - teste
    - Split dos dados de treino em treino - validação, considerando uma divisão estratificada.
    - Limpeza dos dados de treino (Limpeza de Na, podemos usar uma media ou mediana nos Na)
    - Verificar balanceamento dos dados
    - Verificar tipo de cada variável e converter se necessário
    - Preparação dos dados: Teste de normalização para saber se usa Standard Scaler, ou RobustScaler - Recomendo usar robustscaler nesse primeiro ciclo. Tambem incorpora os encoders, como padrao nesse ciclo, usamos o one hot.
    - Modelagem: Testar um algoritmo baseline padrao (como um DummyClassifier), além de outros modelos, como Logist Regression, Random Forest, XGBoost, LGBM, etc. Ainda nao usamos tunagem de hiperpametros e cross validation aqui. Ao final, geramos uma tabela comparativa com as metricas Accuracy	F1-Score	Precision	Recall	AUC. Recomendo usar curva ROC quando os dados foram equilibrados, e curva PR para dados desbalanceados.
    - Escolha do melhor threshold: = Recomendo gerar métricas como Threshold, F1-Score, Precision e Recall para por exemplo, recall minima de 0.8, precision de 0.7 e melhor F1-Score.