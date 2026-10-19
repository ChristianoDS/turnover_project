## Por que Normalizar e Reescalar?

Muitos algoritmos de Machine Learning são sensíveis à escala das variáveis. Modelos como Regressão Logística, SVM, KNN e Redes Neurais calculam distâncias ou gradientes  e se uma variável está na escala de 0 a 1 e outra de 0 a 100.000, a segunda vai dominar o aprendizado de forma injustificada.

O objetivo da preparação numérica é colocar todas as features em uma escala comparável, sem destruir a informação que elas carregam.

Algoritmos baseados em árvores, trabalham com cortes, subdivisão (Lazy algoritimos), não usam multiplicação em sua estrutura, e por isso, tanto faz usar rescaling e normalização.

### Quando normalizar vs. reescalar?

- Use **StandardScaler** quando a variável segue uma distribuição aproximadamente Normal (Gaussiana)
- Use **MinMaxScaler** quando a variável tem distribuição Não-Normal, assimétrica ou com outliers controlados
- **Modelos baseados em árvores** (Random Forest, XGBoost, LightGBM) **não precisam** de normalização ou rescaling — são invariantes à escala

## 3. Estratégia Completa de Preparação

A ordem correta de aplicação em um projeto CRISP é fundamental para evitar data leakage e garantir resultados reproduzíveis.

### Ordem recomendada

1. **Split treino / teste** — sempre o primeiro passo
2. **Tratar valores ausentes** — imputar com estatísticas do treino
3. **Encoders categóricos** — fit no treino, transform em treino e teste
4. **Normalização / Rescaling** — fit no treino, transform em treino e teste
5. **Feature Selection** — baseada no conjunto de treino preparado