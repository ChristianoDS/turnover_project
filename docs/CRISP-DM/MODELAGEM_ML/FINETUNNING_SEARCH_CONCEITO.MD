## O que é Fine-Tuning?

O Fine-Tuning é o processo de encontrar a combinação ótima de **hiperparâmetros** de um modelo de Machine Learning — aqueles valores que não são aprendidos durante o treinamento, mas sim definidos pelo cientista de dados **antes** do treino começar.

### Parâmetros vs Hiperparâmetros

- **Parâmetros** são aprendidos pelo modelo a partir dos dados: coeficientes da Regressão Logística, pesos de uma Rede Neural, valores das folhas na XGBoost
- **Hiperparâmetros** são definidos pelo cientista de dados antes do treino: learning rate, número de árvores, profundidade máxima, regularização L1/L2, num_leaves

### Por que Fine-Tuning é necessário?

O mesmo algoritmo pode ter performances radicalmente diferentes dependendo dos hiperparâmetros escolhidos. Um modelo mal ajustado pode:

- Fazer **underfitting** (modelo muito simples — bias alto)
- Fazer **overfitting** (modelo muito complexo — variância alta)
- Treinar desnecessariamente lento sem ganho de performance

### O problema do espaço de busca

O conjunto de todos os possíveis valores de hiperparâmetros forma um **espaço de busca** de alta dimensão. Testar todas as combinações possíveis é computacionalmente inviável — por isso existem estratégias inteligentes de busca.

```python
# Exemplo de espaço de busca para XGBoost
# 4 x 3 x 5 x 4 x 4 x 4 x 4 = 15.360 combinações possíveis

param_space = {
    'n_estimators':    [100, 500, 1000, 2000],
    'learning_rate':   [0.01, 0.05, 0.1],
    'max_depth':       [3, 4, 5, 6, 8],
    'min_child_weight':[1, 3, 5, 10],
    'subsample':       [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree':[0.6, 0.7, 0.8, 0.9],
    'reg_lambda':      [0.1, 1.0, 5.0, 10.0],
}

# Testar todas: 15.360 modelos x 5-fold CV = 76.800 treinamentos
# Com 1 min por treino = 53 dias. Impraticável.
```

## 2. Técnica 3: Bayesian Search

A Bayesian Search é a abordagem mais inteligente das três. Ela **aprende com cada avaliação realizada** e usa esse conhecimento para decidir qual próxima combinação de hiperparâmetros tem mais chance de melhorar o resultado. Em vez de explorar o espaço aleatoriamente, a busca bayesiana constrói um **modelo substituto (surrogate model)** da função objetivo e o usa para guiar a busca.

### Como funciona

O processo se repete iterativamente:

1. Avalia algumas combinações iniciais aleatoriamente (warm-up)
2. Ajusta um modelo substituto (Gaussian Process ou Tree Parzen Estimator) sobre os resultados observados
3. Usa uma **função de aquisição** para escolher a próxima combinação com maior potencial
4. Avalia essa combinação, atualiza o modelo substituto e repete

### Vantagens

- **Muito mais eficiente** que Grid Search e Random Search: encontra soluções melhores com menos avaliações
- Equilíbrio automático entre exploração (regiões inexploradas) e explotação (regiões promissoras)
- Ideal quando cada avaliação é cara (modelos lentos, datasets grandes)

### Desvantagens

- Mais complexo de configurar e interpretar
- Avaliações são **sequenciais** por padrão (cada ponto depende dos anteriores), dificultando paralelização pura
- O modelo substituto tem custo computacional próprio (pequeno, mas presente)

### Quando usar

- Modelos caros de treinar (LightGBM, XGBoost com datasets grandes, Redes Neurais)
- Orçamento muito limitado (< 50 avaliações)
- Quando se quer extrair o máximo de performance com o mínimo de tempo

### Implementação com Optuna

```python
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    """Função objetivo: cada chamada treina e avalia uma combinação"""
    params = {
        'n_estimators':       trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate':      trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'max_depth':          trial.suggest_int('max_depth', 3, 10),
        'min_child_weight':   trial.suggest_int('min_child_weight', 1, 20),
        'subsample':          trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':   trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_lambda':         trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
        'reg_alpha':          trial.suggest_float('reg_alpha', 0.001, 1.0, log=True),
        'use_label_encoder':  False,
        'eval_metric':        'auc',
        'random_state':       42,
        'n_jobs':             -1
    }

    model = XGBClassifier(**params)
    scores = cross_val_score(
        model, X_train, y_train,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1
    )
    return scores.mean()

# Criar e executar o estudo
study = optuna.create_study(
    direction='maximize',                        # maximizar AUC
    sampler=optuna.samplers.TPESampler(seed=42)  # Tree Parzen Estimator
)

study.optimize(
    objective,
    n_trials=100,        # número de avaliações
    timeout=3600,        # limite de 1 hora (opcional)
    show_progress_bar=True
)

# Resultados
print(f"Melhor AUC (CV):     {study.best_value:.4f}")
print(f"Melhores parâmetros: {study.best_params}")

# Retreinar com os melhores parâmetros no treino completo
best_model = XGBClassifier(**study.best_params,
                            use_label_encoder=False,
                            eval_metric='auc',
                            random_state=42,
                            n_jobs=-1)
best_model.fit(X_train, y_train)
print(f"AUC Validação: {roc_auc_score(y_val, best_model.predict_proba(X_val)[:,1]):.4f}")
```

### Visualizando o processo de otimização com Optuna

```python
import optuna.visualization as vis

# Histórico de otimização — como a AUC evoluiu
vis.plot_optimization_history(study).show()

# Importância dos hiperparâmetros
vis.plot_param_importances(study).show()

# Relação entre hiperparâmetro e AUC
vis.plot_slice(study, params=['learning_rate', 'max_depth']).show()
```

## 3. Comparativo das 3 Técnicas

| Critério | Grid Search | Random Search | Bayesian Search |
| --- | --- | --- | --- |
| Estratégia | Exaustiva | Aleatória | Guiada por modelo substituto |
| Eficiência | Baixa (exponencial) | Média | Alta |
| Aprende com resultados | Não | Não | Sim |
| Paralelizável | Sim | Sim | Parcialmente |
| Espaços contínuos | Não | Sim | Sim |
| Reprodutibilidade | Total | Com random_state | Com seed fixo |
| Complexidade de configuração | Baixa | Baixa | Média |
| Melhor para | Espaços pequenos | Exploração inicial ampla | Orçamento limitado e modelos caros |

### Estratégia recomendada no CRISP

1. **Random Search** com n_iter=50–100 para explorar o espaço e identificar regiões promissoras
2. **Bayesian Search** (Optuna) para refinar nas regiões promissoras, especialmente se o treino é lento
3. **Grid Search** para busca exaustiva final em um espaço estreito e bem definido

### Boas práticas gerais

- Sempre use **cross-validation** para avaliar cada combinação — nunca o conjunto de validação diretamente
- Use **early stopping** dentro do treino de cada modelo para evitar n_estimators excessivo
- Fixe `random_state` para reprodutibilidade
- Monitore **overfitting**: AUC de treino muito acima do CV indica modelo muito complexo
- Após encontrar os melhores hiperparâmetros, retreine o modelo final no **dataset completo** (treino + validação)