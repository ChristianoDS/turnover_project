## Abordagem 1: Filter Methods

Os Filter Methods avaliam cada variável individualmente usando métricas estatísticas, sem treinar nenhum modelo. São rápidos, escaláveis e independentes do algoritmo escolhido.

O trade-off é que eles não capturam interações entre variáveis — uma feature fraca individualmente pode ser poderosa em combinação com outras.

### 2.1. Variância Zero / Quasi-Zero

Uma variável com variância zero tem o mesmo valor em todas as observações — ela não carrega nenhuma informação. Uma variável com variância quasi-zero tem um único valor dominante (ex: 99% dos casos são 0).

Essas variáveis devem ser removidas antes de qualquer análise, pois não contribuem com discriminação alguma.

```python
from sklearn.feature_selection import VarianceThreshold

# Remove variáveis com variância abaixo do threshold
selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(num_attributes)

# Verificar variância manualmente
variances = num_attributes.var()
low_var_cols = variances[variances < 0.01].index.tolist()
print("Variáveis com variância quasi-zero:", low_var_cols)
```

**Critério de remoção:** variância < 0.01 ou quando uma categoria representa > 95% das observações.

### 2.2. Correlação de Pearson

A Correlação de Pearson mede a associação linear entre duas variáveis numéricas. No contexto de Feature Selection, é usada de duas formas:

1. **Correlação com o target:** mede o quanto a variável preditora está linearmente associada à variável resposta. Variáveis com correlação próxima de 0 com o target são candidatas à remoção.
2. **Correlação entre preditoras:** identifica multicolinearidade. Quando duas features têm correlação muito alta (|r| > 0.85), elas carregam informação redundante — uma pode ser removida sem perda significativa.

**Limitação:** Pearson só detecta relações lineares. Relações não-lineares passam despercebidas.

### 2.3. Correlação de Cramér's V

O Cramér's V é o equivalente do Pearson para variáveis categóricas. Ele mede a força da associação entre duas variáveis categóricas, normalizado entre 0 e 1.

No contexto de Feature Selection, é usado para:

1. **Associação com o target:** medir se uma variável categórica discrimina as classes da resposta.
2. **Redundância entre preditoras:** identificar variáveis categóricas que carregam informação duplicada.

**Referência:** V < 0.10 → fraco · V 0.10–0.30 → moderado · V > 0.30 → forte

### 2.4. AUC Univariado

O AUC Univariado usa cada variável individualmente como score de um classificador trivial e mede a área sob a curva ROC. É o método mais direto para ranquear features pelo poder discriminativo — e funciona tanto para variáveis numéricas quanto categóricas (via target rate encoding).

A vantagem é que ele tem a mesma escala do AUC do modelo final, tornando direto comparar a contribuição individual de cada variável com a performance do modelo completo.

**Referência:** AUC > 0.55 → fraco · AUC > 0.70 → moderado · AUC > 0.80 → forte