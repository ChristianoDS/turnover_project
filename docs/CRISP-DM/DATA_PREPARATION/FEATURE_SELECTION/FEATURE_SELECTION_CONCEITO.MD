## O que é Feature Selection?

Feature Selection, ou Seleção de Features, é o processo de identificar e selecionar o subconjunto de variáveis preditoras mais relevantes para um modelo de Machine Learning.

O objetivo não é usar todas as variáveis disponíveis, é usar as certas.

Variáveis irrelevantes ou redundantes aumentam a complexidade do modelo, reduzem a performance, aumentam o risco de overfitting e dificultam a interpretação dos resultados.

### Por que fazer Feature Selection?

- Reduzir a dimensionalidade do problema
- Melhorar a performance e generalização do modelo
- Reduzir tempo de treinamento
- Eliminar ruído trazido por variáveis irrelevantes
- Melhorar a interpretabilidade

### As 3 abordagens

Existem três grandes famílias de métodos de Feature Selection:

1. **Filter Methods:** seleção baseada em estatística, sem usar o modelo
2. **Wrapper Methods:** seleção baseada na performance do modelo
3. **Embedded Methods:** a importância das features emerge de dentro do próprio modelo

## 3. Comparativo das 3 Abordagens

| Abordagem | Velocidade | Capta interações | Depende do modelo | Quando usar |
| --- | --- | --- | --- | --- |
| Filter Methods | Alta | Não | Não | Primeiro filtro — eliminar ruído antes de modelar |
| Wrapper Methods | Baixa | Sim | Sim | Quando precisar do subconjunto ótimo de features |
| Embedded Methods | Média | Sim | Sim | Quando já tem um modelo definido e quer interpretabilidade |

### Estratégia recomendada na prática

1. **Comece com Filter Methods** para eliminar variáveis obviamente irrelevantes (variância zero, AUC < 0.55)
2. **Use Embedded Methods** (SHAP ou Feature Importance) para um ranking mais refinado com o modelo escolhido
3. **Aplique Wrapper Methods** (Boruta ou RFECV) quando precisar de um resultado rigoroso e tiver tempo computacional disponível