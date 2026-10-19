## Wrapper Methods Boruta

Os Wrapper Methods selecionam features treinando o modelo iterativamente — testam diferentes subconjuntos de variáveis e avaliam a performance de cada combinação. Capturam interações entre features, mas são computacionalmente caros.

### 1.1. Boruta

O Boruta é um algoritmo de seleção de features baseado em Random Forest. Sua lógica é elegante: ele cria cópias aleatórias de todas as features (chamadas de "shadow features") e as adiciona ao dataset. Em seguida, treina o modelo e verifica se cada feature real é mais importante do que as suas versões aleatórias.

Se uma feature não consegue superar nem a sua própria cópia embaralhada, ela provavelmente não carrega informação real.

**O processo do Boruta:**

1. Para cada feature real, cria uma cópia com valores embaralhados (shadow feature)
2. Treina um Random Forest no dataset aumentado (reais + shadows)
3. Registra a importância de cada feature
4. Uma feature real é "confirmada" se consistentemente supera a shadow feature mais importante
5. O processo se repete por várias iterações até a decisão convergir

Vantagem: considera interações entre features e é robusto contra overfitting. Limitação: lento em datasets com muitas features.