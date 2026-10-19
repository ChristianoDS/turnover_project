## Objetivo da Análise Univariada

O objetivo da análise univariada é medir o grau de discriminiação ou separabilidade da variável preditora entre as classes da variável resposta e identificar os potenciais outliers.

Em outras palavras, queremos saber: *"essa variável consegue distinguir um cliente bom de um mau pagador? Um usuário que vai churnar do que vai ficar?"*

Uma variável com **alta separação** tem distribuições bem distintas entre as classes diferenciando os grupos. Uma variável com **baixa separação** tem distribuições sobrepostas, contribuindo pouco para o modelo.

## 2. Testes de Separabilidade para dados numéricos

### 2.1. KS Statistics (Kolgomorov-Sminov)

O KS compara as distribuições acumuladas de uma variável numérica entre duas classes (ex: inadimplente vs. adimplente). Em cada ponto do eixo X, calculamos a diferença entre as duas CDFs — o KS é o valor máximo dessas diferenças.

- 2.1.1. Baixa Separação
    
    !image.png
    
- 2.1.2. Alta Separação
    
    !image.png
    

### 2.2. Overlap Coefficient

Mede diretamente a sobreposição entre as distribuições das duas classes, quanto menos sobreposição, melhor a separação. 

É a medida mais intuitiva: ele calcula a área de interseção entre as duas distribuições de probabilidade. Uma sobreposição de 0 significa separação perfeita; uma sobreposição de 1 significa que as duas distribuições são idênticas e a variável não discrimina nada.

- 2.2.1. Baixa Separação
    
    !image.png
    
- 2.2.2. Alta Separação
    
    !image.png
    

### 2.3. AUC Univariate

A AUC Univariada usa a variável numérica diretamente como "score" de um classificador trivial. Ela pergunta: *se eu ranquear todos os exemplos por esse atributo, com que frequência o exemplo positivo vem antes do negativo?*

- 2.3.1. Baixa Separação
    
    !image.png
    
- 2.3.2. Alta Separação
    
    !image.png
    

## 3. Deteção de Outlier

### 3.1. Método IQR

O **Intervalo Interquartil (IQR)** é a distância entre o primeiro quartil (Q1, 25%) e o terceiro quartil (Q3, 75%). Ele captura os 50% centrais dos dados, a faixa onde a maioria dos valores "normais" se concentra. 

Ao usar o IQR como régua, conseguimos identificar o quanto um valor se afasta do comportamento típico.

Se um valor está muito além de onde 75% dos dados estão concentrados, ele é um candidato a outlier. 

O IQR mede "quão longe é longe demais" de forma resistente a valores extremos, diferente da média e do desvio padrão, que já são distorcidos pelos próprios outliers.

```python
IQR = Q3 − Q1

Limite inferior leve = Q1 − 1.5 × IQR
Limite superior leve = Q3 + 1.5 × IQR

Limite inferior extremo = Q1 − 3.0 × IQR
Limite superior extremo = Q3 + 3.0 × IQR
```

## 4. Testes de Separabilidade para dados categóricos

## Categorical univariate analysis

Variáveis categóricas não têm uma escala numérica contínua, então não dá para simplesmente calcular uma média ou ordenar os valores. 

O que podemos fazer é analisar como a **frequência de cada categoria se distribui entre as classes** da variável resposta. 

Se a distribuição muda muito de uma classe para outra, a variável tem poder discriminativo.

**Objetivo:** para cada categoria da variável preditora, observar se a proporção de eventos difere da proporção global. 

Quanto maior essa diferença, maior o poder de separação e mais útil a variável é para o modelo.

## 2. Métodos de Separação

## 2.1. Test Qui-Quadrado

Testa se a distribuição de uma variável categórica é independente da variável resposta. 

Quanto maior o χ², menor a chance de a associação ser por acaso.

O Chi-Square compara as **frequências observadas** (o que vemos nos dados) com as **frequências esperadas** (o que veríamos se não houvesse nenhuma relação entre as variáveis). Se a diferença entre observado e esperado for grande, a variável tem associação com a resposta.

**Limitação importante:** o χ² é sensível ao tamanho da amostra. Com N muito grande, mesmo uma associação praticamente irrelevante gera um χ² alto e p-valor significativo. Por isso, ele responde apenas se há associação — mas não diz o quanto ela é forte. Para isso, usamos o Cramér's V.

## 2.2. Cramer’s V

Normaliza o χ² para produzir um índice de associação entre 0 e 1 — independente do tamanho da amostra e do número de categorias.

O Cramér's V corrige os dois problemas do Chi-Square: ele é **insensível ao tamanho da amostra** (N) e **comparável entre variáveis com números diferentes de categorias**. Isso o torna o método ideal para ranquear variáveis categóricas pelo poder de associação.

## 2.3. AUC Univariate

Converte cada categoria em um score numérico e sua taxa de evento para então aplica a AUC como classificador trivial.

Para variáveis categóricas, calculamos a AUC substituindo cada categoria pelo seu **target rate** (proporção de eventos naquela categoria). Isso cria um score ordinal. 

A AUC mede se esse ranqueamento separa bem os eventos dos não-eventos, sendo diretamente comparável à AUC de variáveis numéricas.