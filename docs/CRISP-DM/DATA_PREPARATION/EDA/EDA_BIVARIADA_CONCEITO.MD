## Objetivo da Análise Bivariada

O objetivo da análise univariada é medir o grau de discriminiação ou separabilidade da variável preditora entre as classes da variável resposta e identificar os potenciais outliers.

Em outras palavras, queremos saber: *"essa variável consegue distinguir um cliente bom de um mau pagador? Um usuário que vai churnar do que vai ficar?"*

Uma variável com **alta separação** tem distribuições bem distintas entre as classes diferenciando os grupos. Uma variável com **baixa separação** tem distribuições sobrepostas, contribuindo pouco para o modelo.

## 2. Variáveis Numéricas

### 2.1. Boxplot por classe

Mede a mediana e spread. Visão rápida da separação

### 2.2. Violin plot

Distribuição completo e outliers por classe. Quando a forma importa

### 2.3. Probability Plot

Taxa da variável resposta conforme a variável aumenta. Entender a direção do efeito.

### 2.4. Correlação ponto-biserial

Correlação numérica x binário. Medir a força linear

## 3. Variáveis Categóricas

### 3.1. Stacked bar (% normalizado)

Esse gráfico mostra a proporção de Yes/No dentro de cada categoria e assim, permite uma comparação visual direta

### 3.2. Variável Resposta por Categoria

Mostra qual categoria tem mais risco, facilitando a interpretação do negócio. 

### 3.3. Heatmap de Contingência

Mostrar a proporção e o percentual de cada valor da categoria dentro de cada classe

### 3.4. Cramer’s V

Calcula a força de associação entre as proporções da tabela de frequência