## O que é Feature Engineering?

Feature Engineering é o processo de transformar dados brutos em variáveis (features) que aumentam a capacidade do modelo de aprender padrões relevantes. Em termos simples: Feature Engineering é ensinar o algoritmo a enxergar melhor o problema.

Os modelos de Machine Learning não “entendem” contexto. Eles não sabem o que é um funcionário desmotivado, um cliente insatisfeito ou um colaborador sobrecarregado. Eles apenas enxergam números.

O papel do Cientista de Dados é traduzir o problema de negócio em variáveis que carreguem significado estatístico.

## 3. O que é uma Feature?

Feature é qualquer variável utilizada como entrada para o modelo.

Exemplo no dataset de rotatividade de RH:

- Age
- MonthlyIncome
- JobSatisfaction
- OverTime
- YearsAtCompany

Mas essas são variáveis originais.

Feature Engineering começa quando criamos variáveis novas ou transformamos as existentes.

## 4. O Processo de Feature Engineering

Feature Engineering não é apenas criar colunas novas.

É um processo estruturado que envolve:

- Transformação de variáveis
- Criação de novas variáveis
- Codificação de variáveis categóricas
- Escalonamento
- Agrupamentos
- Interações entre variáveis
- Extração de informação temporal
- Redução de dimensionalidade

Ele ocorre geralmente na fase de **Preparação dos Dados (CRISP-DM)**.

É um processo criativo e analítico ao mesmo tempo.

## 5. A importância do Feature Engineering

Na prática, um modelo simples com boas features pode superar um modelo complexo com dados mal preparados.

O algoritmo aprende padrões com base nas variáveis fornecidas.

Se as variáveis não carregam informação útil, o modelo não tem como aprender algo relevante.

Feature Engineering melhora:

- Separabilidade das classes
- Capacidade de generalização
- Estabilidade do modelo
- Interpretabilidade
- Performance nas métricas (Recall, Precision, F1)

## 6. Exemplos práticos

### Ex. 1: Transformação Simples

Variável original: YearsAtCompany

Nova feature: IsNewEmployee = 1 se YearsAtCompany < 2, senão 0

Isso pode capturar comportamento específico de funcionários recém-contratados.

### Ex. 2: Interação entre Variáveis

MonthlyIncome e JobLevel separadamente podem ter impacto moderado.

Mas a relação:

IncomeRelativeToLevel = MonthlyIncome / Média do nível

Pode indicar insatisfação salarial relativa.

Isso adiciona contexto.

### Ex. 3: Transformação Logarítmica

Se salários têm distribuição muito assimétrica (skew alto), aplicar log pode:

- Reduzir distorção
- Melhorar modelos lineare