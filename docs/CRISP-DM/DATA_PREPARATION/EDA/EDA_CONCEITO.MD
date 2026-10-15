## O que é uma EDA?

EDA significa **Exploratory Data Analysis**, ou Análise Exploratória de Dados.

É o processo de investigar, resumir e visualizar os dados antes de aplicar qualquer modelo estatístico ou algoritmo de Machine Learning.

EDA é a etapa onde o Cientista de Dados desenvolve intuição sobre o dataset.

## 2.1 Os objetivos da EDA

A EDA tem quatro grandes objetivos:

1. Entender a estrutura dos dados
2. Identificar padrões e tendências
3. Detectar problemas e inconsistências
4. Gerar hipóteses

Ela é tanto técnica quanto investigativa. Não é apenas fazer gráficos. É interpretar o que os gráficos significam.

## 2.2 Os tipos

A EDA pode ser dividida em três níveis principais:

- Análise Univariada
- Análise Bivariada
- Análise Multivariada

Cada uma tem um propósito específico.

## 3. Análise Univariada

É a análise de uma única variável por vez. Ela busca entender o comportamento individual de cada coluna.

A análise univariada cria a base de entendimento. Ela responde: “Essa variável faz sentido sozinha?”

### Objetivo

- Entender distribuição
- Identificar outliers
- Avaliar dispersão
- Detectar assimetria
- Identificar erros

### Ferramentas comuns

- Histogramas
- Boxplots
- Estatísticas descritivas
- Método IQR (Outliers)
- Frequência de categorias

## 4. Análise Bivariada

É a análise da relação entre duas variáveis, sendo uma delas a variável alvo (target). A análise bivariada ajuda a responder: “Essa variável está associada ao problema que quero prever?” Ela começa a indicar quais variáveis podem ser importantes para o modelo.

### Objetivo

- Identificar correlações
- Verificar associação entre variável e target
- Detectar padrões comparativos
- Testar hipóteses iniciais

### Ferramentas comuns

- Gráficos de dispersão (Visual)
- Boxplot por classe (Visual)
- Tabelas de contingência (Visual)
- Correlação (Tabular)
- Testes estatísticos

## 5. Análise Multivariada

É a análise da relação entre três ou mais variáveis simultaneamente. Ela busca entender interações e revela relações que não são visíveis em análises isoladas.

### Objetivo

- Identificar interações entre variáveis
- Detectar colinearidade
- Avaliar impacto conjunto
- Descobrir padrões ocultos

### Ferramentas comuns

- Matriz de correlação
- Pairplots
- Heatmaps
- Análises segmentadas
- Modelos exploratórios simples