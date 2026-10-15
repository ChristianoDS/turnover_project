## Objetivo da Análise Univariada

O objetivo da análise univariada é medir o grau de discriminiação ou separabilidade da variável preditora entre as classes da variável resposta e identificar os potenciais outliers.

Em outras palavras, queremos saber: *"essa variável consegue distinguir um cliente bom de um mau pagador? Um usuário que vai churnar do que vai ficar?"*

Uma variável com **alta separação** tem distribuições bem distintas entre as classes diferenciando os grupos. Uma variável com **baixa separação** tem distribuições sobrepostas, contribuindo pouco para o modelo.

## 2. Testes de Separabilidade

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