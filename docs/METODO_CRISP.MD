CRISP-DM significa: Cross-Industry Standard Process for Data Mining (Processo Padrão Interindustrial para Mineração de Dados)

Ele surgiu no final da década de 1990, desenvolvido por um consórcio de empresas como:

- SPSS
- Daimler-Benz
- NCR

O objetivo era criar um **padrão estruturado** para conduzir projetos de Data Mining (hoje chamado de Data Science). Mesmo após mais de 20 anos, continua sendo a metodologia mais utilizada no mercado.

O CRISP-DM é dividido em **6 fases principais**:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

Se você já comprou algo online, já foi influenciado por um recomendador.

### **Fase 1: Entendimento do Negócio**

Esta é a etapa mais crítica. Aqui, o objetivo é converter os objetivos da empresa em um problema de Ciência de Dados.

- **O que fazemos:** Definimos o que é sucesso para o projeto.
- **Exemplo:** A empresa quer "vender mais". O Cientista traduz para: "Prever a probabilidade de compra de cada cliente para otimizar campanhas de marketing".

### **Fase 2: Entendimento dos Dados**

Aqui você coleta os dados e começa a se familiarizar com eles.

- **O que fazemos:** Verificamos a qualidade, o volume e se os dados realmente respondem às perguntas de negócio.
- **Dica:** É nesta fase que descobrimos se faltam dados cruciais para o projeto.

### **Fase 3: Preparação dos Dados**

É a fase onde "limpamos a cozinha". Geralmente consome 70% a 80% do tempo do projeto.

- **O que fazemos:** Seleção de tabelas, limpeza de valores nulos, tratamento de outliers e criação de novas variáveis (Feature Engineering).

### **Fase 4: Modelagem ML**

É aqui que a "mágica" acontece, onde aplicamos os algoritmos de Machine Learning.

- **O que fazemos:** Selecionamos as técnicas de modelagem (Regressão, Florestas Aleatórias, Redes Neurais), calibramos os parâmetros e treinamos o computador para aprender os padrões.

### **Fase 5: Avaliação**

Antes de colocar o modelo no ar, precisamos saber se ele realmente funciona — não só tecnicamente, mas para o negócio.

- **O que fazemos:** Verificamos se o modelo atingiu a meta de negócio. Se o objetivo era reduzir custos, o modelo de ML consegue identificar os pontos de economia com precisão?
- **Ponto de Decisão:** Se a avaliação for ruim, voltamos para o **Entendimento de Negócio** ou **Preparação de Dados**. Se for boa, seguimos para o deploy.

### **Fase 6: Implantação**

O modelo sai do seu computador (Jupyter Notebook) e vai para o mundo real.

- **O que fazemos:** Integramos o modelo em um sistema, criamos um dashboard no Streamlit ou uma API para que os diretores da empresa possam usar os resultados para tomar decisões.

Ao olhar para o diagrama do CRISP-DM, você notará que as setas não apontam apenas para frente. Isso acontece porque a Ciência de Dados é um processo de **aprendizado contínuo**.

Exemplo Prático: Você está na fase de Modelagem e percebe que o modelo não está aprendendo bem. Ao investigar, você descobre que precisa de uma informação que não estava no seu dataset original. O que você faz? Você volta para a Preparação de Dados para criar essa nova variável.

O CRISP-DM permite esse ciclo de forma organizada, garantindo que você não se perca na complexidade do projeto