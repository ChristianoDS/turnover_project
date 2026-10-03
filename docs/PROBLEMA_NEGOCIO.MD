# CONTEXTO GERAL

A **InovaTech Solutions**, uma gigante do setor de tecnologia com mais de 1.400 colaboradores, está enfrentando um sinal de alerta. O departamento de Recursos Humanos (RH) notou que o índice de rotatividade subiu drasticamente no último ano, especialmente em cargos técnicos e de gestão.

A diretoria está preocupada porque a empresa está perdendo talentos para a concorrência e o custo de novas contratações está corroendo a margem de lucro.

**Você foi contratado(a) como Cientista de Dados** para ajudar o RH a agir de forma **proativa** em vez de reativa.

O RH possui uma base de dados com 35 variáveis, incluindo:

- **Satisfação no Trabalho:** Como o funcionário se sente.
- **Remuneração:** Salário mensal, bônus e aumentos.
- **Equilíbrio Vida/Trabalho:** Distância de casa, horas extras e viagens.
- **Histórico:** Anos na empresa, promoções e tempo no cargo atual.

**Sua missão é:**

Ajudar o time de RH a diminuir a rotatividade de funcionários da empresa.

A rotatividade de funcionários, ou **attrition**, é um dos problemas mais caros para qualquer organização. Segundo pesquisas recentes de mercado (como da SHRM e Forbes):

- **Custo Financeiro:** Substituir um funcionário pode custar entre **50% a 200% do seu salário anual**. Isso inclui gastos com rescisão, novos processos seletivos, anúncios de vaga e integração.
- **Custo Médio:** Em 2025/2026, estima-se que o custo médio por saída de um colaborador seja de aproximadamente **R$ 45.000,00** (considerando empresas de médio/grande porte).
- **Perda de Conhecimento:** Quando um talento sai, ele leva consigo o "know-how" e o contexto histórico de projetos, o que atrasa as entregas de toda a equipe.
- **Queda de Moral:** Equipes com alta rotatividade tendem a ficar sobrecarregadas e desmotivadas, gerando um efeito dominó de novas saídas.

# PESQUISA DE MERCADO PARA VER O CUSTO DE PERDA DE FUNCIONÁRIOS

Position Type / Level	Typical Replacement Cost (% of Salary)	Estimated Cost Example
Entry-Level / Frontline	30% – 50%	$15,000 – $25,000 (on a $50k salary)
Mid-Level Professional	50% – 150%	$40,000 – $120,000 (on a $80k salary)
Specialized Technical / Supervisor	100% – 150%	$120,000 – $180,000 (on a $120k salary)
Executive / C-Suite	200% – 400%	$500,000+ (on a $250k salary)

Vamos considerar:

Funcionário junior: 40% do salário anual
Funcionário Mid-level: 100% do salário anual
Funcionário senior: 150% do salário anual
Funcionário Staff/gerentes: 200% do salário anual

# Variável que mais afeta o problema de negócio

Trata-se de um problema de classificação (Funcionário vai sair ou nao vai sair).

Nesse caso, vamos considerar como variável mais impactante a recall, pois a perda financeir é maior quando o modelo erra, ou seja, diz que o funcionário nao vai sair, sendo que ele está em turnover.
