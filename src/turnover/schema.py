"""Contrato de dados (data contract) do CSV de entrada.

Centraliza as colunas esperadas e o alvo, para que app, treino e testes
compartilhem a mesma fonte de verdade sobre o formato dos dados.
"""

from __future__ import annotations

# Coluna alvo do problema de classificação.
TARGET_COLUMN = "Attrition"

# Colunas mínimas que o pipeline espera encontrar no CSV de entrada.
# Baseado no dataset HR-Employee-Attrition.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "Age",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "JobSatisfaction",
    "MonthlyIncome",
    "OverTime",
    "TotalWorkingYears",
    "YearsAtCompany",
)


def missing_columns(columns: list[str]) -> list[str]:
    """Retorna as colunas obrigatórias ausentes na lista informada."""
    present = set(columns)
    return [col for col in REQUIRED_COLUMNS if col not in present]
