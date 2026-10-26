"""Engenharia de features do pipeline de turnover.

Classe FeatureEngineer no padrão fit/transform. Separa dois tipos de feature:

- Linha a linha (seguras): dependem só da própria linha, podem ser aplicadas em
  qualquer conjunto sem risco.
- Estatística de grupo: dependem de uma estatística agregada (ex.: mediana de
  renda por JobLevel). Essa estatística é APRENDIDA no fit (apenas no treino) e
  reaplicada no transform, evitando data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-6  # evita divisão por zero em razões


class FeatureEngineer:
    """Cria as features derivadas do dataset de turnover."""

    def __init__(self) -> None:
        self.median_income_by_level_: pd.Series | None = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> FeatureEngineer:
        """Aprende, apenas no treino, as estatísticas de grupo usadas em transform."""
        if "JobLevel" in df.columns and "MonthlyIncome" in df.columns:
            self.median_income_by_level_ = (
                df.groupby("JobLevel")["MonthlyIncome"].median()
            )
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica a engenharia de features. Não altera o DataFrame de entrada."""
        if not self._fitted:
            raise RuntimeError("FeatureEngineer.transform chamado antes de fit.")

        out = df.copy()

        # --- Features linha a linha (seguras) ---
        out["CareerStability"] = out["TotalWorkingYears"] / (out["NumCompaniesWorked"] + 1)
        out["LoyaltyRatio"] = out["YearsAtCompany"] / (out["TotalWorkingYears"] + _EPS)
        out["CommuteImpact"] = out["DistanceFromHome"] / (out["MonthlyIncome"] + _EPS)
        out["ManagerTrust"] = out["YearsWithCurrManager"] / (out["YearsAtCompany"] + _EPS)
        out["TotalSatisfactionScore"] = (
            out["EnvironmentSatisfaction"]
            + out["JobSatisfaction"]
            + out["RelationshipSatisfaction"]
            + out["WorkLifeBalance"]
        )
        out["OverTimeRisk"] = (
            (out["OverTime"].astype(str).str.strip().str.lower() == "yes")
            & (out["JobInvolvement"] < 3)
        ).astype(int)
        out["AgeGroup"] = out["Age"].apply(
            lambda x: "young" if x < 25 else ("adult" if x < 45 else "senior")
        )
        out["Income_Age_Ratio"] = out["MonthlyIncome"] / out["Age"]

        # --- Feature de grupo (usa estatística aprendida no fit) ---
        if self.median_income_by_level_ is not None:
            mediana = out["JobLevel"].map(self.median_income_by_level_)
            # categorias de JobLevel não vistas no treino ficam NaN -> preenche
            # com a mediana global do treino (fallback seguro).
            mediana = mediana.fillna(self.median_income_by_level_.median())
            out["IncomeRelativeToLevel"] = out["MonthlyIncome"] / (mediana + _EPS)

        # Razões podem gerar inf (denominador ~0) ou NaN; normaliza para 0.
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
