"""Geração do score de rotatividade.

Enquanto o modelo de ML definitivo não é treinado e serializado, esta camada
expõe um placeholder determinístico com a MESMA interface que o modelo real
terá (`score(df) -> df com coluna de probabilidade`). Assim o app Streamlit, a
esteira e os testes já funcionam ponta a ponta e o modelo pode ser plugado
depois sem mudar o contrato.
"""

from __future__ import annotations

import pandas as pd

SCORE_COLUMN = "Score_Rotatividade"


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona a coluna de score de rotatividade ao DataFrame.

    Placeholder: usa uma heurística simples e transparente (OverTime +
    baixa satisfação elevam o score). Substituir pela carga do modelo
    treinado (ex.: joblib.load) mantendo esta assinatura.
    """
    out = df.copy()

    base = 0.1
    prob = pd.Series(base, index=out.index, dtype="float64")

    if "OverTime" in out.columns:
        prob += out["OverTime"].astype(str).str.strip().str.lower().eq("yes") * 0.35
    if "JobSatisfaction" in out.columns:
        prob += (pd.to_numeric(out["JobSatisfaction"], errors="coerce") <= 2) * 0.25
    if "EnvironmentSatisfaction" in out.columns:
        prob += (pd.to_numeric(out["EnvironmentSatisfaction"], errors="coerce") <= 2) * 0.2

    out[SCORE_COLUMN] = prob.clip(0.0, 1.0).round(4)
    return out
