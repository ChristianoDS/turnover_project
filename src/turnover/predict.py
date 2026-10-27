"""Geração do score de rotatividade.

Carrega o pipeline treinado (artefato .joblib gerado pelo train.py) e expõe
`score(df) -> df com coluna de probabilidade`, mantendo a interface que o app
Streamlit, a esteira e os testes já consomem.

Se o artefato ainda não existir (modelo não treinado), cai em um fallback
heurístico transparente — assim o app funciona ponta a ponta mesmo antes do
primeiro treino, sem quebrar.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from turnover.config import PIPELINE_PATH

SCORE_COLUMN = "Score_Rotatividade"


@lru_cache(maxsize=1)
def _load_pipeline(path_str: str):
    """Carrega e memoiza o pipeline treinado.

    A desserialização via joblib executa código: carregamos APENAS o artefato
    gerado pelo próprio projeto (caminho fixo em config.PIPELINE_PATH), nunca
    um arquivo fornecido por usuário. Por isso o parâmetro não é exposto na
    API pública `score`.
    """
    from turnover.pipeline import TurnoverPipeline

    return TurnoverPipeline.load(path_str)


def _score_heuristico(df: pd.DataFrame) -> pd.Series:
    """Fallback determinístico usado quando o modelo ainda não foi treinado."""
    base = 0.1
    prob = pd.Series(base, index=df.index, dtype="float64")
    if "OverTime" in df.columns:
        prob += df["OverTime"].astype(str).str.strip().str.lower().eq("yes") * 0.35
    if "JobSatisfaction" in df.columns:
        prob += (pd.to_numeric(df["JobSatisfaction"], errors="coerce") <= 2) * 0.25
    if "EnvironmentSatisfaction" in df.columns:
        prob += (pd.to_numeric(df["EnvironmentSatisfaction"], errors="coerce") <= 2) * 0.2
    return prob.clip(0.0, 1.0)


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona a coluna de score de rotatividade (probabilidade de sair).

    Usa o pipeline treinado se o artefato existir e o DataFrame tiver as
    colunas necessárias para a engenharia de features; caso contrário (artefato
    ausente ou entrada incompleta), aplica o fallback heurístico. A assinatura
    é estável para o app e os testes.
    """
    out = df.copy()
    prob: pd.Series | None = None

    if Path(PIPELINE_PATH).exists():
        try:
            pipeline = _load_pipeline(str(PIPELINE_PATH))
            prob = pd.Series(pipeline.predict_proba(out), index=out.index)
        except (KeyError, ValueError):
            # entrada não tem todas as colunas que o pipeline exige:
            # degrada para o fallback heurístico em vez de estourar erro.
            prob = None

    if prob is None:
        prob = _score_heuristico(out)

    out[SCORE_COLUMN] = prob.clip(0.0, 1.0).round(4)
    return out
