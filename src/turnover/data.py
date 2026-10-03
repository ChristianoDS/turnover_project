"""Carregamento e validação dos dados de entrada."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from turnover.schema import missing_columns


class DataContractError(ValueError):
    """Erro levantado quando o CSV de entrada não cumpre o contrato de dados."""


def load_csv(source: str | Path) -> pd.DataFrame:
    """Carrega um CSV em DataFrame."""
    return pd.read_csv(source)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o DataFrame contra o contrato de dados.

    Levanta DataContractError se colunas obrigatórias estiverem ausentes
    ou se o DataFrame estiver vazio.
    """
    if df.empty:
        raise DataContractError("O arquivo de entrada não possui linhas.")

    ausentes = missing_columns(list(df.columns))
    if ausentes:
        raise DataContractError(
            f"Colunas obrigatórias ausentes no CSV: {', '.join(ausentes)}"
        )
    return df
