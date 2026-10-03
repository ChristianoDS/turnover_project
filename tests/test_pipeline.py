"""Smoke test do pipeline de predição ponta a ponta."""

import pandas as pd
import pytest

from turnover.data import DataContractError, validate
from turnover.predict import SCORE_COLUMN, score
from turnover.schema import REQUIRED_COLUMNS


def _linha_valida() -> dict:
    base = dict.fromkeys(REQUIRED_COLUMNS, 1)
    base["OverTime"] = "Yes"
    base["JobSatisfaction"] = 1
    base["EnvironmentSatisfaction"] = 1
    return base


def test_validate_rejeita_df_vazio():
    with pytest.raises(DataContractError):
        validate(pd.DataFrame())


def test_validate_rejeita_colunas_ausentes():
    df = pd.DataFrame([{"Age": 30}])
    with pytest.raises(DataContractError):
        validate(df)


def test_score_adiciona_coluna_entre_0_e_1():
    df = pd.DataFrame([_linha_valida()])
    df = validate(df)
    scored = score(df)

    assert SCORE_COLUMN in scored.columns
    assert len(scored) == len(df)
    assert scored[SCORE_COLUMN].between(0.0, 1.0).all()


def test_score_overtime_eleva_risco():
    low = _linha_valida()
    low["OverTime"] = "No"
    low["JobSatisfaction"] = 4
    low["EnvironmentSatisfaction"] = 4

    high = _linha_valida()

    df = pd.DataFrame([low, high])
    scored = score(df)
    assert scored[SCORE_COLUMN].iloc[1] > scored[SCORE_COLUMN].iloc[0]
