"""Testes do contrato de dados."""

from turnover.schema import REQUIRED_COLUMNS, missing_columns


def test_missing_columns_detecta_ausentes():
    cols = list(REQUIRED_COLUMNS[:-1])  # remove uma coluna obrigatória
    ausentes = missing_columns(cols)
    assert ausentes == [REQUIRED_COLUMNS[-1]]


def test_missing_columns_sem_ausentes():
    cols = list(REQUIRED_COLUMNS) + ["ExtraColuna"]
    assert missing_columns(cols) == []
