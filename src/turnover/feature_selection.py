"""Seleção de features do pipeline de turnover.

A decisão de QUAIS features remover já foi tomada, com rigor, nos Ciclos 02/03
(método filter de variância/baixa discriminação + Boruta). Esta classe não
re-executa esses algoritmos a cada treino (seria lento e menos reprodutível):
ela guarda a LISTA FINAL de colunas a descartar e a aplica de forma consistente
em treino, validação e teste.

A lista vem de `config.DROP_COLUMNS`. A seleção roda APÓS a engenharia de
features (algumas colunas descartadas são features criadas, ex.: AgeGroup,
CommuteImpact) e ANTES do pré-processamento (scaler/encoder).
"""

from __future__ import annotations

import pandas as pd

from turnover.config import DROP_COLUMNS


class FeatureSelector:
    """Remove as colunas decididas na feature selection dos Ciclos 02/03."""

    def __init__(self, drop_columns: list[str] | None = None) -> None:
        # lista fixa (decidida nos ciclos anteriores); pode ser sobrescrita.
        self.drop_columns = list(DROP_COLUMNS if drop_columns is None else drop_columns)
        self.dropped_: list[str] = []
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> FeatureSelector:
        """Registra quais das colunas a descartar existem de fato em X."""
        self.dropped_ = [c for c in self.drop_columns if c in X.columns]
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove as colunas selecionadas. errors='ignore' para robustez."""
        if not self._fitted:
            raise RuntimeError("FeatureSelector.transform chamado antes de fit.")
        return X.drop(columns=self.dropped_, errors="ignore")

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
