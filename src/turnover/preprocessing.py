"""Pré-processamento numérico e categórico do pipeline de turnover.

Classe Preprocessor no padrão fit/transform, encapsulando RobustScaler (para
numéricas, resistente a outliers) e OneHotEncoder (para categóricas). Tudo é
ajustado SÓ no treino e reaplicado no transform; a ordem final das colunas é
guardada para alinhar treino e teste (evita descompasso de features).
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler


class Preprocessor:
    """Escala numéricas e codifica categóricas, preservando o esquema do treino."""

    def __init__(self) -> None:
        self.scaler = RobustScaler()
        self.encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", drop="if_binary"
        )
        self.num_cols_: list[str] = []
        self.cat_cols_: list[str] = []
        self.feature_names_: list[str] = []
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> Preprocessor:
        """Ajusta scaler e encoder apenas no conjunto de treino."""
        self.num_cols_ = X.select_dtypes(include=["number"]).columns.tolist()
        # inclui object, string (Arrow) e category para pegar todas as categóricas
        self.cat_cols_ = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        if self.num_cols_:
            self.scaler.fit(X[self.num_cols_])
        if self.cat_cols_:
            self.encoder.fit(X[self.cat_cols_])

        # guarda a ordem final das colunas produzidas pelo transform
        self.feature_names_ = self._transform(X).columns.tolist()
        self._fitted = True
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform interno (sem reindexar), usado por fit e transform."""
        partes = []

        if self.num_cols_:
            num = pd.DataFrame(
                self.scaler.transform(X[self.num_cols_]),
                columns=self.num_cols_,
                index=X.index,
            )
            partes.append(num)

        if self.cat_cols_:
            cat = pd.DataFrame(
                self.encoder.transform(X[self.cat_cols_]),
                columns=self.encoder.get_feature_names_out(self.cat_cols_),
                index=X.index,
            )
            partes.append(cat)

        if not partes:
            return X.copy()
        return pd.concat(partes, axis=1)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica o pré-processamento e alinha as colunas às vistas no treino."""
        if not self._fitted:
            raise RuntimeError("Preprocessor.transform chamado antes de fit.")

        out = self._transform(X)
        # reindexa pelas colunas do treino: remove colunas extras e
        # cria (preenchendo com 0) colunas vistas no treino e ausentes aqui.
        out = out.reindex(columns=self.feature_names_, fill_value=0)
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
