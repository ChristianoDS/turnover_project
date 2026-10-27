"""Pipeline de ponta a ponta do turnover.

Orquestra FeatureEngineer -> Preprocessor -> ModelWrapper como uma única
unidade com fit/predict. Como as três etapas seguem fit/transform aprendendo
só no treino, aplicar o pipeline no teste reproduz exatamente os mesmos passos,
sem data leakage e sem descompasso de colunas.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from turnover.config import BEST_THRESHOLD, TARGET
from turnover.feature_engineering import FeatureEngineer
from turnover.feature_selection import FeatureSelector
from turnover.model import ModelWrapper
from turnover.preprocessing import Preprocessor


class TurnoverPipeline:
    """Encadeia engenharia de features, pré-processamento e modelo."""

    def __init__(self, model_params: dict | None = None) -> None:
        self.fe = FeatureEngineer()
        self.selector = FeatureSelector()
        self.prep = Preprocessor()
        self.model = ModelWrapper(params=model_params)
        self._fitted = False

    @staticmethod
    def _as_binary_target(y: pd.Series) -> pd.Series:
        """Converte 'Yes'/'No' em 1/0; se já for numérico (0/1), mantém.

        Usa is_numeric_dtype em vez de checar `object`, porque o pandas pode
        ler colunas de texto como dtype 'string'/Arrow (não só 'object').
        """
        if pd.api.types.is_numeric_dtype(y):
            return y.astype(int)
        return (y.astype(str).str.strip().str.lower() == "yes").astype(int)

    def _split_X_y(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        if TARGET in df.columns:
            X = df.drop(columns=[TARGET])
            y = self._as_binary_target(df[TARGET])
            return X, y
        return df.copy(), None

    def fit(self, df_train: pd.DataFrame) -> TurnoverPipeline:
        """Ajusta todas as etapas usando apenas o conjunto de treino."""
        X, y = self._split_X_y(df_train)
        if y is None:
            raise ValueError(f"Coluna alvo '{TARGET}' ausente no DataFrame de treino.")
        X = self.fe.fit_transform(X)
        X = self.selector.fit_transform(X)  # feature selection (Ciclos 02/03)
        X = self.prep.fit_transform(X)
        self.model.fit(X, y)
        self._fitted = True
        return self

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        X, _ = self._split_X_y(df)  # ignora o alvo se vier junto
        X = self.fe.transform(X)
        X = self.selector.transform(X)  # mesma seleção aplicada no fit
        X = self.prep.transform(X)
        return X

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict_proba(self._prepare(df))

    def predict(self, df: pd.DataFrame, threshold: float = BEST_THRESHOLD) -> np.ndarray:
        self._check_fitted()
        return (self.predict_proba(df) >= threshold).astype(int)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("TurnoverPipeline usado antes de fit.")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> TurnoverPipeline:
        # Desserialização executa código: carregue APENAS artefatos gerados
        # pelo próprio projeto (confiáveis), nunca arquivos de terceiros.
        return joblib.load(path)
