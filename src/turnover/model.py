"""Wrapper do modelo final de turnover (Logistic Regression).

Encapsula a LogisticRegression com os melhores hiperparâmetros do Ciclo 03 e
class_weight='balanced' (tratamento do desbalanceamento). Expõe fit,
predict_proba, predict e persistência via joblib.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from turnover.config import BEST_PARAMS, BEST_THRESHOLD, SEED


class ModelWrapper:
    """Modelo de classificação de turnover baseado em Logistic Regression."""

    def __init__(self, params: dict | None = None) -> None:
        self.params = {**BEST_PARAMS} if params is None else dict(params)
        self.model = LogisticRegression(
            **self.params,
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
        )
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ModelWrapper:
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidade da classe positiva (saiu = 1)."""
        self._check_fitted()
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = BEST_THRESHOLD) -> np.ndarray:
        """Classe prevista no limiar informado (default: BEST_THRESHOLD)."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("ModelWrapper usado antes de fit.")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> ModelWrapper:
        # Carregar artefato serializado executa código na desserialização.
        # Carregue APENAS artefatos gerados pelo próprio projeto (confiáveis),
        # nunca arquivos enviados por usuários.
        return joblib.load(path)
