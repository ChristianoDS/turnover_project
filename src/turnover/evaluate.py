"""Métricas de avaliação e escolha de limiar.

Migra o `ml_metrics` dos notebooks para o pacote e adiciona a escolha do melhor
threshold via curva precision-recall (seção de Best Threshold do Ciclo 03).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as mt


def ml_metrics(
    model_name: str,
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Calcula Accuracy, F1, Precision, Recall e AUC em uma linha de DataFrame."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return pd.DataFrame(
        {
            "Model Name": model_name,
            "Accuracy": mt.accuracy_score(y_true, y_pred),
            "F1-Score": mt.f1_score(y_true, y_pred, zero_division=0),
            "Precision": mt.precision_score(y_true, y_pred, zero_division=0),
            "Recall": mt.recall_score(y_true, y_pred, zero_division=0),
            "AUC": mt.roc_auc_score(y_true, y_prob),
        },
        index=[0],
    )


def sweep_thresholds(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
    thresholds: tuple[float, ...] | list[float] | None = None,
) -> pd.DataFrame:
    """Compara métricas para vários limiares, uma linha por threshold.

    Útil para enxergar o trade-off recall x precision e escolher o corte que
    atende ao objetivo de negócio. Se `thresholds` for None, usa
    config.THRESHOLDS_TO_TEST.
    """
    from turnover.config import THRESHOLDS_TO_TEST

    thresholds = thresholds if thresholds is not None else THRESHOLDS_TO_TEST
    linhas = [
        ml_metrics(f"threshold={t:.2f}", y_true, y_prob, threshold=t) for t in thresholds
    ]
    return pd.concat(linhas, ignore_index=True)


def best_threshold_by_f1(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
) -> tuple[float, float]:
    """Retorna (melhor_threshold, melhor_f1) varrendo a curva precision-recall.

    Escolhe o limiar que maximiza o F1 (equilíbrio precision/recall). Para
    priorizar recall, troque o critério por um recall-alvo com precision mínima.
    """
    precision, recall, thresholds = mt.precision_recall_curve(y_true, y_prob)
    # precision/recall têm 1 elemento a mais que thresholds; alinhamos o corte.
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    f1 = f1[:-1]  # descarta o ponto sem threshold correspondente
    if len(thresholds) == 0:
        return 0.5, 0.0
    melhor_idx = int(np.argmax(f1))
    return float(thresholds[melhor_idx]), float(f1[melhor_idx])
