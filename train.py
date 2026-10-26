"""Treino do modelo final de turnover (Ciclo 04).

Fluxo:
    1. Carrega e valida o CSV (contrato de dados).
    2. Split treino/teste e, do treino, separa a validação (mesmos seeds dos
       ciclos anteriores).
    3. Combina treino + validação para o treino final (a validação já cumpriu
       seu papel nos ciclos 1-3).
    4. Ajusta o TurnoverPipeline (feature engineering + pré-processamento + LR).
    5. Avalia no TESTE uma única vez.
    6. Serializa o pipeline treinado com joblib (artefato de produção).

Uso:
    uv run train.py
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from turnover.config import (
    BEST_THRESHOLD,
    DATA_PATH,
    PIPELINE_PATH,
    SEED,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
)
from turnover.data import DataContractError, load_csv, validate
from turnover.evaluate import ml_metrics
from turnover.pipeline import TurnoverPipeline


def main() -> int:
    # 1. Carga + validação (contrato de colunas em schema.py).
    try:
        df = validate(load_csv(DATA_PATH))
    except (FileNotFoundError, DataContractError) as exc:
        print(f"Erro ao carregar os dados: {exc}")
        return 1

    # 2. Split treino/teste e treino/val, estratificado no alvo.
    df_train_full, df_test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=SEED
    )
    df_train, df_val = train_test_split(
        df_train_full,
        test_size=VAL_SIZE,
        stratify=df_train_full[TARGET],
        random_state=SEED,
    )

    # 3. Modelo final: treino + validação combinados (a val já cumpriu o papel
    #    de ajudar nas decisões dos ciclos anteriores).
    df_train_final = pd.concat([df_train, df_val], ignore_index=True)
    print(
        f"Treino final: {len(df_train_final)} linhas "
        f"(treino {len(df_train)} + val {len(df_val)}) | Teste: {len(df_test)} linhas"
    )

    # 4. Ajusta o pipeline inteiro.
    pipeline = TurnoverPipeline().fit(df_train_final)

    # 5. Avalia no TESTE (primeira e única vez).
    y_test = TurnoverPipeline._as_binary_target(df_test[TARGET])
    proba_test = pipeline.predict_proba(df_test)
    report = ml_metrics("Logistic Regression - Teste", y_test, proba_test, BEST_THRESHOLD)
    print("\n=== Desempenho no conjunto de TESTE ===")
    print(report.to_string(index=False))

    # 6. Persiste o pipeline treinado para produção (Streamlit / predict.py).
    pipeline.save(PIPELINE_PATH)
    print(f"\nPipeline salvo em: {PIPELINE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
