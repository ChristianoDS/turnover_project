"""App Streamlit: upload de CSV e exibição do Score de Rotatividade.

Entrypoint usado tanto localmente (streamlit run app.py) quanto pelo
Streamlit Community Cloud (campo "Main file path": app.py).
"""

from __future__ import annotations

import streamlit as st

from turnover.data import DataContractError, validate
from turnover.predict import SCORE_COLUMN, score

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    st.stop()


st.set_page_config(page_title="Score de Rotatividade", page_icon="📉", layout="wide")

st.title("📉 Score de Rotatividade de Funcionários")
st.caption(
    "Faça upload de um arquivo .csv com os dados dos colaboradores para obter "
    "o score de probabilidade de turnover."
)

uploaded = st.file_uploader("Arquivo CSV", type=["csv"])

if uploaded is None:
    st.info("Aguardando o upload de um arquivo .csv.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
    df = validate(df)
except DataContractError as exc:
    st.error(f"Arquivo inválido: {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Não foi possível ler o CSV: {exc}")
    st.stop()

scored = score(df)

st.success(f"{len(scored)} registros processados.")
st.dataframe(scored, use_container_width=True)

st.download_button(
    "Baixar resultado (.csv)",
    data=scored.to_csv(index=False).encode("utf-8"),
    file_name="score_rotatividade.csv",
    mime="text/csv",
)

col1, col2 = st.columns(2)
col1.metric("Score médio", f"{scored[SCORE_COLUMN].mean():.2%}")
col2.metric("Em risco (score ≥ 50%)", int((scored[SCORE_COLUMN] >= 0.5).sum()))
