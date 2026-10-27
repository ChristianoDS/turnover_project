"""App Streamlit: Simulador de Rotatividade de Funcionários.

Quatro abas:
  1. Simulação Individual  — formulário de um colaborador.
  2. Simulação em Lote      — amostra aleatória do dataset.
  3. Simulação de ROI       — economia/ROI de agir sobre quem tem risco.
  4. Upload CSV             — score em massa de um arquivo enviado.

Backend: carrega o TurnoverPipeline treinado (artefato .joblib do train.py) e
usa predict_proba. As métricas do teste vêm serializadas no próprio pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from turnover.config import BEST_THRESHOLD, DATA_PATH, PIPELINE_PATH, TARGET
from turnover.data import DataContractError, load_csv, validate
from turnover.pipeline import TurnoverPipeline
from turnover.predict import SCORE_COLUMN, score

st.set_page_config(
    page_title="Simulador de Rotatividade",
    page_icon="📉",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Carregamento (cache) do pipeline e do dataset
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipeline() -> TurnoverPipeline | None:
    """Carrega o pipeline treinado. Retorna None se o artefato não existir.

    A desserialização via joblib executa código: carregamos apenas o artefato
    gerado pelo próprio projeto (caminho fixo em config.PIPELINE_PATH).
    """
    if not PIPELINE_PATH.exists():
        return None
    return TurnoverPipeline.load(PIPELINE_PATH)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Carrega e valida o dataset de referência (usado em lote e ROI)."""
    return validate(load_csv(DATA_PATH))


def classify(prob: float, threshold: float) -> tuple[str, str]:
    """Rótulo e tipo de alerta Streamlit conforme o risco."""
    if prob >= threshold:
        return "🔴 Alto risco de saída", "error"
    if prob >= threshold * 0.6:
        return "🟡 Risco moderado", "warning"
    return "🟢 Baixo risco", "success"


# Significado dos níveis de cargo (JobLevel). A faixa salarial é aproximada,
# baseada na mediana por nível observada no dataset.
JOB_LEVELS = {
    0: "Nível 0 — Estágio / Trainee (entrada)",
    1: "Nível 1 — Júnior / Operacional",
    2: "Nível 2 — Pleno",
    3: "Nível 3 — Sênior / Especialista",
    4: "Nível 4 — Liderança / Gerência",
    5: "Nível 5 — Executivo / Diretoria",
}


pipeline = load_pipeline()
df_full = load_dataset()

st.title("📉 Simulador de Rotatividade de Funcionários")

if pipeline is None:
    st.warning(
        "Modelo ainda não treinado. Rode `uv run train.py` para gerar o "
        "artefato antes de usar as simulações."
    )
    st.stop()

METRICS = pipeline.metrics or {}
THRESHOLD = getattr(pipeline, "threshold", BEST_THRESHOLD)

st.caption(
    "Backend via classe `TurnoverPipeline`; modelo carregado do artefato "
    "`turnover_pipeline.joblib`. Métricas referentes ao conjunto de teste."
)

# --- Métricas do modelo (teste) no topo ------------------------------------
if METRICS:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC (teste)", f"{METRICS.get('auc', float('nan')):.3f}")
    c2.metric("Threshold", f"{THRESHOLD:.3f}")
    c3.metric("Recall (teste)", f"{METRICS.get('recall', float('nan')):.1%}")
    c4.metric("Precisão (teste)", f"{METRICS.get('precision', float('nan')):.1%}")

st.divider()

tab_ind, tab_lote, tab_roi, tab_csv = st.tabs(
    ["Simulação Individual", "Simulação em Lote", "Simulação de ROI", "Upload CSV"]
)


# ---------------------------------------------------------------------------
# TAB 1 — Simulação individual
# ---------------------------------------------------------------------------
with tab_ind:
    st.subheader("Preencha os dados do funcionário")

    with st.form("form_individual"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Perfil**")
            age = st.slider("Idade", 18, 60, 35)
            job_level = st.select_slider(
                "Nível do cargo",
                options=list(JOB_LEVELS.keys()),
                value=2,
                format_func=lambda x: JOB_LEVELS[x],
                help="Senioridade do cargo (0 = entrada, 5 = executivo).",
            )
            monthly_income = st.number_input("Salário Mensal (R$)", 1000, 25000, 6500, step=250)
            stock = st.select_slider("Stock Option Level (0-3)", options=[0, 1, 2, 3], value=1)
            with st.expander("O que significam os níveis de cargo?"):
                for _nivel, _desc in JOB_LEVELS.items():
                    st.markdown(f"- {_desc}")

        with col2:
            st.markdown("**Carreira**")
            total_working = st.slider("Anos totais de experiência", 0, 40, 10)
            years_company = st.slider("Anos na empresa", 0, 40, 5)
            years_role = st.slider("Anos no cargo atual", 0, 20, 3)
            years_manager = st.slider("Anos com o gestor atual", 0, 20, 3)

        with col3:
            st.markdown("**Engajamento & Satisfação**")
            overtime = st.radio("Faz hora extra?", ["No", "Yes"], horizontal=True)
            job_involve = st.select_slider("Envolvimento (1-4)", options=[1, 2, 3, 4], value=3)
            env_sat = st.select_slider("Satisfação ambiente (1-4)", options=[1, 2, 3, 4], value=3)
            job_sat = st.select_slider("Satisfação trabalho (1-4)", options=[1, 2, 3, 4], value=3)
            rel_sat = st.select_slider(
                "Satisfação relacionamentos (1-4)", options=[1, 2, 3, 4], value=3
            )
            wlb = st.select_slider("Equilíbrio vida-trabalho (1-4)", options=[1, 2, 3, 4], value=3)

        submitted = st.form_submit_button(
            "Prever Rotatividade", use_container_width=True, type="primary"
        )

    if submitted:
        # Monta a linha a partir de um TEMPLATE do dataset (primeira linha) e
        # sobrescreve apenas os campos coletados. Isso garante que TODAS as
        # colunas exigidas pela engenharia de features estejam presentes,
        # evitando erro por coluna faltante.
        template = df_full.drop(columns=[TARGET]).iloc[[0]].copy()
        overrides = {
            "Age": age,
            "JobLevel": job_level,
            "MonthlyIncome": monthly_income,
            "StockOptionLevel": stock,
            "TotalWorkingYears": total_working,
            "YearsAtCompany": years_company,
            "YearsInCurrentRole": years_role,
            "YearsWithCurrManager": years_manager,
            "OverTime": overtime,
            "JobInvolvement": job_involve,
            "EnvironmentSatisfaction": env_sat,
            "JobSatisfaction": job_sat,
            "RelationshipSatisfaction": rel_sat,
            "WorkLifeBalance": wlb,
        }
        for col, val in overrides.items():
            if col in template.columns:
                template[col] = val

        prob = float(pipeline.predict_proba(template)[0])
        label, kind = classify(prob, THRESHOLD)

        st.divider()
        res1, res2 = st.columns([1, 2])
        with res1:
            getattr(st, kind)(f"### {label}")
            st.metric("Probabilidade de saída", f"{prob:.1%}")
        with res2:
            st.progress(min(prob, 1.0))
            st.caption(f"Decisão baseada no threshold **{THRESHOLD:.3f}**.")
            if prob >= THRESHOLD:
                st.warning(
                    "Recomenda-se **ação de retenção** (conversa com gestor, "
                    "plano de carreira, revisão salarial)."
                )
            else:
                st.info("Perfil com baixo risco de saída no momento.")


# ---------------------------------------------------------------------------
# TAB 2 — Simulação em lote (amostra do dataset)
# ---------------------------------------------------------------------------
with tab_lote:
    st.subheader("Prever para uma amostra aleatória de funcionários")
    n = st.slider("Quantidade de funcionários", 5, 50, 10)

    if st.button("Sortear e prever", type="primary"):
        sample = df_full.sample(n=n, random_state=None).reset_index(drop=True)
        probs = pipeline.predict_proba(sample)

        out = pd.DataFrame(
            {
                "Idade": sample["Age"],
                "Salário": sample["MonthlyIncome"],
                "Estado Civil": sample.get("MaritalStatus", ""),
                "Hora Extra": sample["OverTime"],
                "Anos na Empresa": sample["YearsAtCompany"],
                "Prob. Saída": probs,
                "Previsão": np.where(probs >= THRESHOLD, "🔴 Sai", "🟢 Permanece"),
                "Real (histórico)": sample[TARGET].map({"Yes": "Saiu", "No": "Ficou"}),
            }
        )

        st.dataframe(
            out.style.format(
                {"Prob. Saída": "{:.1%}", "Salário": "R$ {:,.0f}"}
            ).background_gradient(subset=["Prob. Saída"], cmap="RdYlGn_r"),
            use_container_width=True,
            hide_index=True,
        )
        n_risco = int((probs >= THRESHOLD).sum())
        st.caption(f"**{n_risco} de {n}** funcionários sinalizados como risco de saída.")


# ---------------------------------------------------------------------------
# TAB 3 — Simulação de ROI
# ---------------------------------------------------------------------------
with tab_roi:
    st.subheader("Retorno sobre o investimento em retenção")
    st.caption("Estima economia e ROI ao usar o modelo para agir sobre quem tem risco de saída.")

    base = df_full.copy()
    base["Annual_Salary"] = base["MonthlyIncome"] * 12
    base["Replacement_Multiplier"] = base["JobLevel"].apply(
        lambda x: 0.40 if x <= 2 else (1.00 if x == 3 else 2.00)
    )
    base["Replacement_Cost"] = base["Annual_Salary"] * base["Replacement_Multiplier"]
    leaving = base.loc[base[TARGET] == "Yes"]
    leaving_employee = len(leaving)
    avg_replacement_cost = float(leaving["Replacement_Cost"].mean()) if leaving_employee else 0.0

    colr1, colr2 = st.columns(2)
    with colr1:
        recall_default = float(round(METRICS.get("recall", 0.75), 2))
        recall = st.slider("Recall do modelo", 0.10, 1.00, recall_default, 0.05)
        retention_rate = st.slider("Taxa de retenção do RH", 0.10, 1.00, 0.30, 0.05)
    with colr2:
        intervention_cost = st.number_input("Custo por intervenção (R$)", 0, 50000, 5000, step=500)
        maintenance_annual_cost = st.number_input(
            "Custo anual do modelo (R$)", 0, 100000, 13000, step=1000
        )
        production_cost = st.number_input(
            "Custo de produção/infra (R$)", 0, 1000000, 200000, step=10000
        )

    true_positive = int(leaving_employee * recall)
    retention_employee = int(retention_rate * true_positive)
    avg_savings = retention_employee * avg_replacement_cost
    annual_cost = maintenance_annual_cost + retention_employee * intervention_cost + production_cost
    liquid_savings = avg_savings - annual_cost
    roi_annual = avg_savings / annual_cost if annual_cost else 0.0

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Funcionários retidos", f"{retention_employee}")
    m2.metric("Economia bruta", f"R$ {avg_savings:,.0f}")
    m3.metric("Economia líquida", f"R$ {liquid_savings:,.0f}")
    m4.metric("ROI anual", f"{roi_annual:.2f}x")
    st.caption(
        f"Base: {leaving_employee} saídas/ano · custo médio de substituição "
        f"R$ {avg_replacement_cost:,.0f}."
    )


# ---------------------------------------------------------------------------
# TAB 4 — Upload de CSV (score em massa)
# ---------------------------------------------------------------------------
with tab_csv:
    st.subheader("Score de rotatividade a partir de um CSV")
    uploaded = st.file_uploader("Arquivo CSV", type=["csv"])

    if uploaded is None:
        st.info("Envie um arquivo .csv com os dados dos colaboradores.")
    else:
        try:
            df_up = validate(pd.read_csv(uploaded))
        except DataContractError as exc:
            st.error(f"Arquivo inválido: {exc}")
            st.stop()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Não foi possível ler o CSV: {exc}")
            st.stop()

        scored = score(df_up)  # usa o pipeline treinado (via predict.py)
        st.success(f"{len(scored)} registros processados.")
        st.dataframe(scored, use_container_width=True)
        st.download_button(
            "Baixar resultado (.csv)",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="score_rotatividade.csv",
            mime="text/csv",
        )
        colA, colB = st.columns(2)
        colA.metric("Score médio", f"{scored[SCORE_COLUMN].mean():.2%}")
        colB.metric(
            f"Em risco (score ≥ {THRESHOLD:.0%})",
            int((scored[SCORE_COLUMN] >= THRESHOLD).sum()),
        )
