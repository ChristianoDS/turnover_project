from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from attrition_pipeline import AttritionPipeline

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Simulador de Rotatividade v2",
    page_icon="",
    layout="wide",
)

ARTIFACT_PATH = "artifacts/attrition_pipeline.pkl"
DATASET_PATH = "dataset/HR-Employee-Attrition-balanced.csv"
TARGET_RECALL = AttritionPipeline.TARGET_RECALL

# ===========================================================================
# UI
# ===========================================================================
pipeline = load_pipeline()
df_full = load_dataset()
METRICS = pipeline.metrics
THRESHOLD = pipeline.threshold

st.title("Simulador de Rotatividade de Funcionários — v2")
st.caption(
    "Backend em Python via classe `AttritionPipeline`; modelo carregado do "
    "artefato `attrition_pipeline.pkl`."
)

# --- Métricas do modelo no topo -------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("AUC (validação)", f"{METRICS['auc']:.3f}")
c2.metric("Threshold de decisão", f"{THRESHOLD:.3f}")
c3.metric("Recall", f"{METRICS['recall']:.1%}")
c4.metric("Precisão", f"{METRICS['precision']:.1%}")

st.divider()

tab_ind, tab_lote, tab_roi = st.tabs(
    ["Simulação Individual", "Simulação em Lote", "Simulação de ROI"]
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
            education = st.select_slider("Escolaridade (1-5)", options=[1, 2, 3, 4, 5], value=3)
            marital = st.selectbox("Estado Civil", ["Single", "Married", "Divorced"])
            monthly_income = st.number_input("Salário Mensal (R$)", 1000, 25000, 6500, step=250)
            stock = st.select_slider("Stock Option Level (0-3)", options=[0, 1, 2, 3], value=1)

        with col2:
            st.markdown("**Carreira**")
            total_working = st.slider("Anos totais de experiência", 0, 40, 10)
            years_company = st.slider("Anos na empresa", 0, 40, 5)
            years_role = st.slider("Anos no cargo atual", 0, 20, 3)
            years_promo = st.slider("Anos desde última promoção", 0, 15, 1)
            years_manager = st.slider("Anos com o gestor atual", 0, 20, 3)
            num_companies = st.slider("Nº de empresas anteriores", 0, 10, 2)
            distance = st.slider("Distância de casa (km)", 1, 30, 8)

        with col3:
            st.markdown("**Engajamento & Satisfação**")
            overtime = st.radio("Faz hora extra?", ["No", "Yes"], horizontal=True)
            job_involve = st.select_slider("Envolvimento no trabalho (1-4)", options=[1, 2, 3, 4], value=3)
            env_sat = st.select_slider("Satisfação com o ambiente (1-4)", options=[1, 2, 3, 4], value=3)
            job_sat = st.select_slider("Satisfação com o trabalho (1-4)", options=[1, 2, 3, 4], value=3)
            rel_sat = st.select_slider("Satisfação com relacionamentos (1-4)", options=[1, 2, 3, 4], value=3)
            wlb = st.select_slider("Equilíbrio vida-trabalho (1-4)", options=[1, 2, 3, 4], value=3)
            perf = st.select_slider("Avaliação de desempenho (1-4)", options=[1, 2, 3, 4], value=3)
            training = st.slider("Treinamentos no último ano", 0, 6, 2)

        submitted = st.form_submit_button("Prever Rotatividade", use_container_width=True, type="primary")

    if submitted:
        inputs = {
            "Age": age,
            "Education": education,
            "MaritalStatus": marital,
            "MonthlyIncome": monthly_income,
            "StockOptionLevel": stock,
            "TotalWorkingYears": total_working,
            "YearsAtCompany": years_company,
            "YearsInCurrentRole": years_role,
            "YearsSinceLastPromotion": years_promo,
            "YearsWithCurrManager": years_manager,
            "NumCompaniesWorked": num_companies,
            "DistanceFromHome": distance,
            "OverTime": overtime,
            "JobInvolvement": job_involve,
            "EnvironmentSatisfaction": env_sat,
            "JobSatisfaction": job_sat,
            "RelationshipSatisfaction": rel_sat,
            "WorkLifeBalance": wlb,
            "PerformanceRating": perf,
            "TrainingTimesLastYear": training,
        }

        # Backend: a classe aplica limpeza + feature eng. + scale e o modelo
        # carregado do pickle realiza a previsão.
        raw = pd.DataFrame([inputs])
        prob = float(pipeline.predict_proba(raw)[0])
        label, kind = classify(prob, THRESHOLD)

        st.divider()
        res1, res2 = st.columns([1, 2])
        with res1:
            getattr(st, kind)(f"### {label}")
            st.metric("Probabilidade de saída", f"{prob:.1%}")

        with res2:
            st.progress(min(prob, 1.0))
            st.caption(
                f"Decisão baseada no threshold **{THRESHOLD:.3f}** "
                f"(otimizado para Recall ≥ {TARGET_RECALL:.0%})."
            )

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
        sample = df_full.sample(n=n).reset_index(drop=True)
        probs = pipeline.predict_proba(sample)

        out = pd.DataFrame({
            "Idade": sample["Age"],
            "Salário": sample["MonthlyIncome"],
            "Estado Civil": sample["MaritalStatus"],
            "Hora Extra": sample["OverTime"],
            "Anos na Empresa": sample["YearsAtCompany"],
            "Prob. Saída": probs,
            "Previsão": np.where(probs >= THRESHOLD, "🔴 Sai", "🟢 Permanece"),
            "Real (histórico)": sample["Attrition"].map({"Yes": "Saiu", "No": "Ficou"}),
        })

        st.dataframe(
            out.style.format({"Prob. Saída": "{:.1%}", "Salário": "R$ {:,.0f}"})
               .background_gradient(subset=["Prob. Saída"], cmap="RdYlGn_r"),
            use_container_width=True,
            hide_index=True,
        )

        n_risco = int((probs >= THRESHOLD).sum())
        st.caption(f"**{n_risco} de {n}** funcionários sinalizados como risco de saída.")

# ---------------------------------------------------------------------------
# TAB 3 — Simulação de ROI (baseada no notebook)
# ---------------------------------------------------------------------------
with tab_roi:
    st.subheader("Retorno sobre o investimento em retenção")
    st.caption(
        "Estima economia e ROI ao usar o modelo para agir sobre funcionários "
        "com risco de saída."
    )

    # custo médio de substituição a partir do dataset
    base = df_full.copy()
    base["Annual_Salary"] = base["MonthlyIncome"] * 12
    base["Replacement_Multiplier"] = base["JobLevel"].apply(lambda x: 0.40 if x <= 2 else 1.00 if x == 3 else 2.00)
    base["Replacement_Cost"] = base["Annual_Salary"] * base["Replacement_Multiplier"]
    leaving = base.loc[base["Attrition"] == "Yes"]
    leaving_employee = len(leaving)
    avg_replacement_cost = float(leaving["Replacement_Cost"].mean())

    colr1, colr2 = st.columns(2)
    with colr1:
        recall = st.slider("Recall do modelo (% de quem sai que é identificado)", 0.10, 1.00, float(round(METRICS["recall"], 2)), 0.05)
        retention_rate = st.slider("Taxa de retenção do RH (% dos identificados que são mantidos)", 0.10, 1.00, 0.30, 0.05)

    with colr2:
        intervention_cost = st.number_input("Custo por intervenção (R$)", 0, 50000, 5000, step=500)
        maintenance_annual_cost = st.number_input("Custo anual de manutenção do modelo (R$)", 0, 100000, 13000, step=1000)
        production_cost = st.number_input("Custo de produção/infra (R$)", 0, 1000000, 200000, step=10000)

    true_positive = int(leaving_employee * recall)
    retention_employee = int(retention_rate * true_positive)
    avg_savings = retention_employee * avg_replacement_cost
    annual_cost = (
        maintenance_annual_cost
        + retention_employee * intervention_cost
        + production_cost
    )

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
