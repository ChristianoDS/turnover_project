"""Configuração central do pacote: constantes, caminhos e hiperparâmetros.

Fonte única de verdade para seed, alvo, melhores hiperparâmetros (vindos do
fine-tuning do Ciclo 03) e caminhos de dados/artefatos. Nenhum desses valores
vem de input do usuário — são fixos no projeto, o que evita path traversal e
mantém o comportamento determinístico/reprodutível.
"""

from __future__ import annotations

from pathlib import Path

from turnover.schema import TARGET_COLUMN

# Reexporta o alvo para quem importa de config (mantém schema como fonte única).
TARGET = TARGET_COLUMN

# Semente única usada em splits e no modelo, para reprodutibilidade.
SEED = 42

# Proporções dos splits (iguais às usadas nos notebooks dos ciclos 1-3).
TEST_SIZE = 0.20
VAL_SIZE = 0.20

# Melhores hiperparâmetros da Logistic Regression (modelo vencedor no Ciclo 03).
# Ajuste aqui com o best_params_ encontrado no fine-tuning.
# Obs.: a partir do scikit-learn 1.8 o par 'penalty'/'solver' foi sendo
# substituído por 'C'/'l1_ratio'. Mantemos apenas 'C' (regularização L2 é o
# default) para evitar FutureWarning; ajuste se o tuning exigir L1/elasticnet.
#
# Grid Search do Ciclo 03 (StratifiedKFold, scoring=roc_auc):
#   best_params = {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}  (AUC CV 0.8976)
# penalty='l2' e solver='lbfgs' são o default da LogisticRegression, então
# basta fixar C=1.0 — o modelo resultante é idêntico ao vencedor do tuning.
BEST_PARAMS: dict = {
    "C": 1.0,
}

# Limiar de decisão escolhido na seção de Best Threshold do Ciclo 03.
# Altere este valor para mudar o corte padrão usado em train.py / predict.py.
BEST_THRESHOLD = 0.5

# Lista de limiares para explorar o trade-off recall x precision
# (ver evaluate.sweep_thresholds). Ajuste à vontade para testar outros cortes.
THRESHOLDS_TO_TEST: tuple[float, ...] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)

# Colunas descartadas na Feature Selection (Ciclo 02): união do método filter
# (variância quasi-zero + baixa discriminação univariada/bivariada) com as
# variáveis rejeitadas pelo Boruta. Aplicada APÓS a engenharia de features
# (algumas são features criadas) e ANTES do pré-processamento.
DROP_COLUMNS: tuple[str, ...] = (
    # filter + baixa discriminação (drop_vars)
    "DistanceFromHome",
    "Over18",
    "IncomeRelativeToLevel",
    "CommuteImpact",
    "HourlyRate",
    "NumCompaniesWorked",
    "EmployeeNumber",
    "EducationField",
    "MonthlyRate",
    "PercentSalaryHike",
    "BusinessTravel",
    "Gender",
    # rejeitadas pelo Boruta (as ainda não listadas acima)
    "AgeGroup",
    "Department",
    "Education",
    "JobRole",
    "PromotionLag",
    "YearsSinceLastPromotion",
    # colunas constantes (sem variância) identificadas no Ciclo 01/02
    "EmployeeCount",
    "StandardHours",
)

# --- Caminhos resolvidos a partir da raiz do projeto (nunca de input externo) ---


def _project_root(start: Path | None = None) -> Path:
    """Sobe na árvore de diretórios até encontrar a raiz (com pyproject.toml)."""
    start = start if start is not None else Path(__file__).resolve()
    for parent in start.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback defensivo: dois níveis acima de src/turnover/config.py.
    return start.parents[2]


PROJECT_ROOT = _project_root()
DATA_PATH = PROJECT_ROOT / "data" / "HR-Employee-Attrition-balanced.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PIPELINE_PATH = ARTIFACTS_DIR / "turnover_pipeline.joblib"
