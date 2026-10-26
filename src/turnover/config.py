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
BEST_PARAMS: dict = {
    "C": 1.0,
}

# Limiar de decisão escolhido na seção de Best Threshold do Ciclo 03.
BEST_THRESHOLD = 0.5

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
