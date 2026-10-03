"""CLI simples: gera o score de rotatividade a partir de um CSV.

Uso:
    uv run main.py data/HR-Employee-Attrition-balanced.csv
"""

from __future__ import annotations

import sys

from turnover.data import DataContractError, load_csv, validate
from turnover.predict import score


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("Uso: main.py <caminho_do_csv>")
        return 1

    try:
        df = validate(load_csv(argv[0]))
    except (FileNotFoundError, DataContractError) as exc:
        print(f"Erro: {exc}")
        return 1

    scored = score(df)
    print(scored.head().to_string(index=False))
    print(f"\n{len(scored)} registros processados.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
