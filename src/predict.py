"""저장된 최종 모델(results/model.joblib)로 셀 수명을 예측한다.

입력 CSV에는 `cell`과 100회차까지의 피처 열(`dq_log_var`)이 있어야 한다.
    python -m src.predict                     # data/processed/cells.csv 전체 셀
    python -m src.predict new_cells.csv       # 새 셀
"""
import sys

import joblib
import pandas as pd

from src.features import ROOT

MODEL = ROOT / "results/model.joblib"


def predict(cells, path=MODEL):
    saved = joblib.load(path)
    return pd.Series(saved["model"].predict(cells[saved["features"]]), index=cells.index, name="pred_life")


if __name__ == "__main__":
    cells = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data/processed/cells.csv")
    print(cells[["cell"]].assign(pred_life=predict(cells).round()).to_string(index=False))
