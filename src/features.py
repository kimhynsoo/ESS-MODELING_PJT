"""피처 후보 정의와 충전 정책 단위 데이터 분할."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

BASE = ["dq_log_var"]  # ΔQ 로그분산
# 충전 조건: 충전 전류, 또는 같은 정보를 담은 첫 C-rate(대체안)
CHARGE = {"충전 전류": "charge_current_p90_c10", "첫 C-rate": "first_c_rate"}
# 그 밖의 후보: 온도·저항, 초기 용량 변화
EXTRA = {"평균온도": "Tavg_early", "내부저항": "IR_early", "초기 용량 변화": "qd_delta_100_10"}

# 02 노트북의 단계별 CV 결과로 확정한 입력
FEATURES = ["dq_log_var"]


def load(path=ROOT / "data/processed/cells.csv"):
    """수명 라벨이 있는 셀만 읽고 role(train/valid/test/batch3)과 cv_fold를 붙인다."""
    df = pd.read_csv(path)
    df = df[df.life.notna()].reset_index(drop=True)
    return split(df)


def split(df):
    # 같은 정책 셀이 학습·검증에 나뉘지 않도록 정책 단위로 나눈다.
    # 정책 평균 수명 순으로 5구간을 만들어 구간마다 정책 1개를 Hold-out으로 뽑는다.
    b1 = df[df.batch == "Batch 1"]
    policy_life = b1.groupby("policy").life.mean().sort_values()
    rng = np.random.default_rng(SEED)
    holdout = {str(rng.choice(g)) for g in np.array_split(policy_life.index.to_numpy(), 5)}
    folds = np.array_split(rng.permutation(sorted(set(policy_life.index) - holdout)), 4)
    fold_of = {p: i for i, group in enumerate(folds) for p in group}

    df = df.copy()
    df["role"] = df.batch.map({"Batch 2": "test", "Batch 3": "batch3"})
    is_b1 = df.batch == "Batch 1"
    df.loc[is_b1, "role"] = np.where(df.loc[is_b1, "policy"].isin(holdout), "valid", "train")
    df["cv_fold"] = df.policy.map(fold_of).where(df.role == "train", -1).astype(int)
    return df
