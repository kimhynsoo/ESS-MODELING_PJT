"""원본 .mat(약 8GB, HDF5)에서 셀 단위 피처와 EDA용 곡선을 한 번 추출해 CSV로 저장한다."""
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"
FILES = {
    "Batch 1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch 2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch 3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
# 원논문 저자 코드에서 측정 노이즈로 제거한 Batch 3 채널(0-based)
NOISY_B3 = {2, 23, 32, 37, 42, 43}


def arr(f, ref):
    return np.asarray(f[ref]).ravel().astype(float)


def text(f, ref):
    return "".join(chr(int(c)) for c in arr(f, ref))


def cell_rows(f, i, name):
    b = f["batch"]
    cell = f"{name[-1]}-{i + 1:02d}"
    policy = text(f, b["policy_readable"][i, 0])
    row = {"batch": name, "cell": cell, "policy": policy, "life": arr(f, b["cycle_life"][i, 0])[0],
           "structure": "newstructure" if "newstructure" in policy else "original",
           "noisy": name == "Batch 3" and i in NOISY_B3}
    s = f[b["summary"][i, 0]]
    s = {k: np.asarray(s[k]).ravel().astype(float) for k in ("cycle", "QDischarge", "IR", "Tavg", "Tmax", "chargetime")}
    cyc, qd = s["cycle"], s["QDischarge"]
    c = f[b["cycles"][i, 0]]
    k = {n: int(np.flatnonzero(cyc == n)[0]) for n in (10, 11, 100, 101)}

    # 핵심 피처: 같은 전압 격자(3.5→2.0V, 1000점)에서 Q100(V) - Q10(V)
    voltage = arr(f, b["Vdlin"][i, 0])
    q10, q100 = arr(f, c["Qdlin"][k[10], 0]), arr(f, c["Qdlin"][k[100], 0])
    dq = q100 - q10
    current = arr(f, c["I"][k[10], 0])
    early = (cyc >= 10) & (cyc <= 100)
    rate = re.search(r"([0-9.]+)C\(([0-9.]+)%\)-([0-9.]+)C", policy)
    row.update({
        "dq_log_var": np.log10(np.var(dq)), "dq_min": dq.min(), "dq_mean": dq.mean(),
        "qd_delta_100_10": qd[k[100]] - qd[k[10]],
        "charge_current_p90_c10": np.quantile(current[current > 0.1], 0.9),
        **{f"{key}_early": np.nanmedian(s[key][early]) for key in ("IR", "Tavg", "Tmax", "chargetime")},
        "first_c_rate": float(rate[1]) if rate else np.nan,
        "switch_soc_pct": float(rate[2]) if rate else np.nan,
        "second_c_rate": float(rate[3]) if rate else np.nan,
        # 배치 간 Qdlin 비교 점검용(모델 입력 아님)
        "first_cycle_qd": qd[0],
        "cycle_contiguous": bool(np.array_equal(cyc, np.arange(1, len(cyc) + 1))),
        "dq_log_var_c11_c101": np.log10(np.var(arr(f, c["Qdlin"][k[101], 0]) - arr(f, c["Qdlin"][k[11], 0]))),
        "q10_at_3v": np.interp(3.0, voltage[::-1], q10[::-1]),
        "charge_start_v_c10": arr(f, c["V"][k[10], 0])[0],
        "charge_start_i_c10": current[0],
    })
    traj = pd.DataFrame({"cell": cell, "cycle": cyc, "qd": qd})
    curve = pd.DataFrame({"cell": cell, "voltage": voltage, "q10": q10, "q100": q100}).iloc[::10]
    return row, traj, curve


def main():
    rows, trajs, curves = [], [], []
    for name, filename in FILES.items():
        print("reading", name, flush=True)
        with h5py.File(RAW / filename, "r") as f:
            for i in range(f["batch"]["cycle_life"].shape[0]):
                row, traj, curve = cell_rows(f, i, name)
                rows.append(row)
                trajs.append(traj)
                curves.append(curve)
    OUT.mkdir(parents=True, exist_ok=True)
    cells = pd.DataFrame(rows)
    cells.to_csv(OUT / "cells.csv", index=False)
    pd.concat(trajs).to_csv(OUT / "capacity.csv.gz", index=False)
    pd.concat(curves).to_csv(OUT / "qv_curves.csv.gz", index=False)
    print(cells.groupby("batch").agg(cells=("cell", "size"), labeled=("life", "count"), noisy=("noisy", "sum")))


if __name__ == "__main__":
    main()
