"""후보 모델 CV 비교, 최종 모델 학습, 과제 포맷 성능표 생성."""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURES, ROOT, SEED, load

TARGET_MAPE = 9.1  # 원논문 회귀 성능
TOLERANCE = 0.5  # 최저 CV와 이 차이(%p) 안이면 더 단순한 모델을 택한다
LOG_TARGET = False  # 02 노트북에서 원래 수명 타깃의 CV가 더 낮았다

# 단순한 순서대로. 각 모델은 작은 그리드에서 CV로 하이퍼파라미터를 고른다.
CANDIDATES = {
    "Median": [DummyRegressor(strategy="median")],
    "Ridge": [Ridge(alpha=a) for a in (0.01, 0.1, 1, 10)],
    "ElasticNet": [ElasticNet(alpha=a, l1_ratio=r, max_iter=50_000) for a in (0.001, 0.01, 0.1) for r in (0.2, 0.5, 0.8)],
    "RandomForest": [RandomForestRegressor(n_estimators=300, max_depth=d, min_samples_leaf=3, random_state=SEED) for d in (2, 3)],
}


def mape(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return float(np.mean(np.abs(pred - y) / y) * 100)


def exp10(v):
    return 10**v


def make_model(estimator, log_target=LOG_TARGET):
    pipe = make_pipeline(StandardScaler(), clone(estimator))  # 스케일러는 학습 폴드에서만 적합
    if log_target:
        return TransformedTargetRegressor(pipe, func=np.log10, inverse_func=exp10)  # 저장 가능하도록 lambda 대신 함수
    return pipe


def cv_scores(train, cols, model):
    """정책 단위 4-fold. 폴드별 MAPE 리스트를 돌려준다."""
    scores = []
    for k in range(4):
        fit, check = train[train.cv_fold != k], train[train.cv_fold == k]
        scores.append(mape(check.life, clone(model).fit(fit[cols], fit.life).predict(check[cols])))
    return scores


def compare_models(train, cols):
    """모델별로 CV 평균이 가장 낮은 하이퍼파라미터 하나씩 남긴다."""
    rows = []
    for name, grid in CANDIDATES.items():
        for est in grid:
            s = cv_scores(train, cols, make_model(est))
            params = {k: v for k, v in est.get_params().items() if k in ("alpha", "l1_ratio", "max_depth")}
            rows.append({"model": name, "params": params, "cv_mape": np.mean(s), "cv_std": np.std(s), "estimator": est})
    table = pd.DataFrame(rows)
    return table.loc[table.groupby("model", sort=False).cv_mape.idxmin()].reset_index(drop=True)


def choose(best):
    """기준선을 뺀 후보 중 최저 CV에서 TOLERANCE 안에 드는 가장 단순한 모델."""
    models = best[best.model != "Median"]
    return models[models.cv_mape <= models.cv_mape.min() + TOLERANCE].iloc[0]


def evaluate(df, model, cols, cv_mape):
    train = df[df.role == "train"]
    final = clone(model).fit(train[cols], train.life)
    pred = df[df.role != "train"].copy()
    pred["pred"] = final.predict(pred[cols])
    pred["ape"] = (pred.pred - pred.life).abs() / pred.life * 100
    m = {r: mape(g.life, g.pred) for r, g in pred.groupby("role")}
    b3_clean = pred[(pred.role == "batch3") & ~pred.noisy]
    m["batch3_clean"] = mape(b3_clean.life, b3_clean.pred)
    table = pd.DataFrame([
        ("Train (Batch 1 CV)", cv_mape, f"정책 단위 4-fold 평균, {len(train)}셀"),
        ("Valid (Batch 1 Hold-out)", m["valid"], f"학습에 없는 정책, {(pred.role == 'valid').sum()}셀"),
        ("Test (Batch 2)", m["test"], f"{(pred.role == 'test').sum()}셀"),
        ("Gap (Train-Valid)", m["valid"] - cv_mape, "(+) : 과적합 의심"),
        ("Gap (Valid-Test)", m["test"] - m["valid"], "(+) : 배치간 일반화 저하 의심"),
        ("Gap (Target-Test)", m["test"] - TARGET_MAPE, "Target : 원논문 9.1%"),
        ("Test (Batch 3)", m["batch3"], f"{(pred.role == 'batch3').sum()}셀"),
        ("Gap (Batch2-Batch3)", m["test"] - m["batch3"], "Test 성능 간 비교"),
        ("Gap (Target-Test)", m["batch3"] - TARGET_MAPE, "Batch 3 기준, 원논문 성능 비교"),
        ("Test (Batch 3, 노이즈 4셀 제외)", m["batch3_clean"], f"{len(b3_clean)}셀, 원논문 제거 채널 제외"),
    ], columns=["구분", "MAPE (%)", "비고"])
    return table, pred, final


def run(cols=FEATURES):
    df = load()
    train = df[df.role == "train"]
    best = compare_models(train, cols)
    chosen = choose(best)
    table, pred, final = evaluate(df, make_model(chosen.estimator), cols, chosen.cv_mape)
    return df, best, chosen, table, pred, final


def main():
    _, best, chosen, table, pred, final = run()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    table.round(2).to_csv(out / "model_performance.csv", index=False, encoding="utf-8-sig")
    pred[["batch", "cell", "policy", "structure", "noisy", "role", "life", "pred", "ape"]].round(2).to_csv(
        out / "predictions.csv", index=False)
    joblib.dump({"model": final, "features": FEATURES}, out / "model.joblib")  # Train 37셀로 학습한 최종 모델
    print(best.drop(columns="estimator").round(2).to_string(index=False))
    print("chosen:", chosen.model, chosen.params)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
