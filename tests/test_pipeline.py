from src.features import load
from src.train import mape


def test_policy_split_has_no_leakage():
    df = load()
    train, valid = df[df.role == "train"], df[df.role == "valid"]
    assert (len(train), len(valid)) == (37, 9)
    assert set(train.policy).isdisjoint(valid.policy)
    assert train.groupby("policy").cv_fold.nunique().max() == 1
    assert set(train.cv_fold) == {0, 1, 2, 3}


def test_mape():
    assert mape([100, 200], [110, 180]) == 10.0
