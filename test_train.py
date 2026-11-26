import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from train import decimal_cols, run


@pytest.fixture(autouse=True)
def tmp_tracking(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")


def test_run_binary(tmp_path):
    X, y = make_classification(n_samples=3000, n_features=8, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["cat"] = np.where(df.f0 > 0, "a", "b")
    df.insert(0, "id", range(len(df)))
    df["label"] = y
    df.iloc[:2000].to_csv(tmp_path / "train.csv", index=False)
    df.iloc[2000:].drop(columns="label").to_csv(tmp_path / "test.csv", index=False)
    df.iloc[2000:][["id", "label"]].assign(label=0.5).to_csv(tmp_path / "sample_submission.csv", index=False)

    for kwargs in ({}, {"keys": lambda X: X.round(1), "glm": True}):
        score, sub = run(tmp_path, folds=3, **kwargs)

        assert score > 0.85
        assert list(sub.columns) == ["id", "label"] and len(sub) == 1000
        assert sub.label.between(0, 1).all()


def test_target_encoding_does_not_leak(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"id": range(4000), "noise_key": rng.integers(0, 1500, 4000), "f0": rng.normal(size=4000)})
    df["label"] = rng.integers(0, 2, 4000)
    df.iloc[:3000].to_csv(tmp_path / "train.csv", index=False)
    df.iloc[3000:].drop(columns="label").to_csv(tmp_path / "test.csv", index=False)
    df.iloc[3000:][["id", "label"]].to_csv(tmp_path / "sample_submission.csv", index=False)

    score, _ = run(tmp_path, folds=3, keys=lambda X: X[["noise_key"]])

    assert score < 0.56


def test_decimal_cols():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"money": rng.uniform(0, 500, 1000).round(2), "whole": rng.integers(0, 9, 1000).astype(float),
                       "raw": rng.normal(size=1000)})
    assert decimal_cols(df) == ["money"]
