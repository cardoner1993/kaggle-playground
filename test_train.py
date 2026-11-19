import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from train import run


def test_run_binary(tmp_path):
    X, y = make_classification(n_samples=3000, n_features=8, random_state=0)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["cat"] = np.where(df.f0 > 0, "a", "b")
    df.insert(0, "id", range(len(df)))
    df["label"] = y
    df.iloc[:2000].to_csv(tmp_path / "train.csv", index=False)
    df.iloc[2000:].drop(columns="label").to_csv(tmp_path / "test.csv", index=False)
    df.iloc[2000:][["id", "label"]].assign(label=0.5).to_csv(tmp_path / "sample_submission.csv", index=False)

    score, sub = run(tmp_path, folds=3)

    assert score > 0.85
    assert list(sub.columns) == ["id", "label"] and len(sub) == 1000
    assert sub.label.between(0, 1).all()
