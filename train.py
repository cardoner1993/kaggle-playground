import argparse
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

PARAMS = {"n_estimators": 5000, "learning_rate": 0.05, "num_leaves": 63, "verbose": -1}


def run(data_dir: Path, folds: int = 5, seed: int = 42, prep=lambda df: df, params=PARAMS):
    train = prep(pd.read_csv(data_dir / "train.csv"))
    test = prep(pd.read_csv(data_dir / "test.csv"))
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    id_col, target = sample.columns[:2]

    features = [c for c in train.columns if c not in (id_col, target)]
    for c in train[features].select_dtypes(["object", "str"]):
        cats = pd.concat([train[c], test[c]]).astype("category").cat.categories
        train[c] = pd.Categorical(train[c], categories=cats)
        test[c] = pd.Categorical(test[c], categories=cats)

    X, y, X_test = train[features], train[target], test[features]
    binary = y.nunique() == 2
    # ponytail: binary->AUC, else RMSE; set the metric per episode when it differs
    if binary:
        model_cls, score_fn = lgb.LGBMClassifier, roc_auc_score
        splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    else:
        model_cls, score_fn = lgb.LGBMRegressor, root_mean_squared_error
        splitter = KFold(folds, shuffle=True, random_state=seed)

    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    for tr, va in splitter.split(X, y):
        model = model_cls(**params, random_state=seed)
        model.fit(
            X.iloc[tr], y.iloc[tr],
            eval_X=X.iloc[va], eval_y=y.iloc[va],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        predict = (lambda d: model.predict_proba(d)[:, 1]) if binary else model.predict
        oof[va] = predict(X.iloc[va])
        preds += predict(X_test) / folds

    score = score_fn(y, oof)
    return score, sample.assign(**{target: preds})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("comp", help="competition slug, e.g. playground-series-s6e11")
    args = parser.parse_args()

    score, submission = run(Path("data") / args.comp)
    out = Path("submissions") / args.comp / f"{datetime.now():%Y%m%d-%H%M}-cv{score:.5f}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out, index=False)
    print(f"CV {score:.5f} -> {out}")
