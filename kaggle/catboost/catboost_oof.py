import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED, FOLDS = 42, 5
SMOKE_ROWS = int(os.environ.get("SMOKE_ROWS", 0))  # local smoke test: tiny sample, CPU

data = next(Path(os.environ.get("INPUT_DIR", "/kaggle/input")).rglob("train.csv")).parent
train, test = (pd.read_csv(data / f, nrows=SMOKE_ROWS or None) for f in ("train.csv", "test.csv"))
X, y = train.drop(columns=["id", "is_fraud"]), train["is_fraud"]
X_test = test.drop(columns=["id"])
DECIMALS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km",
            "account_balance_usd", "velocity_score", "merchant_risk_score"]


def features(df):
    """Raw columns plus every value and digit key as a categorical string; CatBoost encodes them itself."""
    out = df.copy()
    for c in df.select_dtypes("number"):
        out[f"{c}_v"] = df[c].astype(str)
    for c in DECIMALS:
        cents = (df[c] * 100).round().astype("int64")
        for name, key in (("int10", cents // 1000), ("d1", cents // 10 % 10), ("cents", cents % 100),
                          ("lead", cents.astype(str).str[0])):
            out[f"{c}_{name}"] = key.astype(str)
    for c in out:
        if out[c].dtype == bool or not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype(str)
    return out


X_feat, X_test_feat = features(X), features(X_test)
cat_cols = [c for c in X_feat if not pd.api.types.is_numeric_dtype(X_feat[c])]

params = {
    "iterations": 20000, "learning_rate": 0.05, "depth": 4, "l2_leaf_reg": 3,
    "max_ctr_complexity": 1, "eval_metric": "AUC", "metric_period": 100,
    "od_type": "Iter", "od_wait": 300, "random_seed": SEED, "verbose": 1000,
    "task_type": "CPU" if SMOKE_ROWS else "GPU",
}
if SMOKE_ROWS:
    params |= {"iterations": 200}

oof, fold_ids, preds = np.zeros(len(X)), np.zeros(len(X), dtype=int), np.zeros(len(X_test))
test_pool = Pool(X_test_feat, cat_features=cat_cols)
for fold, (tr, va) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(X, y)):
    fold_ids[va] = fold
    # early stopping watches a slice of the training rows, never the scored fold
    fit, es = train_test_split(tr, test_size=0.1, random_state=SEED, stratify=y.iloc[tr])
    model = CatBoostClassifier(**params)
    model.fit(Pool(X_feat.iloc[fit], y.iloc[fit], cat_features=cat_cols),
              eval_set=Pool(X_feat.iloc[es], y.iloc[es], cat_features=cat_cols), use_best_model=True)
    oof[va] = model.predict_proba(Pool(X_feat.iloc[va], cat_features=cat_cols))[:, 1]
    preds += model.predict_proba(test_pool)[:, 1] / FOLDS
    print(f"fold {fold} AUC {roc_auc_score(y.iloc[va], oof[va]):.5f} iters {model.get_best_iteration()}", flush=True)

print(f"OOF AUC {roc_auc_score(y, oof):.5f}")
pd.DataFrame({"id": train["id"], "fold": fold_ids, "pred": oof}).to_csv("oof.csv", index=False)
pd.DataFrame({"id": test["id"], "pred": preds}).to_csv("test.csv", index=False)
