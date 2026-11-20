# Features and RealMLP settings from kaggle.com/code/yekenot/ps-s6-e11-realmlp-pytabkit
import os
import subprocess
import sys
from pathlib import Path

SMOKE_ROWS = int(os.environ.get("SMOKE_ROWS", 0))  # local smoke test: tiny sample, tiny ensemble
if not SMOKE_ROWS:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytabkit"], check=True)

import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

SEED, FOLDS = 42, 5
MODEL_SEEDS = [42, 7, 123]
np.random.seed(SEED)
torch.manual_seed(SEED)

data = next(Path(os.environ.get("INPUT_DIR", "/kaggle/input")).rglob("train.csv")).parent
train, test = (pd.read_csv(data / f, nrows=SMOKE_ROWS or None) for f in ("train.csv", "test.csv"))
X, y = train.drop(columns=["id", "is_fraud"]), train["is_fraud"]
X_test = test.drop(columns=["id"])

num_cols = X.select_dtypes("number").columns.tolist()
DECIMALS = ["amount_usd", "hours_since_last_txn", "velocity_score", "distance_from_home_km"]
COMBOS = [("merchant_category", "card_type", "txn_count_last_24h"), ("merchant_category", "customer_age")]
BINS = {"merchant_risk_score": [7], "hours_since_last_txn": [500, 1000]}


def as_codes(train_vals, apply_vals):
    uniques = pd.Index(train_vals.unique())
    return pd.Categorical(uniques.get_indexer(apply_vals))


def features(fit_df, df):
    out = df.copy()
    for c in DECIMALS:
        out[f"_{c}_decimal"] = (df[c] % 1).round(2).astype("float32")
    for c in num_cols:
        out[f"{c}_cat_"] = as_codes(np.floor(fit_df[c]), np.floor(df[c]))
    for c in DECIMALS:
        out[f"{c}_decimal_cat_"] = as_codes((fit_df[c] % 1).round(2), (df[c] % 1).round(2))
    for c, sizes in BINS.items():
        for n in sizes:
            kb = KBinsDiscretizer(n_bins=n, encode="ordinal", strategy="quantile", subsample=None).fit(fit_df[[c]])
            out[f"{c}_{n}_bin_"] = pd.Categorical(kb.transform(df[[c]]).ravel().astype("int32"))
    for cols in COMBOS:
        key = lambda d: d[list(cols)].astype(str).agg("_".join, axis=1)
        out["_".join(cols) + "_"] = as_codes(key(fit_df), key(df))
    return out


# category codes and bins are unsupervised, so fitting them on all train rows does not leak labels
X_feat, X_test_feat = features(X, X), features(X, X_test)
combo_cols = ["_".join(c) + "_" for c in COMBOS] + ["hours_since_last_txn"]

params = {
    "random_state": SEED, "verbosity": 1, "val_metric_name": "1-auc_ovr",
    "n_ens": 16, "n_epochs": 3, "batch_size": 288, "use_early_stopping": False,
    "lr": 0.073, "wd": 0.038, "sq_mom": 0.988, "lr_sched": "flat_anneal", "wd_sched": "cos_log_15",
    "first_layer_lr_factor": 0.25, "embedding_size": 5, "max_one_hot_cat_size": 16,
    "hidden_sizes": [512, 256, 128], "act": "silu", "p_drop": 0.05, "p_drop_sched": "pow5",
    "plr_hidden_1": 16, "plr_hidden_2": 8, "plr_act_name": "gelu", "plr_lr_factor": 0.1151, "plr_sigma": 0.75,
    "ls_eps": 0.01, "ls_eps_sched": "sqrt_cos", "add_front_scale": False,
    "bias_init_mode": "neg-uniform-dynamic-2",
    "tfms": ["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"],
}
if SMOKE_ROWS:
    params |= {"n_ens": 2, "n_epochs": 1, "device": "cpu"}

oof, fold_ids, preds = np.zeros(len(X)), np.zeros(len(X), dtype=int), np.zeros(len(X_test))
for fold, (tr, va) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(X, y)):
    fold_ids[va] = fold
    # validation for epoch selection is a slice of the training rows, never the scored fold
    fit, es = train_test_split(tr, test_size=0.1, random_state=SEED, stratify=y.iloc[tr])
    parts = {k: X_feat.iloc[idx].copy() for k, idx in (("fit", fit), ("es", es), ("va", va))}
    parts["test"] = X_test_feat.copy()

    enc = TargetEncoder(cv=FOLDS, smooth="auto", shuffle=True, random_state=SEED)
    te_names = [f"_{c}TE" for c in combo_cols]
    parts["fit"][te_names] = enc.fit_transform(parts["fit"][combo_cols], y.iloc[fit])
    for k in ("es", "va", "test"):
        parts[k][te_names] = enc.transform(parts[k][combo_cols])

    # folds stay on SEED; only the network initialisation varies across MODEL_SEEDS
    for s in MODEL_SEEDS:
        model = RealMLP_TD_Classifier(**params | {"random_state": s})
        model.fit(parts["fit"], y.iloc[fit], parts["es"], y.iloc[es])
        oof[va] += model.predict_proba(parts["va"])[:, 1] / len(MODEL_SEEDS)
        preds += model.predict_proba(parts["test"])[:, 1] / FOLDS / len(MODEL_SEEDS)
        torch.cuda.empty_cache()
    print(f"fold {fold} AUC {roc_auc_score(y.iloc[va], oof[va]):.5f}", flush=True)

print(f"OOF AUC {roc_auc_score(y, oof):.5f}")
pd.DataFrame({"id": train["id"], "fold": fold_ids, "pred": oof}).to_csv("oof.csv", index=False)
pd.DataFrame({"id": test["id"], "pred": preds}).to_csv("test.csv", index=False)
