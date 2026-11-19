import argparse
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PARAMS = {"n_estimators": 5000, "learning_rate": 0.05, "num_leaves": 63, "verbose": -1}


def target_encode(keys_fit, y_fit, keys_apply, folds=5, seed=0, smoothing=20):
    """Out-of-fold rates for keys_fit, full-fit rates for each frame in keys_apply."""
    prior = y_fit.mean()

    def rates(kf, yf, ka):
        out = {}
        for c in kf:
            g = yf.groupby(kf[c].to_numpy()).agg(["sum", "count"])
            rate = (g["sum"] + smoothing * prior) / (g["count"] + smoothing)
            out[f"te_{c}"] = pd.Series(ka[c].to_numpy()).map(rate).fillna(prior).to_numpy()
        return pd.DataFrame(out, index=ka.index)

    oof = pd.DataFrame(0.0, index=keys_fit.index, columns=[f"te_{c}" for c in keys_fit])
    for a, b in KFold(folds, shuffle=True, random_state=seed).split(keys_fit):
        oof.iloc[b] = rates(keys_fit.iloc[a], y_fit.iloc[a], keys_fit.iloc[b]).to_numpy()
    return oof, [rates(keys_fit, y_fit, k) for k in keys_apply]


def glm_offsets(te_fit, y_fit, te_apply):
    """Logistic regression on encoding logits; returns its logit for te_fit and each te_apply frame."""
    logit = lambda te: np.log(te.clip(1e-6, 1 - 1e-6) / (1 - te.clip(1e-6, 1 - 1e-6)))
    glm = make_pipeline(StandardScaler(), LogisticRegression(solver="newton-cholesky", max_iter=200))
    glm.fit(logit(te_fit), y_fit)
    assert glm[-1].n_iter_[0] < 200, "GLM did not converge"
    return [glm.decision_function(logit(te)) for te in (te_fit, *te_apply)]


def run(data_dir: Path, folds: int = 5, seed: int = 42, prep=lambda df: df, params=PARAMS, keys=None, glm=False):
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
    K, K_test = (keys(X), keys(X_test)) if keys else (None, None)
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
        # early stopping watches a slice of the training rows, never the scored fold
        fit, es = train_test_split(tr, test_size=0.1, random_state=seed, stratify=y.iloc[tr] if binary else None)
        X_fit, X_es, X_va, X_te = X.iloc[fit], X.iloc[es], X.iloc[va], X_test
        off_fit = off_es = off_va = off_te = None
        if K is not None:
            te_fit, (te_es, te_va, te_te) = target_encode(
                K.iloc[fit], y.iloc[fit], [K.iloc[es], K.iloc[va], K_test], seed=seed
            )
            if glm:
                off_fit, off_es, off_va, off_te = glm_offsets(te_fit, y.iloc[fit], [te_es, te_va, te_te])
            X_fit, X_es, X_va, X_te = (
                pd.concat([a, b], axis=1) for a, b in
                ((X_fit, te_fit), (X_es, te_es), (X_va, te_va), (X_te, te_te))
            )

        model = model_cls(**params, random_state=seed)
        model.fit(
            X_fit, y.iloc[fit], init_score=off_fit,
            eval_X=X_es, eval_y=y.iloc[es], eval_init_score=None if off_es is None else [off_es],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        if glm:
            # raw_score excludes init_score, so add the GLM logit back before the sigmoid
            predict = lambda d, off: 1 / (1 + np.exp(-(model.predict(d, raw_score=True) + off)))
        elif binary:
            predict = lambda d, off: model.predict_proba(d)[:, 1]
        else:
            predict = lambda d, off: model.predict(d)
        oof[va] = predict(X_va, off_va)
        preds += predict(X_te, off_te) / folds

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
