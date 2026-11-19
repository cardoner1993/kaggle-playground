# %%
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

comp = sys.argv[1] if len(sys.argv) > 1 else "playground-series-s6e11"
train = pd.read_csv(f"data/{comp}/train.csv")
test = pd.read_csv(f"data/{comp}/test.csv")
id_col, target = pd.read_csv(f"data/{comp}/sample_submission.csv", nrows=0).columns[:2]
features = [c for c in train.columns if c not in (id_col, target)]
cats = list(train[features].select_dtypes(["object", "str", "bool"]).columns)
nums = [c for c in features if c not in cats]
y = train[target]
print(f"train {train.shape}, test {test.shape}, {target} rate {y.mean():.4f}")

# %% fraud rate per category, sorted by spread
rows = []
for c in cats:
    g = train.groupby(c, observed=True)[target].agg(["mean", "size"])
    rows.append((c, g["mean"].max() / max(g["mean"].min(), 1e-9), g.round(4).to_dict("index")))
for c, lift, g in sorted(rows, key=lambda r: -r[1]):
    print(f"\n{c} (max/min rate {lift:.1f}x)")
    for k, v in g.items():
        print(f"  {k!s:<20} rate {v['mean']:.4f}  n {int(v['size'])}")

# %% single-feature AUC: raw value vs value treated as a category (out-of-fold target mean)
folds = np.arange(len(train)) % 5
print("\nfeature                       raw_auc  cat_auc  nunique")
for c in nums:
    raw = roc_auc_score(y, train[c])
    te = np.empty(len(train))
    for f in range(5):
        means = train[folds != f].groupby(c)[target].mean()
        te[folds == f] = train.loc[folds == f, c].map(means).fillna(y.mean())
    as_cat = roc_auc_score(y, te)
    print(f"{c:<28} {max(raw, 1 - raw):.4f}  {as_cat:.4f}  {train[c].nunique():>7}")

# %% adversarial validation: can a model tell train from test? ~0.5 means same distribution
both = pd.concat([train[features], test[features]], ignore_index=True)
for c in both.select_dtypes(["object", "str"]):
    both[c] = both[c].astype("category")
is_test = pd.Series([0] * len(train) + [1] * len(test))
p = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, verbose=-1), both, is_test, cv=3, method="predict_proba")[:, 1]
print(f"\nadversarial AUC {roc_auc_score(is_test, p):.4f}")

# %% gain importance from one full-data model
X = train[features].copy()
for c in X.select_dtypes(["object", "str"]):
    X[c] = X[c].astype("category")
model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, verbose=-1).fit(X, y)
imp = pd.Series(model.booster_.feature_importance("gain"), index=features)
print("\ngain share")
print((imp / imp.sum()).sort_values(ascending=False).round(4).to_string())
