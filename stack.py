import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

comp = sys.argv[1] if len(sys.argv) > 1 else "playground-series-s6e11"
data, pred_dir = Path("data") / comp, Path("preds") / comp
sample = pd.read_csv(data / "sample_submission.csv")
id_col, target = sample.columns[:2]
y = pd.read_csv(data / "train.csv", usecols=[id_col, target]).set_index(id_col)[target]

models = sorted(p.name for p in pred_dir.iterdir() if (p / "oof.csv").exists())
oofs = {m: pd.read_csv(pred_dir / m / "oof.csv").set_index(id_col) for m in models}
tests = {m: pd.read_csv(pred_dir / m / "test.csv").set_index(id_col) for m in models}
folds = oofs[models[0]]["fold"]
for m in models:
    assert oofs[m].index.equals(y.index), f"{m}: oof ids differ from train.csv"
    assert (oofs[m]["fold"] == folds).all(), f"{m}: fold assignment differs"
    assert tests[m].index.equals(sample.set_index(id_col).index), f"{m}: test ids differ"


def logit_rank(s):
    r = s.rank(pct=True).clip(1e-6, 1 - 1e-6)
    return np.log(r / (1 - r))


X = pd.DataFrame({m: logit_rank(oofs[m]["pred"]) for m in models})
X_test = pd.DataFrame({m: logit_rank(tests[m]["pred"]) for m in models})

for m in models:
    print(f"{m:<20} OOF {roc_auc_score(y, oofs[m]['pred']):.5f}")
print("rank correlation\n", X.corr(method="spearman").round(4))

# stacker scored on a different split than the base models' folds
stacker = LogisticRegression(C=0.01, solver="newton-cholesky")
stack_oof = cross_val_predict(stacker, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=7),
                              method="predict_proba")[:, 1]
score = roc_auc_score(y, stack_oof)
stacker.fit(X, y)
print("weights", dict(zip(models, stacker.coef_[0].round(3))))
print(f"stack CV {score:.5f}")

out = Path("submissions") / comp / f"stack-{'+'.join(models)}-cv{score:.5f}.csv"
sample.assign(**{target: stacker.predict_proba(X_test)[:, 1]}).to_csv(out, index=False)
print("->", out)
