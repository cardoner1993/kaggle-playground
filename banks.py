"""Convert public S6E11 OOF banks (CC0, same folds as ours) into preds/<comp>/bank_<name>/{oof,test}.csv."""
from pathlib import Path

import numpy as np
import pandas as pd

comp = "playground-series-s6e11"
data, banks, preds = Path("data") / comp, Path("data/banks"), Path("preds") / comp
ids = pd.read_csv(data / "train.csv", usecols=["id"]).id
test_ids = pd.read_csv(data / "test.csv", usecols=["id"]).id
ours = pd.read_csv(preds / "lgbm_te" / "oof.csv")

theirs = pd.read_csv(banks / "s6e11-oof-bank" / "folds.csv")
assert theirs.id.equals(ids) and (theirs.fold.to_numpy() == ours.fold.to_numpy()).all(), "bank folds differ from ours"

bank = banks / "s6e11-oof-bank"
man = pd.read_csv(bank / "manifest.csv")
members = {n: (bank / f"oof_{n}.npy", bank / f"test_{n}.npy") for n in man.name[man.core]}
lm, gpt = banks / "s6e11-compact-lm-seed7", banks / "s6e11-target-first-gpt2-seed-2026"
members |= {
    "lm7": (lm / "lm7_oof_5fold.npy", lm / "lm7_test_5fold.npy"),
    "megalm": (lm / "megalm_oof_sha181c.npy", lm / "megalm_test_sha3168.npy"),
    "gpt2_s2026": (gpt / "gpt2_s2026_oof_5fold.npy", gpt / "gpt2_s2026_test_5fold.npy"),
    "gpt2_mk": (banks / "s6e11-gpt2-target-first-oof/oof.npy", banks / "s6e11-gpt2-target-first-oof/test.npy"),
}
mlp = banks / "s6e11-realmlp-member"
members |= {n: (mlp / f"{n}_oof.npy", mlp / f"{n}_test.npy") for n in ["mlp", "mlp4"]}
arrays = {n: (np.load(o, allow_pickle=False), np.load(t, allow_pickle=False)) for n, (o, t) in members.items()}

for bank, cols in {"s6e11-next2060-oof": ["glmboost", "glmboost_pair", "realmlp"],
                   "s6e11-digit-te-shallow-trees-oof": ["glm", "lgb3_s1", "lgb4", "xgb2"]}.items():
    o, t = pd.read_csv(banks / bank / "oof.csv"), pd.read_csv(banks / bank / "test.csv")
    assert (o.fold.to_numpy() == ours.fold.to_numpy()).all() and o.id.equals(ids) and t.id.equals(test_ids), bank
    arrays |= {f"{bank.split('-')[1]}_{c}": (o[c].to_numpy(), t[c].to_numpy()) for c in cols}

for name, (o, t) in arrays.items():
    assert o.shape == (len(ids),) and t.shape == (len(test_ids),), f"{name}: {o.shape} {t.shape}"
    out = preds / f"bank_{name}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "fold": ours.fold, "pred": o}).to_csv(out / "oof.csv", index=False)
    pd.DataFrame({"id": test_ids, "pred": t}).to_csv(out / "test.csv", index=False)
    print(f"bank_{name}")
