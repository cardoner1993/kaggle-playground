import json
import sys
from pathlib import Path

import optuna

from train import PARAMS, run

comp = sys.argv[1] if len(sys.argv) > 1 else "playground-series-s6e11"
trials = int(sys.argv[2]) if len(sys.argv) > 2 else 40
data = Path("data") / comp


def objective(trial):
    params = PARAMS | {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 500, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 30, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
    }
    return run(data, folds=3, params=params)[0]


study = optuna.create_study(
    study_name=comp, storage="sqlite:///optuna.db", load_if_exists=True,
    direction="maximize",
)
study.optimize(objective, n_trials=trials)
print(f"best 3-fold CV {study.best_value:.5f}")

best = PARAMS | study.best_params | {"subsample_freq": 1, "learning_rate": 0.02, "n_estimators": 20000}
score, submission = run(data, params=best)
out = Path("submissions") / comp / f"tuned-cv{score:.5f}.csv"
submission.to_csv(out, index=False)
Path(out.with_suffix(".json")).write_text(json.dumps(best, indent=2))
print(f"final 5-fold CV {score:.5f} -> {out}")
