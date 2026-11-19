import json
import sys
from pathlib import Path

import optuna

from experiments import digit_keys
from train import PARAMS, run

comp = sys.argv[1] if len(sys.argv) > 1 else "playground-series-s6e11"
trials = int(sys.argv[2]) if len(sys.argv) > 2 else 40
study_name = sys.argv[3] if len(sys.argv) > 3 else comp
data = Path("data") / comp


def objective(trial):
    params = PARAMS | {
        "num_leaves": trial.suggest_int("num_leaves", 2, 32, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 500, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 30, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
    }
    # ponytail: digit_keys is S6E11-specific; make keys per-episode config next month
    return run(data, folds=3, params=params, keys=digit_keys)[0]


study = optuna.create_study(
    study_name=study_name, storage="sqlite:///optuna.db", load_if_exists=True,
    direction="maximize",
)
if len(sys.argv) > 4 and not study.trials:
    start = json.loads(Path(sys.argv[4]).read_text())
    tuned = ["num_leaves", "min_child_samples", "reg_lambda", "reg_alpha", "colsample_bytree", "subsample"]
    study.enqueue_trial({k: start[k] for k in tuned})
study.optimize(objective, n_trials=trials)
print(f"best 3-fold CV {study.best_value:.5f}")

best = PARAMS | study.best_params | {"subsample_freq": 1, "learning_rate": 0.02, "n_estimators": 20000}
score, submission = run(data, params=best, keys=digit_keys)
out = Path("submissions") / comp / f"tuned-cv{score:.5f}.csv"
submission.to_csv(out, index=False)
Path(out.with_suffix(".json")).write_text(json.dumps(best, indent=2))
print(f"final 5-fold CV {score:.5f} -> {out}")
