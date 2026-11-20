import sys
import tomllib
from pathlib import Path

import mlflow
import optuna

from train import config_run_kwargs, run

config, trials, study_name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cfg = tomllib.loads(Path(config).read_text())
data = Path("data") / cfg["comp"]
kwargs = config_run_kwargs(cfg)
TUNED = ["num_leaves", "min_child_samples", "reg_lambda", "reg_alpha", "colsample_bytree", "subsample"]


def objective(trial):
    params = cfg["params"] | {
        "learning_rate": 0.05, "n_estimators": 5000,
        "num_leaves": trial.suggest_int("num_leaves", 2, 32, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 500, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 30, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }
    return run(data, folds=3, **kwargs | {"params": params, "tags": kwargs["tags"] | {"trial": trial.number}})[0]


study = optuna.create_study(study_name=study_name, storage="sqlite:///optuna.db", load_if_exists=True,
                            direction="maximize")
if not study.trials:
    study.enqueue_trial({k: cfg["params"][k] for k in TUNED})

mlflow.set_experiment(cfg["comp"])
with mlflow.start_run(run_name=f"tune:{study_name}"):
    mlflow.set_tags(kwargs["tags"] | {"study": study_name, "config": config})
    study.optimize(objective, n_trials=trials)
    mlflow.log_metric("best_3fold_cv", study.best_value)
    score, _ = run(data, **kwargs | {"params": cfg["params"] | study.best_params})
    print(f"best 3-fold CV {study.best_value:.5f}, 5-fold CV {score:.5f}")

print("[params]  # paste into", config)
for k, v in (cfg["params"] | study.best_params).items():
    print(f"{k} = {v}")
