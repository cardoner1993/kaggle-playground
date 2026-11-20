# kaggle-playground

Learning repo for the monthly Kaggle Playground Series (tabular). One pipeline reused across episodes; each month adds one MLOps piece.

## Run

- Install: `uv sync` (macOS also needs `brew install libomp` for LightGBM)
- Data: `data/<comp-slug>/{train,test,sample_submission}.csv` (gitignored). Accept the rules on Kaggle first.
- Episode config: `episodes/<episode>.toml` (comp, model name, keys, decimals, LightGBM params). It is the source of truth for the best single model.
- Train: `uv run python train.py episodes/s6e11.toml` writes `preds/<comp>/<model>/{oof,test}.csv` and `submissions/<comp>/<model>-<timestamp>-cv<score>.csv`
- Feature tests: `uv run python experiments.py episodes/s6e11.toml [variant ...]`
- Tune: `uv run python tune.py episodes/s6e11.toml <trials> <study-name>`, then paste the printed `[params]` into the config
- Stack: `uv run python stack.py <comp> <member> ...` (members are folders in `preds/<comp>/`)
- Kaggle GPU models: `kaggle/<model>/` + `kernel-metadata.json`; smoke-test locally with `SMOKE_ROWS=5000 INPUT_DIR=data/<comp>`, then `uv run kaggle kernels push -p kaggle/<model>` and `uv run kaggle kernels output <user>/<slug> -p preds/<comp>/<model>`
- Tracking: every `run()` and stack logs to MLflow (`sqlite:///mlflow.db`, one experiment per comp). UI: `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`
- Test: `uv run pytest -q`
- Submit: `uv run kaggle competitions submit <comp> -f <file> -m "<msg>"`

## Conventions

- Id and target columns come from `sample_submission.csv` (first two columns).
- Binary target means AUC, anything else means RMSE. Change `score_fn` when an episode uses another metric.
- Always compare on the CV score, not the public leaderboard (it uses a small slice of test).
- Every model uses `StratifiedKFold(5, shuffle=True, random_state=42)` and writes `id, fold, pred`; `stack.py` refuses members whose ids or folds differ.
- Early stopping and epoch selection use an inner slice of the training fold, never the scored fold.
- Target encodings are out-of-fold; `prep` only sees feature columns, never id or target.
- Commit before a tracked run: MLflow's git commit tag is only meaningful when `git_dirty` is false.
