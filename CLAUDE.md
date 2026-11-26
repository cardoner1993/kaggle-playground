# kaggle-playground

Learning repo for the monthly Kaggle Playground Series (tabular). One pipeline reused across episodes; each month adds one MLOps piece.

## Run

- Install: `uv sync` (macOS also needs `brew install libomp` for LightGBM)
- Data: `data/<comp-slug>/{train,test,sample_submission}.csv` (gitignored). Accept the rules on Kaggle first.
- Episode config: `episodes/<episode>.toml` holds comp, decimals (optional, auto-detected as float columns with at most 2 decimals), shared LightGBM `[params]`, one `[models.<name>]` per local model, `[kernels.<name>]` per Kaggle GPU model and the `stack` member list. It is the source of truth for what gets submitted.
- New episode: `uv run kaggle competitions download <comp> -p data/<comp>` and unzip, `cp episodes/s6e11.toml episodes/<episode>.toml`, set `comp`, drop `decimals`, and set `[kernels.realmlp.features]` (combos, bins, te) after EDA.
- Pipeline: `uv run python pipeline.py episodes/s6e11.toml [--gpu] [--force]` trains every model whose `preds/<comp>/<model>/oof.csv` is missing, pushes missing kernels only with `--gpu`, stacks, and prints the submit command (it never submits). `--force` retrains everything.
- Train one model: `uv run python train.py episodes/s6e11.toml [model ...]` writes `preds/<comp>/<model>/{oof,test}.csv` and `submissions/<comp>/<model>-<timestamp>-cv<score>.csv`
- Feature tests: `uv run python experiments.py episodes/s6e11.toml [variant ...]`
- Tune: `uv run python tune.py episodes/s6e11.toml <trials> <study-name> [model]`, then paste the printed `[params]` into the config
- Stack: `uv run python stack.py <comp> <member> ...` (members are folders in `preds/<comp>/`)
- Public OOF banks (S6E11, same folds): `uv run kaggle datasets download <owner>/<bank> -p data/banks/<bank> --unzip` for each bank listed in `banks.py`, then `uv run python banks.py` writes `preds/<comp>/bank_<name>/` after checking their folds match ours. masayakawamata's GPT-2 is CC BY 4.0: credit it in anything published
- Kaggle GPU models: script + `kernel-metadata.json` template in `kaggle/<model>/`. Episode features come from `[kernels.<name>.features]`; the pipeline prepends them as `FEATURES = {...}` to a copy in `build/<episode>/<name>/`, writes the metadata (`<owner>/<episode>-<name>-oof`) there and pushes from there. `pipeline.py <config> --build` only writes the copies; smoke-test one from an empty dir with `SMOKE_ROWS=3000 INPUT_DIR=<repo>/data/<comp> uv run --project <repo> --with pytabkit python <repo>/build/<episode>/realmlp/realmlp.py`.
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
