# kaggle-playground

Learning repo for the monthly Kaggle Playground Series (tabular). One pipeline reused across episodes; each month adds one MLOps piece.

## Run

- Install: `uv sync` (macOS also needs `brew install libomp` for LightGBM)
- Data: `data/<comp-slug>/{train,test,sample_submission}.csv` (gitignored). Accept the rules on Kaggle first.
- Train: `uv run python train.py playground-series-s6e11` writes `submissions/<comp>/<timestamp>-cv<score>.csv`
- Test: `uv run pytest -q`
- Submit: `uv run kaggle competitions submit -c <comp> -f <file> -m "<msg>"` (needs `KAGGLE_API_TOKEN` or `~/.kaggle/kaggle.json`)

## Conventions

- Id and target columns come from `sample_submission.csv` (first two columns).
- Binary target means AUC, anything else means RMSE. Change `score_fn` when an episode uses another metric.
- Always compare on the CV score, not the public leaderboard (it uses a small slice of test).
