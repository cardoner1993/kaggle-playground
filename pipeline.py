import json
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import mlflow

from train import train

config = Path(sys.argv[1])
force, gpu = "--force" in sys.argv, "--gpu" in sys.argv
cfg = tomllib.loads(config.read_text())
comp, episode = cfg["comp"], config.stem
preds = Path("preds") / comp


def have(model):
    return not force and (preds / model / "oof.csv").exists()


def kaggle(*args):
    return subprocess.run(["kaggle", *args], check=True, capture_output=True, text=True).stdout


def run_kernel(name, script, gpu_minutes):
    script = Path(script)
    meta = json.loads((script.parent / "kernel-metadata.json").read_text())
    owner = meta["id"].split("/")[0]
    # Kaggle derives the slug from the title, so they must agree
    meta |= {"id": f"{owner}/{episode}-{name}-oof", "title": f"{episode} {name} oof",
             "code_file": script.name, "competition_sources": [comp]}
    build = Path("build") / episode / name
    build.mkdir(parents=True, exist_ok=True)
    shutil.copy(script, build)
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"{name}: pushing {meta['id']}, about {gpu_minutes} min of GPU", flush=True)
    kaggle("kernels", "push", "-p", str(build))
    for _ in range(gpu_minutes * 3):
        time.sleep(60)
        status = kaggle("kernels", "status", meta["id"])
        if "COMPLETE" in status:
            break
        if "ERROR" in status or "CANCEL" in status:
            sys.exit(f"{name}: {status.strip()}")
    else:
        sys.exit(f"{name}: still running after {gpu_minutes * 3} min, fetch it later with kaggle kernels output")
    kaggle("kernels", "output", meta["id"], "-p", str(preds / name))


mlflow.set_experiment(comp)
with mlflow.start_run(run_name=f"pipeline:{episode}"):
    mlflow.set_tags({"config": str(config), "stack": "+".join(cfg["stack"])})
    for model in cfg["models"]:
        if have(model):
            print(f"{model}: preds exist, skipped", flush=True)
        else:
            train(config, model)

    for name, k in cfg.get("kernels", {}).items():
        if have(name):
            print(f"{name}: preds exist, skipped", flush=True)
        elif gpu:
            run_kernel(name, k["script"], k["gpu_minutes"])
        else:
            sys.exit(f"{name}: no preds yet, rerun with --gpu to run it on Kaggle (about {k['gpu_minutes']} min of GPU)")

subprocess.run([sys.executable, "stack.py", comp, *cfg["stack"]], check=True)
print(f"submit: uv run kaggle competitions submit {comp} -f <stack file above> -m \"<msg>\"")
