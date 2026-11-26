import json
import subprocess
import sys
import time
from pathlib import Path

import mlflow

from train import load_config, train

config = Path(sys.argv[1])
force, gpu = "--force" in sys.argv, "--gpu" in sys.argv
cfg = load_config(config)
comp, episode = cfg["comp"], config.stem
preds = Path("preds") / comp


def have(model):
    return not force and (preds / model / "oof.csv").exists()


def kaggle(*args):
    return subprocess.run(["kaggle", *args], check=True, capture_output=True, text=True).stdout


def build_kernel(name, k):
    script = Path(k["script"])
    meta = json.loads((script.parent / "kernel-metadata.json").read_text())
    owner = meta["id"].split("/")[0]
    # Kaggle derives the slug from the title, so they must agree
    meta |= {"id": f"{owner}/{episode}-{name}-oof", "title": f"{episode} {name} oof",
             "code_file": script.name, "competition_sources": [comp]}
    build = Path("build") / episode / name
    build.mkdir(parents=True, exist_ok=True)
    features = {"decimals": cfg["decimals"], "combos": [], "bins": {}, "te": []} | k.get("features", {})
    (build / script.name).write_text(f"FEATURES = {features!r}\n" + script.read_text())
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return build, meta["id"]


def run_kernel(name, k):
    build, kernel_id = build_kernel(name, k)
    gpu_minutes = k["gpu_minutes"]
    print(f"{name}: pushing {kernel_id}, about {gpu_minutes} min of GPU", flush=True)
    kaggle("kernels", "push", "-p", str(build))
    for _ in range(gpu_minutes * 3):
        time.sleep(60)
        status = kaggle("kernels", "status", kernel_id)
        if "COMPLETE" in status:
            break
        if "ERROR" in status or "CANCEL" in status:
            sys.exit(f"{name}: {status.strip()}")
    else:
        sys.exit(f"{name}: still running after {gpu_minutes * 3} min, fetch it later with kaggle kernels output")
    kaggle("kernels", "output", kernel_id, "-p", str(preds / name))


if "--build" in sys.argv:
    for name, k in cfg.get("kernels", {}).items():
        print(build_kernel(name, k)[0] / Path(k["script"]).name)
    sys.exit()

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
            run_kernel(name, k)
        else:
            sys.exit(f"{name}: no preds yet, rerun with --gpu to run it on Kaggle (about {k['gpu_minutes']} min of GPU)")

subprocess.run([sys.executable, "stack.py", comp, *cfg["stack"]], check=True)
print(f"submit: uv run kaggle competitions submit {comp} -f <stack file above> -m \"<msg>\"")
