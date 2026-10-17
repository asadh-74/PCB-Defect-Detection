"""Run the whole project with ONE command (Windows, macOS, Linux; works from the VS Code terminal).

    python run_pipeline.py --profile quick   # smoke test: checks every step works (minutes)
    python run_pipeline.py --profile gpu     # full run on an NVIDIA GPU
    python run_pipeline.py --profile cpu     # lighter run for machines without a GPU

Re-running skips steps whose outputs already exist, so you can stop and resume.
Use --force to redo everything. Switching profile automatically retrains.
"""
import hashlib
import re
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

PROFILES = {
    "quick": dict(model="yolov8n.pt", epochs=2, batch=8, fraction=0.1, frcnn_epochs=1, max_train=120, max_tiles=150),
    "cpu": dict(model="yolov8n.pt", epochs=30, batch=8, fraction=1.0, frcnn_epochs=0, max_train=None, max_tiles=None),
    "gpu": dict(model="yolov8s.pt", epochs=100, batch=16, fraction=1.0, frcnn_epochs=20, max_train=None, max_tiles=None),
}


def have_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="default", help="isolated experiment name (letters, digits, _ and -)")
    ap.add_argument("--split-mode", choices=["image", "board"], default="board")
    ap.add_argument("--representation", choices=["tiled", "full"], default="tiled")
    ap.add_argument("--augmentation", choices=["on", "off"], default="on")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--profile", choices=PROFILES, default="quick")
    ap.add_argument("--raw", default=None, help="folder containing Annotations/ and images/ (default: auto-download)")
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: automatic)")
    ap.add_argument("--with-frcnn", action="store_true", help="also train the Faster R-CNN baseline")
    ap.add_argument("--skip-frcnn", action="store_true", help="skip the Faster R-CNN baseline")
    ap.add_argument("--force", action="store_true", help="redo steps even if their outputs exist")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        ap.error("--run-name may contain only letters, digits, underscores and hyphens")
    cfg = dict(PROFILES[args.profile])
    for key in ("epochs", "batch", "model"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if cfg["epochs"] < 1 or cfg["batch"] < 1:
        ap.error("--epochs and --batch must be positive")
    cuda = have_cuda()
    use_frcnn = cfg["frcnn_epochs"] > 0
    if args.profile == "quick":
        use_frcnn = cuda  # on CPU the baseline makes the smoke test very slow
    if args.with_frcnn:
        use_frcnn = True
        cfg["frcnn_epochs"] = cfg["frcnn_epochs"] or 5
    if args.skip_frcnn:
        use_frcnn = False
    print(f"profile={args.profile} | CUDA GPU: {cuda} | Faster R-CNN baseline: {use_frcnn}")
    if args.profile == "gpu" and not cuda and not args.dry_run:
        sys.exit("Profile 'gpu' needs an NVIDIA GPU with a CUDA build of PyTorch (see LOCAL_SETUP.md). "
                 "Use --profile cpu, or train on Colab/Kaggle.")

    state_file = ROOT / "results" / f"pipeline_{args.run_name}.json"
    force = args.force
    workers = 2 if os.name == "nt" else 4
    dev = ["--device", args.device] if args.device else []

    if args.dry_run:
        raw = args.raw or "<auto-downloaded dataset folder>"
    elif args.raw:
        raw = args.raw
    else:
        sys.path.insert(0, str(ROOT))
        from src.download_data import get_dataset
        raw = get_dataset(str(ROOT / "data" / "raw"))
    print("dataset:", raw)

    data = f"data/pcb_{args.run_name}"
    yn, fn = f"{args.run_name}_yolo", f"{args.run_name}_frcnn"
    yw, fw = f"weights/{yn}_best.pt", f"weights/{fn}_best.pt"
    steps = [
        ("Prepare dataset",
         [PY, "-m", "src.prepare_data", "--raw", raw, "--out", data, "--clean", "--split-mode", args.split_mode, "--representation", args.representation, "--seed", str(args.seed)],
         [f"{data}/pcb.yaml", f"{data}/stats.json", f"{data}/manifest.json"]),
        ("Train YOLO",
         [PY, "-m", "src.train_yolo", "--data", f"{data}/pcb.yaml", "--model", cfg["model"],
          "--epochs", str(cfg["epochs"]), "--batch", str(cfg["batch"]), "--fraction", str(cfg["fraction"]),
          "--workers", str(workers), "--name", yn, "--out", yw, "--seed", str(args.seed), "--augmentation", args.augmentation] + dev,
         [yw]),
    ]
    ev = ["--max_tiles", str(cfg["max_tiles"])] if cfg["max_tiles"] else []
    if use_frcnn:
        mt = ["--max_train", str(cfg["max_train"])] if cfg["max_train"] else []
        steps.append(("Train Faster R-CNN baseline",
                      [PY, "-m", "src.train_frcnn", "--data", data, "--epochs", str(cfg["frcnn_epochs"]),
                       "--val_every", "1", "--workers", str(workers), "--seed", str(args.seed), "--out", fw] + mt + dev,
                      [fw]))
    steps.append(("Evaluate YOLO",
                  [PY, "-m", "src.evaluate", "--model", "yolo", "--weights", yw, "--name", yn, "--data", data, "--unit", "board"] + ev + dev,
                  [f"results/{yn}/metrics.json"]))
    names = [yn]
    if use_frcnn:
        steps.append(("Evaluate Faster R-CNN",
                      [PY, "-m", "src.evaluate", "--model", "frcnn", "--weights", fw, "--name", fn, "--data", data, "--unit", "board"] + ev + dev,
                      [f"results/{fn}/metrics.json"]))
        names.append(fn)
    steps.append(("Compare models", [PY, "-m", "src.compare", "--output", f"results/{args.run_name}", "--names"] + names, [f"results/{args.run_name}/comparison.md"]))
    steps.append(("Export weights for the Streamlit app",
                  [PY, "-m", "src.export_for_app", "--model", "yolo", "--weights", yw, "--results", f"results/{yn}"],
                  ["models/yolo_best.pt"]))

    # Config and source changes invalidate completion records, not just the profile.
    # Raw metadata catches local dataset changes; preparation saves content hashes too.
    raw_path = Path(raw).resolve()
    source = [ROOT / "run_pipeline.py"] + sorted((ROOT / "src").glob("*.py"))
    fingerprint = {"options": vars(args) | {"force": False, "dry_run": False},
                   "raw": str(raw_path), "cfg": cfg, "frcnn": use_frcnn,
                   "source": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source}}
    if raw_path.exists():
        fingerprint["raw_files"] = [(str(p.relative_to(raw_path)), p.stat().st_size, p.stat().st_mtime_ns)
                                     for p in sorted(raw_path.rglob("*")) if p.is_file()]
    signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    state = {"signature": signature, "completed": []}
    if state_file.exists() and not force:
        old = json.loads(state_file.read_text())
        if old.get("signature") == signature:
            state = old
    if not args.dry_run:
        state_file.parent.mkdir(exist_ok=True)
        state_file.write_text(json.dumps(state))
    invalidated = False

    for i, (name, cmd, outputs) in enumerate(steps, 1):
        label = f"[{i}/{len(steps)}] {name}"
        if args.dry_run:
            print(f"\n{label}\n  $ {subprocess.list2cmdline([str(x) for x in cmd])}\n  outputs: {outputs}")
            continue
        # the dataset step must also rerun when forced; others skip when outputs exist
        if name != "Export weights for the Streamlit app" and not invalidated and not force and name in state["completed"] and all((ROOT / o).exists() for o in outputs):
            print(f"\n{label}: outputs exist, skipping (use --force to redo)")
            continue
        invalidated = True
        # Remove this and later completion records BEFORE executing a possibly failing step.
        earlier = {entry[0] for entry in steps[:i-1]}
        state["completed"] = [entry for entry in state["completed"] if entry in earlier]
        state_file.write_text(json.dumps(state))
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            sys.exit(f"\nSTEP FAILED: {name} (exit code {rc}). The real error is printed above. "
                     "Fix it and re-run the same command; finished steps are skipped.")
        missing = [o for o in outputs if not (ROOT / o).exists()]
        if missing:
            sys.exit(f"\nSTEP FINISHED but did not produce {missing}. Check the output above.")
        state["completed"].append(name)
        state_file.write_text(json.dumps(state))
        print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)

    if not args.dry_run:
        print(f"\nAll steps finished.\n  results: results/{yn}/ and results/{args.run_name}/comparison.md"
              "\n  the app:  streamlit run streamlit_app/app.py")


if __name__ == "__main__":
    main()
