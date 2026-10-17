"""Run the whole project with ONE command (Windows, macOS, Linux; works from the VS Code terminal).

    python run_pipeline.py --profile quick   # smoke test: checks every step works (minutes)
    python run_pipeline.py --profile gpu     # full run on an NVIDIA GPU
    python run_pipeline.py --profile cpu     # lighter run for machines without a GPU

Re-running skips steps whose outputs already exist, so you can stop and resume.
Use --force to redo everything. Switching profile automatically retrains.
"""
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
    ap.add_argument("--profile", choices=PROFILES, default="quick")
    ap.add_argument("--raw", default=None, help="folder containing Annotations/ and images/ (default: auto-download)")
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: automatic)")
    ap.add_argument("--with-frcnn", action="store_true", help="also train the Faster R-CNN baseline")
    ap.add_argument("--skip-frcnn", action="store_true", help="skip the Faster R-CNN baseline")
    ap.add_argument("--force", action="store_true", help="redo steps even if their outputs exist")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()

    cfg = dict(PROFILES[args.profile])
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

    state_file = ROOT / "results" / ".pipeline_state.json"
    force = args.force
    if state_file.exists() and not args.dry_run:
        last = json.loads(state_file.read_text()).get("profile")
        if last != args.profile:
            print(f"Profile changed ({last} -> {args.profile}): retraining from scratch.")
            force = True

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

    steps = [
        ("Prepare tiled dataset",
         [PY, "-m", "src.prepare_data", "--raw", raw, "--out", "data/pcb_yolo", "--clean"],
         ["data/pcb_yolo/pcb.yaml", "data/pcb_yolo/stats.json"]),
        ("Train YOLO",
         [PY, "-m", "src.train_yolo", "--data", "data/pcb_yolo/pcb.yaml", "--model", cfg["model"],
          "--epochs", str(cfg["epochs"]), "--batch", str(cfg["batch"]), "--fraction", str(cfg["fraction"]),
          "--workers", str(workers), "--name", "yolo"] + dev,
         ["weights/yolo_best.pt"]),
    ]
    ev = ["--max_tiles", str(cfg["max_tiles"])] if cfg["max_tiles"] else []
    if use_frcnn:
        mt = ["--max_train", str(cfg["max_train"])] if cfg["max_train"] else []
        steps.append(("Train Faster R-CNN baseline",
                      [PY, "-m", "src.train_frcnn", "--data", "data/pcb_yolo", "--epochs", str(cfg["frcnn_epochs"]),
                       "--val_every", "1", "--workers", str(workers)] + mt,
                      ["weights/frcnn_best.pt"]))
    steps.append(("Evaluate YOLO",
                  [PY, "-m", "src.evaluate", "--model", "yolo", "--weights", "weights/yolo_best.pt", "--name", "yolo"] + ev + dev,
                  ["results/yolo/metrics.json"]))
    names = ["yolo"]
    if use_frcnn:
        steps.append(("Evaluate Faster R-CNN",
                      [PY, "-m", "src.evaluate", "--model", "frcnn", "--weights", "weights/frcnn_best.pt", "--name", "frcnn"] + ev,
                      ["results/frcnn/metrics.json"]))
        names.append("frcnn")
    steps.append(("Compare models", [PY, "-m", "src.compare", "--names"] + names, ["results/comparison.md"]))
    steps.append(("Export weights for the Streamlit app",
                  [PY, "-m", "src.export_for_app", "--model", "yolo", "--weights", "weights/yolo_best.pt"],
                  ["models/yolo_best.pt"]))

    if not args.dry_run:
        state_file.parent.mkdir(exist_ok=True)
        state_file.write_text(json.dumps({"profile": args.profile}))

    for i, (name, cmd, outputs) in enumerate(steps, 1):
        label = f"[{i}/{len(steps)}] {name}"
        if args.dry_run:
            print(f"\n{label}\n  $ {' '.join(cmd)}\n  outputs: {outputs}")
            continue
        # the dataset step must also rerun when forced; others skip when outputs exist
        if not force and all((ROOT / o).exists() for o in outputs):
            print(f"\n{label}: outputs exist, skipping (use --force to redo)")
            continue
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            sys.exit(f"\nSTEP FAILED: {name} (exit code {rc}). The real error is printed above. "
                     "Fix it and re-run the same command; finished steps are skipped.")
        missing = [o for o in outputs if not (ROOT / o).exists()]
        if missing:
            sys.exit(f"\nSTEP FINISHED but did not produce {missing}. Check the output above.")
        print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)

    if not args.dry_run:
        print("\nAll steps finished.\n  results:  results/comparison.md, results/yolo/, results/frcnn/"
              "\n  the app:  streamlit run streamlit_app/app.py")


if __name__ == "__main__":
    main()
