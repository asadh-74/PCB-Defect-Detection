"""Copy trained weights (and the evaluation summary) into models/ for the Streamlit app.

    python -m src.export_for_app --model yolo --weights weights/yolo_best.pt
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["yolo", "frcnn"], default="yolo")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--results", default=None, help="results/<name> folder (default results/<model>)")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.weights, out / f"{args.model}_best.pt")
    res = Path(args.results or f"results/{args.model}")
    for src, dst in (("metrics.json", f"metrics_{args.model}.json"),
                     ("confusion_matrix.png", f"confusion_matrix_{args.model}.png")):
        if (res / src).exists():
            shutil.copy(res / src, out / dst)
        else:
            print(f"[warn] {res / src} not found; the app will skip it")
    size = (out / f"{args.model}_best.pt").stat().st_size / 1e6
    print(f"exported to {out}/ ({size:.1f} MB weights). Commit this folder to GitHub for deployment.")
    if size > 90:
        print("[warn] file is near GitHub's 100 MB limit; use Git LFS or a smaller model.")


if __name__ == "__main__":
    main()
