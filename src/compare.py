"""Build the comparison table and chart from results/*/metrics.json.

    python -m src.compare --names yolo frcnn
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import CLASSES, SIZE_LABELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["yolo", "frcnn"])
    ap.add_argument("--results", default="results")
    ap.add_argument("--output", default=None, help="output directory; input remains --results")
    args = ap.parse_args()
    output = Path(args.output or args.results)
    output.mkdir(parents=True, exist_ok=True)
    ms = [json.loads((Path(args.results) / n / "metrics.json").read_text()) for n in args.names]

    units = {m.get("evaluation_unit", "tile") for m in ms}
    if len(units) != 1:
        raise ValueError("Cannot compare different evaluation units")
    def fmt(value, places=3):
        return "N/A" if value is None else f"{value:.{places}f}"
    lines = [f"Evaluation unit: {next(iter(units))}. N/A means unavailable, not zero.", "",
             "| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Macro-F1 | ms/image | FPS | Params (M) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in ms:
        params = f"{m['n_params'] / 1e6:.1f}" if m.get("n_params") else "-"
        lines.append(f"| {m['name']} ({m['split_mode'] if 'split_mode' in m else 'unknown'}) | {fmt(m['map50'])} | {fmt(m['map50_95'])} | {fmt(m['precision'])} | "
                     f"{fmt(m['recall'])} | {fmt(m['f1_macro'])} | {fmt(m['ms_per_tile'], 1)} | {fmt(m['fps'], 1)} | {params} |")
    lines += ["", "Per-class AP@0.5:", "", "| Class | " + " | ".join(m["name"] for m in ms) + " |",
              "|---|" + "---|" * len(ms)]
    for c in CLASSES:
        lines.append(f"| {c} | " + " | ".join(fmt(m["per_class_ap50"][c]) for m in ms) + " |")
    lines += ["", "Recall by defect size (sqrt of box area, evaluation image pixels):", "",
              "| Size | " + " | ".join(m["name"] + " (n; recall)" for m in ms) + " |", "|---|" + "---|" * len(ms)]
    for s in SIZE_LABELS:
        vals = [m["size_recall"][s] for m in ms]
        lines.append(f"| {s} | " +
                     " | ".join(f"{m['size_counts'][s]}; {fmt(v)}" for m, v in zip(ms, vals)) + " |")
    out = output / "comparison.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x, w = np.arange(len(CLASSES)), 0.8 / len(ms)
    for k, m in enumerate(ms):
        axes[0].bar(x + k * w, [np.nan if m["per_class_ap50"][c] is None else m["per_class_ap50"][c] for c in CLASSES], w, label=m["name"])
        axes[1].bar(np.arange(len(SIZE_LABELS)) + k * w,
                    [np.nan if m["size_recall"][s] is None else m["size_recall"][s] for s in SIZE_LABELS], w, label=m["name"])
    axes[0].set_xticks(x + w * (len(ms) - 1) / 2); axes[0].set_xticklabels(CLASSES, rotation=30, ha="right")
    axes[0].set_ylabel("AP@0.5"); axes[0].set_title("Per-class AP")
    axes[1].set_xticks(np.arange(len(SIZE_LABELS)) + w * (len(ms) - 1) / 2); axes[1].set_xticklabels(SIZE_LABELS)
    axes[1].set_ylabel("recall"); axes[1].set_title("Recall by defect size")
    for a in axes:
        a.set_ylim(0, 1.05); a.legend()
    fig.tight_layout(); fig.savefig(output / "comparison.png", dpi=150)
    print(f"\nsaved {out} and comparison.png")


if __name__ == "__main__":
    main()
