"""Evaluate a trained detector on the held-out test tiles.

    python -m src.evaluate --model yolo  --weights weights/yolo_best.pt  --name yolo
    python -m src.evaluate --model frcnn --weights weights/frcnn_best.pt --name frcnn

The operating confidence threshold is chosen on the VALIDATION split (best macro-F1) and then
applied unchanged to the test split. mAP uses all detections down to conf 0.001.
Outputs go to results/<name>/: metrics.json, per_class.csv, confusion_matrix.png,
conf_sweep.png, error_examples/.
"""
import hashlib
import re
import shutil
import platform
import importlib.metadata
import argparse
import json
import random
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import CLASSES, SIZE_LABELS, SIZE_EDGES, read_yolo_labels  # noqa: E402
from .infer import draw  # noqa: E402
from .metrics import compute_map, evaluate_at_conf, sweep_conf, match_image  # noqa: E402


def load_split(data_root, split, max_tiles=None):
    paths = sorted((Path(data_root) / "images" / split).glob("*.jpg"))
    if max_tiles and len(paths) > max_tiles:  # quick tests: half defect tiles, half clean tiles
        lbl = Path(data_root) / "labels" / split
        has = [p for p in paths if (lbl / f"{p.stem}.txt").stat().st_size > 0]
        has_set = set(has)
        clean = [p for p in paths if p not in has_set]
        rng = random.Random(0)
        k = max_tiles // 2
        chosen = rng.sample(has, min(k, len(has))) + rng.sample(clean, min(max_tiles - min(k, len(has)), len(clean)))
        selected = set(chosen)
        remaining = [p for p in paths if p not in selected]
        paths = sorted(chosen + rng.sample(remaining, min(max_tiles-len(chosen), len(remaining))))
    if not paths:
        raise ValueError(f"Empty {split} evaluation split under {data_root}")
    gts = []
    for p in paths:
        image = cv2.imread(str(p))
        if image is None:
            raise ValueError(f"Unreadable image: {p}")
        h, w = image.shape[:2]
        boxes, cls = read_yolo_labels(Path(data_root) / "labels" / split / f"{p.stem}.txt", w, h)
        gts.append({"boxes": boxes, "cls": cls})
    return paths, gts


def run_predictions(predictor, paths, conf=0.001, desc="predict"):
    preds = []
    for p in tqdm(paths, desc=desc):
        b, s, c = predictor.predict(cv2.imread(str(p)), conf=conf)
        preds.append({"boxes": b, "scores": s, "cls": c})
    return preds


def measure_speed(predictor, paths, n=200, warmup=10):
    if not paths:
        return None, None
    imgs = [cv2.imread(str(p)) for p in paths[:n]]
    for i in range(warmup):
        predictor.predict(imgs[i % len(imgs)], conf=0.25)
    t0 = time.perf_counter()
    for im in imgs:
        predictor.predict(im, conf=0.25)
    dt = max((time.perf_counter() - t0) / len(imgs), 1e-12)
    return dt * 1000, 1.0 / dt


class BoardPredictor:
    """Same original-board evaluation inputs for both representations."""
    def __init__(self, predictor, representation, tile=640, stride=480):
        self.base, self.representation = predictor, representation
        self.tile, self.stride = tile, stride
        self.name = predictor.name
        self.n_params = getattr(predictor, "n_params", None)

    def predict(self, image, conf=0.001):
        if self.representation == "full":
            return self.base.predict(image, conf=conf)
        from .infer import tiled_predict
        return tiled_predict(self.base, image, tile=self.tile, stride=self.stride, conf=conf)


def save_error_records(out_dir, paths, gts, preds, res, conf, unit="tile"):
    """Machine-readable observations, not inferred physical causes."""
    rows = []
    denominators = {"clean": 0, "contains_gt": 0}
    for gt in gts:
        denominators["contains_gt" if len(gt["boxes"]) else "clean"] += 1
    for kind, items in (("FN", res["fn_list"]), ("FP", res["fp_list"])):
        for idx, k in items:
            image = cv2.imread(str(paths[idx]))
            h, w = image.shape[:2]
            p, g = preds[idx], gts[idx]
            keep = p["scores"] >= conf
            boxes, classes = (g["boxes"], g["cls"]) if kind == "FN" else (p["boxes"][keep], p["cls"][keep])
            box = boxes[k]
            margin = float(min(box[0], box[1], w-box[2], h-box[3]))
            match = re.match(r"^(.*)_x(\d+)_y(\d+)$", paths[idx].stem) if unit == "tile" else None
            source = match[1] if match else paths[idx].stem
            rows.append({"kind": kind, "image": paths[idx].name, "source_image": source,
                         "board_id": source.split("_", 1)[0], "class": CLASSES[int(classes[k])],
                         "score": None if kind == "FN" else float(p["scores"][keep][k]),
                         "box": [float(v) for v in box],
                         "size_sqrt_area_px": float(np.sqrt(max(0, (box[2]-box[0])*(box[3]-box[1])))),
                         "edge_distance_px": margin, "near_edge_16px": margin < 16,
                         "tile_x": int(match[2]) if match else None,
                         "tile_y": int(match[3]) if match else None,
                         "context": "contains_gt" if len(g["boxes"]) else "clean"})
    out_dir = Path(out_dir)
    (out_dir / "errors.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    pd.DataFrame(rows, columns=["kind", "image", "source_image", "board_id", "class", "score", "box",
                               "size_sqrt_area_px", "edge_distance_px", "near_edge_16px", "tile_x", "tile_y", "context"]).to_csv(
        out_dir / "errors.csv", index=False)
    context = {c: {"n_images": n, "fp_count": sum(r["kind"] == "FP" and r["context"] == c for r in rows),
                   "fp_per_image": (sum(r["kind"] == "FP" and r["context"] == c for r in rows) / n if n else None)}
               for c, n in denominators.items()}
    (out_dir / "error_context.json").write_text(json.dumps({"unit": unit, "contexts": context,
        "note": "Near-edge counts describe error locations, not a causal error rate. Inspect crops to label visual causes."}, indent=2), encoding="utf-8")


def plot_confusion(cm, path, title):
    labels = CLASSES + ["background"]
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if norm[i, j] > 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_error_examples(out_dir, paths, gts, preds, res, conf, n=12, seed=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_image in out_dir.glob("*.jpg"):
        old_image.unlink()
    rng = random.Random(seed)
    for kind, items in (("FN", res["fn_list"]), ("FP", res["fp_list"])):
        for idx, k in rng.sample(items, min(n, len(items))):
            img = cv2.imread(str(paths[idx]))
            g, p = gts[idx], preds[idx]
            keep = p["scores"] >= conf
            img = draw(img, g["boxes"], cls=g["cls"], color=(0, 200, 0), thickness=2)       # GT green
            img = draw(img, p["boxes"][keep], p["scores"][keep], p["cls"][keep], color=(0, 0, 255))  # preds red
            hb = g["boxes"][k] if kind == "FN" else p["boxes"][keep][k]
            x0, y0, x1, y1 = (int(v) for v in hb)
            cv2.rectangle(img, (x0 - 6, y0 - 6), (x1 + 6, y1 + 6), (0, 255, 255), 2)         # the error, yellow
            cv2.imwrite(str(out_dir / f"{kind}_{paths[idx].stem}_{k}.jpg"), img)


def analyse(predictor, data_root, out_dir, conf="auto", name="model", do_speed=True, max_tiles=None, unit="tile", weights=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cls = len(CLASSES)
    data_root = Path(data_root)
    stats_path = data_root / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    evaluation_root = data_root / "boards" if unit == "board" else data_root
    if unit == "board":
        predictor = BoardPredictor(predictor, stats.get("representation", "tiled"),
                                   stats.get("tile", 640), stats.get("stride", 480))
    if conf != "auto" and (not np.isfinite(float(conf)) or not 0 <= float(conf) <= 1):
        raise ValueError("Confidence must be 'auto' or a finite number in [0,1]")

    conf_was_auto = conf == "auto"
    if not conf_was_auto:
        for filename in ("confidence_sweep.json", "conf_sweep.png"):
            (out_dir / filename).unlink(missing_ok=True)
    if conf_was_auto:
        v_paths, v_gts = load_split(evaluation_root, "val", max_tiles)
        if not any(len(g["cls"]) for g in v_gts):
            raise ValueError("Validation has no ground-truth defects; cannot tune confidence")
        v_preds = run_predictions(predictor, v_paths, desc="val")
        conf, curve = sweep_conf(v_gts, v_preds, n_cls)
        (out_dir / "confidence_sweep.json").write_text(json.dumps(curve), encoding="utf-8")
        plt.figure(figsize=(5, 3.5)); plt.plot(*zip(*curve), marker="o")
        plt.axvline(conf, color="red", ls="--"); plt.xlabel("confidence"); plt.ylabel("macro-F1 (val)")
        plt.title(f"Operating point: {conf:.2f}"); plt.tight_layout()
        plt.savefig(out_dir / "conf_sweep.png", dpi=150); plt.close()
    conf = float(conf)

    paths, gts = load_split(evaluation_root, "test", max_tiles)
    preds = run_predictions(predictor, paths, desc="test")
    ap = compute_map(gts, preds, n_cls)
    res = evaluate_at_conf(gts, preds, n_cls, conf)
    pc = res["per_class"]

    rows = []
    for i, c in enumerate(CLASSES):
        rows.append({"class": c, "AP50": ap[0, i], "AP50-95": np.nanmean(ap[:, i]) if np.isfinite(ap[:, i]).any() else float("nan"),
                     "precision": pc["precision"][i], "recall": pc["recall"][i], "f1": pc["f1"][i],
                     "TP": int(pc["tp"][i]), "FP": int(pc["fp"][i]), "FN": int(pc["fn"][i])})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_class.csv", index=False)
    totals = np.zeros((n_cls, len(SIZE_LABELS)), dtype=int)
    found = np.zeros_like(totals)
    for gt, pred in zip(gts, preds):
        selected = pred["scores"] >= conf
        matches = match_image(gt["boxes"], gt["cls"], pred["boxes"][selected],
                              pred["scores"][selected], pred["cls"][selected], 0.5)
        recovered = set(matches[matches >= 0].tolist())
        for j, (box, cls) in enumerate(zip(gt["boxes"], gt["cls"])):
            bucket = int(np.digitize(np.sqrt((box[2]-box[0])*(box[3]-box[1])), SIZE_EDGES))
            totals[int(cls), bucket] += 1
            found[int(cls), bucket] += int(j in recovered)
    pd.DataFrame([{"class": c, "size": label, "count": int(totals[i, k]), "found": int(found[i, k]),
                   "recall": float(found[i, k]/totals[i, k]) if totals[i, k] else None}
                  for i, c in enumerate(CLASSES) for k, label in enumerate(SIZE_LABELS)]).to_csv(
        out_dir / "size_by_class.csv", index=False)


    ms, fps = measure_speed(predictor, paths) if do_speed else (None, None)
    metrics = {
        "tile": stats.get("tile", 640), "stride": stats.get("stride", 480),
        "evaluation_unit": unit, "split_mode": stats.get("split_mode", "unknown"),
        "representation": stats.get("representation", "unknown"), "subset_evaluation": max_tiles is not None,
        "threshold_source": "validation" if conf_was_auto else "manual",
        "size_definition": "sqrt(box area) in original board pixels" if unit == "board" else "sqrt(box area) in prepared image pixels",
        "n_images": len(paths), "ms_per_image": ms,
        "name": name, "model": getattr(predictor, "name", name), "split": "test", "conf": conf,
        "n_tiles": len(paths), "n_gt": int(sum(len(g["cls"]) for g in gts)),
        "map50": float(np.nanmean(ap[0])) if np.isfinite(ap[0]).any() else float("nan"), "map50_95": float(np.nanmean(ap)) if np.isfinite(ap).any() else float("nan"),
        "precision": res["precision_micro"], "recall": res["recall_micro"], "f1_macro": res["f1_macro"],
        "per_class_ap50": {c: float(ap[0, i]) for i, c in enumerate(CLASSES)},
        "size_recall": {l: (None if np.isnan(v) else float(v)) for l, v in zip(SIZE_LABELS, res["size_recall"])},
        "size_counts": {l: int(v) for l, v in zip(SIZE_LABELS, res["size_counts"])},
        "confusion": res["confusion"].tolist(),
        "ms_per_tile": ms, "fps": fps, "n_params": getattr(predictor, "n_params", None),
    }
    def finite_json(obj):
        if isinstance(obj, dict):
            return {k: finite_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [finite_json(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    metrics = finite_json(metrics)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    save_error_records(out_dir, paths, gts, preds, res, conf, unit)
    for filename in ("stats.json", "manifest.json", "splits.csv"):
        if (data_root / filename).exists():
            shutil.copyfile(data_root / filename, out_dir / filename)
    if weights and Path(weights).with_suffix(".config.json").exists():
        shutil.copyfile(Path(weights).with_suffix(".config.json"), out_dir / "training_config.json")
    if weights and Path(weights).with_suffix(".history.csv").exists():
        shutil.copyfile(Path(weights).with_suffix(".history.csv"), out_dir / "training_history.csv")
    packages = {}
    for package in ("numpy", "torch", "torchvision", "ultralytics", "opencv-python-headless", "streamlit", "albumentations"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not installed"
    provenance = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
                  "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
                  "weights_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest() if weights else None}
    try:
        import torch
        provenance["hardware"] = {"cuda_available": torch.cuda.is_available(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "cpu": platform.processor(), "predictor_device": str(getattr(getattr(predictor, "base", predictor), "device", "automatic"))}
    except ImportError:
        provenance["hardware"] = {"cpu": platform.processor()}
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    prediction_records = [{"image": p.name, "boxes": pred["boxes"].tolist(), "scores": pred["scores"].tolist(),
                           "cls": pred["cls"].tolist()} for p, pred in zip(paths, preds)]
    ground_truth = [{"image": p.name, "boxes": gt["boxes"].tolist(), "cls": gt["cls"].tolist()}
                    for p, gt in zip(paths, gts)]
    (out_dir / "ground_truth.json").write_text(json.dumps(ground_truth), encoding="utf-8")
    (out_dir / "predictions.json").write_text(json.dumps(prediction_records), encoding="utf-8")
    plot_confusion(res["confusion"], out_dir / "confusion_matrix.png", f"{name} (test, conf {conf:.2f})")
    save_error_examples(out_dir / "error_examples", paths, gts, preds, res, conf)

    print(f"\n{name}: mAP@0.5 {metrics['map50']} | mAP@0.5:0.95 {metrics['map50_95']} | "
          f"P {metrics['precision']:.3f} R {metrics['recall']:.3f} | FPS={fps} ({unit}) @ conf {conf:.2f}")
    print(df.round(3).to_string(index=False))
    print("recall by defect size:", metrics["size_recall"])
    return metrics


def main():
    from .predictors import load_predictor
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", choices=["tile", "board"], default="tile")
    ap.add_argument("--model", choices=["yolo", "frcnn"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/pcb_yolo")
    ap.add_argument("--conf", default="auto", help="'auto' (tuned on val) or a float")
    ap.add_argument("--name", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_tiles", type=int, default=None, help="evaluate on a subset (quick tests only)")
    args = ap.parse_args()
    name = args.name or args.model
    pred = load_predictor(args.model, args.weights, imgsz=args.imgsz, device=args.device)
    analyse(pred, args.data, Path("results") / name, args.conf, name, max_tiles=args.max_tiles, unit=args.unit, weights=args.weights)


if __name__ == "__main__":
    main()
