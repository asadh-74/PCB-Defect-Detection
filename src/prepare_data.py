"""Convert PKU-Market-PCB (Pascal VOC XML) into a tiled YOLO-format dataset.

Key decisions
- Split by SOURCE IMAGE (stratified by the first defect class in each image), then tile.
  Tiling first and splitting after would leak overlapping pixels between train and test.
- Tiles are TILE x TILE with STRIDE overlap so small defects are not downscaled.
- A box is kept in a tile if >= 50% of its area is inside. A tile that cuts through a defect
  (5%-50% visible) with no fully kept box is dropped in every split, so it never teaches or
  scores a half-visible defect as background.
- Train: all tiles with defects + a small fraction of background tiles.
  Val/test: ALL clean tiles are kept so false positives are measured honestly.

Usage:
    python -m src.prepare_data --raw /path/to/PCB_DATASET --out data/pcb_yolo
"""
import argparse
import hashlib
import warnings
import csv
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .common import CLASSES, IMG_EXT, norm_name, positions

Image.MAX_IMAGE_PIXELS = None


def locate_dir(raw: Path, name: str) -> Path:
    cands = [p for p in raw.rglob("*") if p.is_dir() and p.name.lower() == name.lower()]
    if raw.name.lower() == name.lower():
        cands.append(raw)
    if not cands:
        raise FileNotFoundError(f"Could not find a '{name}' folder under {raw}. Use --ann_dir/--img_dir.")
    return min(cands, key=lambda p: len(p.parts))


def parse_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        name = norm_name(obj.findtext("name", ""))
        if name not in CLASSES:
            print(f"  [warn] unknown class '{name}' in {xml_path.name}, skipped")
            continue
        bb = obj.find("bndbox")
        x0, y0, x1, y1 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        if x1 > x0 and y1 > y0:
            boxes.append((CLASSES.index(name), x0, y0, x1, y1))
    return boxes


def split_samples(samples, ratios, seed):
    """samples: list of (stem, img_path, boxes). Stratified by first box class."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for s in samples:
        groups[s[2][0][0]].append(s)
    out = {"train": [], "val": [], "test": []}
    for _, items in sorted(groups.items()):
        items = sorted(items, key=lambda s: s[0])
        rng.shuffle(items)
        n = len(items)
        n_val = max(1, round(n * ratios[1])) if n >= 3 else 0
        n_test = max(1, round(n * ratios[2])) if n >= 3 else 0
        out["val"] += items[:n_val]
        out["test"] += items[n_val:n_val + n_test]
        out["train"] += items[n_val + n_test:]
    return out


def board_id(stem):
    """Dataset convention; verify prefixes against actual template boards."""
    prefix, sep, _ = stem.partition("_")
    if not sep or not prefix:
        raise ValueError(f"Cannot extract board prefix from {stem!r}")
    return prefix


def grouped_split(samples, ratios, seed):
    """Assign whole boards, deterministically. Ratios apply to group counts."""
    groups = defaultdict(list)
    for sample in sorted(samples, key=lambda s: s[0]):
        groups[board_id(sample[0])].append(sample)
    ids = sorted(groups)
    if len(ids) < 3:
        raise ValueError("Board splitting needs at least three distinct board prefixes")
    random.Random(seed).shuffle(ids)
    nv = max(1, round(len(ids) * ratios[1]))
    nt = max(1, round(len(ids) * ratios[2]))
    if nv + nt >= len(ids):
        raise ValueError("Split ratios leave no training boards")
    assignments = {"val": ids[:nv], "test": ids[nv:nv + nt], "train": ids[nv + nt:]}
    return {sp: [s for bid in bids for s in groups[bid]] for sp, bids in assignments.items()}


def save_board(split, stem, img, boxes, out):
    """Keep original-resolution boards and labels for a common evaluation unit."""
    image_dir = out / "boards" / "images" / split
    label_dir = out / "boards" / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    img.save(image_dir / f"{stem}.jpg", quality=95)
    w, h = img.size
    lines = [f"{c} {(a+x)/2/w:.8f} {(b+y)/2/h:.8f} {(x-a)/w:.8f} {(y-b)/h:.8f}"
             for c, a, b, x, y in boxes]
    (label_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def tile_boxes(boxes, x0, y0, tile, min_vis=0.5, min_px=2.0):
    """Return (kept boxes in tile coords, ambiguous flag)."""
    x1, y1 = x0 + tile, y0 + tile
    kept, ambiguous = [], False
    for c, bx0, by0, bx1, by1 in boxes:
        ix0, iy0, ix1, iy1 = max(bx0, x0), max(by0, y0), min(bx1, x1), min(by1, y1)
        iw, ih = ix1 - ix0, iy1 - iy0
        if iw <= 0 or ih <= 0:
            continue
        frac = (iw * ih) / max((bx1 - bx0) * (by1 - by0), 1e-6)
        if frac >= min_vis and iw >= min_px and ih >= min_px:
            kept.append((c, ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0))
        elif frac >= 0.05:
            ambiguous = True
    return kept, ambiguous


def process_split(split, items, out_root, args, rng, stats):
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for stem, img_path, boxes in items:
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        boxes = [(c, max(0, a), max(0, b), min(W, x), min(H, y)) for c, a, b, x, y in boxes
                 if min(W, x) > max(0, a) and min(H, y) > max(0, b)]
        save_board(split, stem, img, boxes, out_root)
        if args.representation == "full":
            img.save(img_dir / f"{stem}.jpg", quality=95)
            shutil.copyfile(out_root / "boards" / "labels" / split / f"{stem}.txt", lbl_dir / f"{stem}.txt")
            stats["tiles"][split] += 1
            for c, a, b, x, y in boxes:
                stats["boxes"][split][CLASSES[c]] += 1
                stats["sizes"].append(float(np.sqrt((x-a)*(y-b))))
            continue
        for y0 in positions(H, args.tile, args.stride):
            for x0 in positions(W, args.tile, args.stride):
                kept, ambiguous = tile_boxes(boxes, x0, y0, args.tile)
                # An unlabelled partial defect is unsafe even beside another kept box.
                if ambiguous:
                    stats.setdefault("dropped_ambiguous", Counter())[split] += 1
                    continue
                if not kept:
                    if split == "train" and rng.random() >= args.bg_ratio:
                        continue
                name = f"{stem}_x{x0}_y{y0}"
                img.crop((x0, y0, x0 + args.tile, y0 + args.tile)).save(img_dir / f"{name}.jpg", quality=95)
                lines = []
                for c, a, b, c2, d in kept:
                    a, b = max(a, 0), max(b, 0)
                    c2, d = min(c2, args.tile), min(d, args.tile)
                    lines.append(f"{c} {(a + c2) / 2 / args.tile:.6f} {(b + d) / 2 / args.tile:.6f} "
                                 f"{(c2 - a) / args.tile:.6f} {(d - b) / args.tile:.6f}")
                    stats["boxes"][split][CLASSES[c]] += 1
                    stats["sizes"].append(float(np.sqrt((c2 - a) * (d - b))))
                (lbl_dir / f"{name}.txt").write_text("\n".join(lines))
                stats["tiles"][split] += 1
                stats["bg_tiles"][split] += int(not kept)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="folder containing Annotations/ and images/")
    ap.add_argument("--ann_dir", default=None)
    ap.add_argument("--img_dir", default=None)
    ap.add_argument("--out", default="data/pcb_yolo")
    ap.add_argument("--split-mode", choices=["image", "board"], default="image")
    ap.add_argument("--representation", choices=["tiled", "full"], default="tiled")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--stride", type=int, default=480)
    ap.add_argument("--split", type=float, nargs=3, default=(0.70, 0.15, 0.15))
    ap.add_argument("--bg_ratio", type=float, default=0.10, help="fraction of clean train tiles to keep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean", action="store_true", help="delete --out first")
    args = ap.parse_args()

    if any(r <= 0 for r in args.split) or not np.isclose(sum(args.split), 1):
        ap.error("--split must contain three positive ratios summing to one")
    if not 0 <= args.bg_ratio <= 1:
        ap.error("--bg_ratio must be between zero and one")
    positions(1, args.tile, args.stride)  # validate before touching outputs
    raw, out = Path(args.raw), Path(args.out)
    if out.resolve() == raw.resolve() or out.resolve() in raw.resolve().parents or raw.resolve() in out.resolve().parents:
        ap.error("--out and --raw must be separate directories, not nested")
    ann_dir = Path(args.ann_dir) if args.ann_dir else locate_dir(raw, "Annotations")
    img_dir = Path(args.img_dir) if args.img_dir else locate_dir(raw, "images")
    print(f"annotations: {ann_dir}\nimages:      {img_dir}")

    index = {}
    for p in img_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            if p.stem in index:
                raise ValueError(f"Duplicate image stem {p.stem}: resolve ambiguity before splitting")
            index[p.stem] = p

    samples, missing = [], 0
    for xml_path in sorted(ann_dir.rglob("*.xml")):
        if xml_path.stem not in index:
            missing += 1
            continue
        boxes = parse_voc(xml_path)
        if boxes:
            samples.append((xml_path.stem, index[xml_path.stem], boxes))
    print(f"{len(samples)} annotated images found ({missing} XML files had no matching image)")
    if not samples:
        raise SystemExit("No samples found. Check the folder layout.")

    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    if len({s[0] for s in samples}) != len(samples):
        raise ValueError("Duplicate XML stems would duplicate samples across splits")
    splits = (grouped_split if args.split_mode == "board" else split_samples)(samples, args.split, args.seed)
    for sp, items in splits.items():
        present = {b[0] for sample in items for b in sample[2]}
        if not items:
            raise ValueError(f"Empty {sp} source split")
        absent = [c for i, c in enumerate(CLASSES) if i not in present]
        if absent:
            warnings.warn(f"{sp} has no source annotations for {absent}; AP is undefined for these classes")
    rng = random.Random(args.seed)
    stats = {"tiles": Counter(), "bg_tiles": Counter(), "boxes": defaultdict(Counter), "sizes": []}
    for split, items in splits.items():
        print(f"{split}: {len(items)} source images")
        process_split(split, items, out, args, rng, stats)

    with open(out / "splits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "split", "board_id"])
        for split, items in splits.items():
            for s in items:
                w.writerow([s[0], split, board_id(s[0])])

    yaml_text = f"path: {json.dumps(out.resolve().as_posix())}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    yaml_text += "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    (out / "pcb.yaml").write_text(yaml_text)

    sizes = np.array(stats["sizes"]) if stats["sizes"] else np.zeros(1)
    summary = {
        "split_mode": args.split_mode, "representation": args.representation,
        "board_ids": {sp: sorted({board_id(s[0]) for s in items}) for sp, items in splits.items()},
        "source_boxes_per_class": {sp: dict(Counter(CLASSES[b[0]] for s in items for b in s[2]))
                                   for sp, items in splits.items()},
        "dropped_ambiguous_tiles": dict(stats.get("dropped_ambiguous", {})),
        "source_images": {k: len(v) for k, v in splits.items()},
        "tiles": dict(stats["tiles"]),
        "background_tiles": dict(stats["bg_tiles"]),
        "boxes_per_class": {k: dict(v) for k, v in stats["boxes"].items()},
        "box_size_px_sqrt_area": {p: float(np.percentile(sizes, p)) for p in (5, 25, 50, 75, 95)},
        "tile": args.tile, "stride": args.stride, "seed": args.seed,
    }
    manifest = {"seed": args.seed, "split_mode": args.split_mode,
                "samples": [{"image": s[0], "split": sp, "board_id": board_id(s[0]),
                             "image_sha256": hashlib.sha256(s[1].read_bytes()).hexdigest()}
                            for sp, items in splits.items() for s in items]}
    manifest["annotations_sha256"] = hashlib.sha256(b"".join(
        p.read_bytes() for p in sorted(ann_dir.rglob("*.xml")))).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nDataset YAML: {out / 'pcb.yaml'}")


if __name__ == "__main__":
    main()
