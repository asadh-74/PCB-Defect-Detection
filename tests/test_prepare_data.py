import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src import prepare_data
from src.common import CLASSES, positions


def _make_raw(root: Path, n_per_class=6, W=1400, H=900, seed=0):
    rng = np.random.default_rng(seed)
    (root / "Annotations").mkdir(parents=True)
    (root / "images").mkdir(parents=True)
    for ci, cname in enumerate(CLASSES):
        (root / "Annotations" / cname).mkdir()
        (root / "images" / cname).mkdir()
        for k in range(n_per_class):
            stem = f"{ci:02d}_{cname}_{k:02d}"
            Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(root / "images" / cname / f"{stem}.jpg")
            objs = ""
            for _ in range(3):
                x0, y0 = int(rng.integers(0, W - 60)), int(rng.integers(0, H - 60))
                objs += (f"<object><name>{cname}</name><bndbox><xmin>{x0}</xmin><ymin>{y0}</ymin>"
                         f"<xmax>{x0 + 40}</xmax><ymax>{y0 + 40}</ymax></bndbox></object>")
            (root / "Annotations" / cname / f"{stem}.xml").write_text(f"<annotation>{objs}</annotation>")


def test_positions_cover_edges():
    assert positions(1400, 640, 480) == [0, 480, 760]
    assert positions(500, 640, 480) == [0]
    assert positions(640, 640, 480) == [0]


def test_pipeline_no_leakage_and_valid_labels(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "out"
    _make_raw(raw)
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--raw", str(raw), "--out", str(out)])
    prepare_data.main()

    rows = list(csv.DictReader(open(out / "splits.csv")))
    split_of = {r["image"]: r["split"] for r in rows}
    assert len(split_of) == 36
    assert all(sum(1 for s in split_of.values() if s == sp) >= len(CLASSES) for sp in ("train", "val", "test"))

    # no source image may appear in two splits (tiles are named <stem>_x.._y..)
    seen = {}
    for sp in ("train", "val", "test"):
        for p in (out / "images" / sp).glob("*.jpg"):
            stem = p.stem.rsplit("_x", 1)[0]
            assert split_of[stem] == sp
            seen.setdefault(stem, set()).add(sp)
            assert Image.open(p).size == (640, 640)
            lab = out / "labels" / sp / f"{p.stem}.txt"
            assert lab.exists()
            for line in lab.read_text().splitlines():
                c, cx, cy, w, h = line.split()
                assert 0 <= int(c) < len(CLASSES)
                assert all(0 <= float(v) <= 1 for v in (cx, cy, w, h))
                assert float(cx) - float(w) / 2 >= -1e-5 and float(cx) + float(w) / 2 <= 1 + 1e-5
    assert all(len(v) == 1 for v in seen.values())
    assert (out / "pcb.yaml").exists() and (out / "stats.json").exists()
