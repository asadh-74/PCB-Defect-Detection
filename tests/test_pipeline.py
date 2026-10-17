"""End-to-end check of tiling inference + evaluation with a fake predictor (no torch needed)."""
import json
import sys

import numpy as np
from PIL import Image

from src import prepare_data
from src.evaluate import analyse
from src.infer import nms, tiled_predict
from tests.test_prepare_data import _make_raw


class OraclePredictor:
    """Returns the ground truth read from the label file of the tile it is shown (via lookup by pixels)."""
    name = "oracle"
    n_params = 0

    def __init__(self, data_root):
        import cv2
        from pathlib import Path
        from src.common import read_yolo_labels
        self.lookup = []
        for sp in ("val", "test"):
            for p in sorted((Path(data_root) / "images" / sp).glob("*.jpg")):
                img = cv2.imread(str(p))
                b, c = read_yolo_labels(Path(data_root) / "labels" / sp / f"{p.stem}.txt", 640, 640)
                self.lookup.append((img.sum(), b, c))

    def predict(self, img, conf=None):
        key = img.sum()
        for s, b, c in self.lookup:
            if s == key:
                return b.copy(), np.full(len(b), 0.9, np.float32), c.copy()
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)


def test_analyse_with_oracle_is_perfect(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "out"
    _make_raw(raw)
    monkeypatch.setattr(sys, "argv", ["p", "--raw", str(raw), "--out", str(out)])
    prepare_data.main()
    m = analyse(OraclePredictor(out), out, tmp_path / "res", conf=0.5, name="oracle", do_speed=False)
    assert np.isclose(m["map50"], 1.0) and np.isclose(m["map50_95"], 1.0)
    assert m["precision"] == 1.0 and m["recall"] == 1.0
    assert (tmp_path / "res" / "confusion_matrix.png").exists()
    assert json.loads((tmp_path / "res" / "metrics.json").read_text())["n_gt"] == m["n_gt"]


def test_tiled_predict_merges_duplicates():
    class Box40:  # reports a 40x40 box at (100,100) in whichever tile covers it
        def predict(self, crop, conf=None):
            return np.array([[100, 100, 140, 140]], np.float32), np.array([0.9], np.float32), np.array([2])
    img = np.zeros((900, 1400, 3), np.uint8)
    b, s, c = tiled_predict(Box40(), img, tile=640, stride=480)
    # every tile reports a box; after offsets + NMS, distinct locations remain but none overlap > 0.5
    assert len(b) >= 1
    keep = nms(b, s, c, 0.5)
    assert len(keep) == len(b)


def test_max_tiles_and_single_model_compare(tmp_path, monkeypatch):
    from src.evaluate import load_split
    from src import compare
    raw, out = tmp_path / "raw", tmp_path / "out"
    _make_raw(raw)
    monkeypatch.setattr(sys, "argv", ["p", "--raw", str(raw), "--out", str(out)])
    prepare_data.main()
    paths, gts = load_split(out, "test", max_tiles=10)
    assert len(paths) == 10 and any(len(g["cls"]) for g in gts) and any(len(g["cls"]) == 0 for g in gts)
    analyse(OraclePredictor(out), out, tmp_path / "results" / "oracle", conf=0.5, name="oracle", do_speed=False, max_tiles=40)
    monkeypatch.setattr(sys, "argv", ["compare", "--names", "oracle", "--results", str(tmp_path / "results")])
    compare.main()
    assert (tmp_path / "results" / "comparison.md").exists() and (tmp_path / "results" / "comparison.png").exists()
