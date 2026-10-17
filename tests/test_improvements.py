"""Regression tests for split integrity, fair evaluation, and resumable execution."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.common import CLASSES, positions
from src.prepare_data import board_id, grouped_split, main as prepare
from src.evaluate import BoardPredictor, analyse, measure_speed, save_error_examples
from src.infer import nms, tiled_predict
from src.metrics import evaluate_at_conf
from src.train_yolo import augmentation_options


def samples(n=10):
    return [(f"{b:02d}_{c}_01", Path("unused"), [(i, 1, 1, 10, 10)])
            for b in range(n) for i, c in enumerate(CLASSES)]


def test_grouped_split_disjoint_repeatable_and_order_independent():
    data = samples()
    split = grouped_split(data, (.7, .15, .15), 42)
    assert split == grouped_split(list(reversed(data)), (.7, .15, .15), 42)
    boards = {sp: {board_id(s[0]) for s in rows} for sp, rows in split.items()}
    assert sum(map(len, boards.values())) == 10
    assert not boards["train"] & boards["val"]
    assert not boards["train"] & boards["test"]
    assert not boards["val"] & boards["test"]
    assert sum(map(len, split.values())) == len(data)
    assert all({b[0] for s in rows for b in s[2]} == set(range(6)) for rows in split.values())


def test_invalid_groups_and_positions():
    with pytest.raises(ValueError):
        board_id("invalid")
    with pytest.raises(ValueError):
        grouped_split(samples(2), (.7, .15, .15), 42)
    for tile, stride in [(0, 1), (640, 0), (640, 700), (-1, 1)]:
        with pytest.raises(ValueError):
            positions(900, tile, stride)


def test_full_representation_and_manifest_with_spaces(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw (PCB data)", tmp_path / "output (full image)"
    (raw / "images").mkdir(parents=True)
    (raw / "Annotations").mkdir()
    for stem, _, _ in samples(3):
        Image.new("RGB", (100, 80), "green").save(raw / "images" / f"{stem}.jpg")
        classname = stem.split("_", 1)[1].rsplit("_", 1)[0]
        (raw / "Annotations" / f"{stem}.xml").write_text(
            f"<annotation><object><name>{classname}</name><bndbox><xmin>10</xmin><ymin>20</ymin>"
            "<xmax>30</xmax><ymax>40</ymax></bndbox></object></annotation>")
    monkeypatch.setattr(sys, "argv", ["prepare", "--raw", str(raw), "--out", str(out),
                                     "--split-mode", "board", "--representation", "full"])
    prepare()
    import yaml
    assert yaml.safe_load((out / "pcb.yaml").read_text())["path"] == out.resolve().as_posix()
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["samples"]) == 18
    assert len(manifest["samples"][0]["image_sha256"]) == 64
    image = next((out / "images" / "test").glob("*.jpg"))
    assert Image.open(image).size == (100, 80)
    assert (out / "boards" / "images" / "test" / image.name).exists()


class EmptyPredictor:
    name = "empty"
    n_params = 0
    def predict(self, image, conf=0.001):
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)


def test_speed_tiny_split_and_empty(tmp_path):
    image = tmp_path / "tiny.jpg"
    Image.new("RGB", (20, 20)).save(image)
    ms, fps = measure_speed(EmptyPredictor(), [image])
    assert np.isfinite(ms) and ms > 0 and fps > 0
    assert measure_speed(EmptyPredictor(), []) == (None, None)


def test_error_examples_do_not_overwrite_and_context_export(tmp_path):
    image = tmp_path / "01_short_01_x0_y480.jpg"
    Image.new("RGB", (100, 100)).save(image)
    gt = {"boxes": np.array([[10, 10, 20, 20], [30, 30, 40, 40]], np.float32), "cls": np.array([3, 3])}
    pred = {"boxes": np.array([[0, 1, 5, 6]], np.float32), "scores": np.array([.9]), "cls": np.array([1])}
    res = evaluate_at_conf([gt], [pred], 6, .5)
    save_error_examples(tmp_path / "examples", [image], [gt], [pred], res, .5)
    assert len(list((tmp_path / "examples").glob("FN*.jpg"))) == 2
    from src.evaluate import save_error_records
    save_error_records(tmp_path, [image], [gt], [pred], res, .5)
    rows = json.loads((tmp_path / "errors.json").read_text())
    fp = next(r for r in rows if r["kind"] == "FP")
    assert fp["tile_y"] == 480 and fp["near_edge_16px"]
    assert fp["source_image"] == "01_short_01"


def test_nms_and_border_recovery():
    b = np.array([[100, 100, 140, 140]] * 3, np.float32)
    assert nms(b, np.array([.9, .8, .7]), np.array([1, 1, 2])).tolist() == [0, 2]
    class BorderObject:
        def __init__(self):
            self.calls = 0
        def predict(self, image, conf=.001):
            # First tile's border-cut box; next tile sees the complete same object.
            box = [620, 100, 640, 140] if self.calls == 0 else [140, 100, 180, 140]
            self.calls += 1
            return np.array([box], np.float32), np.array([.9]), np.array([3])
    boxes, _, _ = tiled_predict(BorderObject(), np.zeros((640, 1120, 3), np.uint8))
    assert boxes.tolist() == [[620, 100, 660, 140]]


def test_board_predictor_dispatch_and_no_augmentation():
    class Counting(EmptyPredictor):
        def __init__(self):
            self.calls = 0
        def predict(self, *a, **kw):
            self.calls += 1
            return super().predict(*a, **kw)
    image = np.zeros((640, 1120, 3), np.uint8)
    predictor = Counting()
    BoardPredictor(predictor, "full").predict(image)
    assert predictor.calls == 1
    BoardPredictor(predictor, "tiled").predict(image)
    assert predictor.calls == 3
    options = augmentation_options(False)
    assert all(options[k] == 0 for k in ("mosaic", "fliplr", "translate", "scale", "hsv_h"))


def test_pipeline_fail_fast_and_resume_invalidation(tmp_path, monkeypatch):
    import run_pipeline as pipeline
    (tmp_path / "run_pipeline.py").write_text("# test source")
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "have_cuda", lambda: False)
    monkeypatch.setattr(sys, "argv", ["run", "--profile", "cpu", "--raw", str(tmp_path / "raw")])
    calls = []
    def execute(cmd, cwd):
        calls.append(cmd)
        module = cmd[2]
        def write(path):
            path = tmp_path / path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        if module == "src.prepare_data":
            data = cmd[cmd.index("--out") + 1]
            write(f"{data}/pcb.yaml"); write(f"{data}/stats.json"); write(f"{data}/manifest.json")
        elif module == "src.train_yolo":
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(pipeline.subprocess, "run", execute)
    with pytest.raises(SystemExit, match="STEP FAILED: Train YOLO"):
        pipeline.main()
    assert len(calls) == 2
    calls.clear()
    with pytest.raises(SystemExit):
        pipeline.main()
    assert len(calls) == 1 and calls[0][2] == "src.train_yolo"
    # Changed source or split invalidates preparation rather than trusting stale outputs.
    monkeypatch.setattr(sys, "argv", ["run", "--profile", "cpu", "--raw", str(tmp_path / "raw"), "--split-mode", "image"])
    calls.clear()
    with pytest.raises(SystemExit):
        pipeline.main()
    assert calls[0][2] == "src.prepare_data"


def test_validation_threshold_is_not_retuned_on_test(tmp_path):
    from src.common import CLASSES
    root = tmp_path / "data"
    (root / "stats.json").parent.mkdir()
    (root / "stats.json").write_text(json.dumps({"representation": "full", "split_mode": "board"}))
    for split, colour in [("val", 100), ("test", 200)]:
        image_dir = root / "boards" / "images" / split
        labels = root / "boards" / "labels" / split
        image_dir.mkdir(parents=True); labels.mkdir(parents=True)
        Image.new("RGB", (100, 100), (colour, colour, colour)).save(image_dir / "01_short_01.jpg")
        (labels / "01_short_01.txt").write_text("0 0.15 0.15 0.1 0.1")
    class DifferentScores:
        name = "threshold-test"
        n_params = 0
        def predict(self, image, conf=.001):
            if image[0, 0, 0] < 150:
                return np.array([[10,10,20,20], [50,50,60,60]], np.float32), np.array([.8,.6]), np.array([0,0])
            return np.array([[10,10,20,20]], np.float32), np.array([.4]), np.array([0])
    result = analyse(DifferentScores(), root, tmp_path / "res", unit="board", do_speed=False)
    assert result["conf"] > .6 and result["recall"] == 0
    assert result["threshold_source"] == "validation"
    assert result["per_class_ap50"][CLASSES[1]] is None
    assert result["fps"] is None
    assert (tmp_path / "res" / "confidence_sweep.json").exists()
    assert (tmp_path / "res" / "size_by_class.csv").exists()
    # Strict JSON parsing: undefined AP is null, never a NaN token.
    json.loads((tmp_path / "res" / "metrics.json").read_text(), parse_constant=lambda x: pytest.fail(x))


def test_small_input_padding_drops_padding_only_predictions():
    class PaddingBoxes:
        def predict(self, image, conf=.001):
            assert image.shape[:2] == (640, 640)
            return np.array([[10,10,20,20], [200,200,220,220]], np.float32), np.array([.9,.8]), np.array([0,0])
    boxes, _, _ = tiled_predict(PaddingBoxes(), np.zeros((80, 100, 3), np.uint8))
    assert boxes.tolist() == [[10,10,20,20]]


def test_ambiguous_fragment_dropped_even_beside_kept_box(tmp_path):
    from collections import Counter, defaultdict
    from src.prepare_data import process_split
    path = tmp_path / "raw.jpg"
    Image.new("RGB", (112,64)).save(path)
    args = SimpleNamespace(tile=64, stride=48, representation="tiled", bg_ratio=1.0)
    stats = {"tiles": Counter(), "bg_tiles": Counter(), "boxes": defaultdict(Counter), "sizes": []}
    import random
    process_split("test", [("01_short_01", path, [(3,10,10,20,20), (3,60,10,80,30)])],
                  tmp_path / "out", args, random.Random(42), stats)
    assert not (tmp_path / "out/images/test/01_short_01_x0_y0.jpg").exists()
    assert (tmp_path / "out/images/test/01_short_01_x48_y0.jpg").exists()
    assert stats["dropped_ambiguous"]["test"] == 1
