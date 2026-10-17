from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import io

from src.app_logic import (annotate, build_table, counts, crops_for, decode_image, detect, encode_png,
                           filter_conf)


class FixedBoxes:
    def predict(self, crop, conf=None):
        h, w = crop.shape[:2]
        return (np.array([[50, 60, 90, 100]], np.float32), np.array([0.8], np.float32), np.array([3]))


def _png(w, h):
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def test_decode_and_rescale():
    assert decode_image(_png(400, 300)).shape == (300, 400, 3)
    assert decode_image(_png(400, 300), 0.5).shape == (150, 200, 3)
    with pytest.raises(ValueError):
        decode_image(b"not an image")


def test_detect_filter_table_outputs():
    bgr = decode_image(_png(1400, 900))
    seen = []
    b, s, c = detect(FixedBoxes(), bgr, progress=lambda i, n: seen.append((i, n)))
    assert seen[-1][0] == seen[-1][1] and len(b) >= 1
    b2, s2, c2 = filter_conf(b, s, c, 0.9)
    assert len(b2) == 0
    df = build_table(b, s, c)
    assert list(df.columns) == ["defect", "confidence", "x", "y", "width", "height"]
    assert (df["defect"] == "short").all()
    assert counts(c)["short"] == len(b)
    out = annotate(bgr, b, s, c)
    assert out.shape == bgr.shape
    assert len(crops_for(bgr, b, s, c)) == min(len(b), 12)
    assert encode_png(out)[:4] == b"\x89PNG"


def test_empty_results_are_handled():
    bgr = decode_image(_png(700, 700))
    e = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))
    assert build_table(*e).empty
    assert crops_for(bgr, *e) == []
    assert annotate(bgr, *e).shape == bgr.shape


def test_streamlit_app_boots_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("PCB_MODEL_DIR", str(tmp_path / "empty_models"))
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(str(Path(__file__).resolve().parent.parent / "streamlit_app" / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("No trained model" in w.value for w in at.warning)


def test_streamlit_app_displays_exported_protocol(tmp_path, monkeypatch):
    import json
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    # No model is deserialized unless a board is uploaded; this test checks metadata rendering.
    (model_dir / "yolo_best.pt").touch()
    (model_dir / "metrics_yolo.json").write_text(json.dumps({
        "conf": .25, "map50": None, "map50_95": None, "precision": 0, "recall": 0,
        "split_mode": "image", "evaluation_unit": "board", "representation": "full",
        "subset_evaluation": True, "per_class_ap50": {c: None for c in __import__("src.common", fromlist=["CLASSES"]).CLASSES}}))
    monkeypatch.setenv("PCB_MODEL_DIR", str(model_dir))
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(str(Path(__file__).resolve().parent.parent / "streamlit_app" / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("layouts may overlap" in item.value for item in at.caption)
    assert any("Smoke-test/subset" in item.value for item in at.warning)
