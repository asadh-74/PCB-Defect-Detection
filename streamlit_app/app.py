"""PCB Defect Detector: upload a bare-PCB image, get defects found, counted, and exportable.

Run locally:   streamlit run streamlit_app/app.py
"""
import os
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from src.app_logic import (annotate, build_table, counts, crops_for, decode_image, detect,  # noqa: E402
                           encode_png, filter_conf)
from src.infer import PALETTE
from src.common import CLASSES  # noqa: E402

MODEL_DIR = Path(os.environ.get("PCB_MODEL_DIR", str(ROOT / "models")))
MODELS = {"YOLO": "yolo", "Faster R-CNN": "frcnn"}
DEFECT_INFO = {
    "missing_hole": "A drill hole that should be present is absent.",
    "mouse_bite": "Small notches along the edge of a copper trace.",
    "open_circuit": "A break in a trace that cuts the electrical path.",
    "short": "An unintended copper connection between two traces.",
    "spur": "A thin copper protrusion extending from a trace.",
    "spurious_copper": "Leftover copper in an area that should be empty.",
}

st.set_page_config(page_title="PCB Defect Detector", page_icon="🔍", layout="wide")


# ---------- model loading ----------
@st.cache_resource(show_spinner="Loading model...")
def get_predictor(kind: str, weights: str, stamp: int = 0):
    from src.predictors import load_predictor
    return load_predictor(kind, weights)


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def saved_weights(kind: str):
    p = MODEL_DIR / f"{kind}_best.pt"
    return p if p.exists() else None


def uploaded_weights_path(upload) -> str:
    data = upload.getvalue()
    tag = hashlib.md5(data).hexdigest()[:10]
    path = Path(tempfile.gettempdir()) / f"pcb_upload_{tag}.pt"
    if not path.exists():
        path.write_bytes(data)
    return str(path)


# ---------- sidebar ----------
available = [name for name, kind in MODELS.items() if saved_weights(kind)]
with st.sidebar:
    st.header("Settings")
    if available:
        model_name = st.selectbox("Model", available)
    else:
        model_name = "YOLO"
    kind = MODELS[model_name]

    metrics = read_json(MODEL_DIR / f"metrics_{kind}.json") or {}
    default_conf = float(min(max(metrics.get("conf", 0.25), 0.001), 0.95))
    conf = st.slider("Confidence threshold", 0.001, 0.95, default_conf, 0.001,
                     help="Higher = fewer false alarms but more missed defects. "
                          "The default is the value tuned on the validation split.")
    rescale = st.slider("Rescale image (%)", 25, 200, 100, 5,
                        help="The model was trained at native resolution on large boards. "
                             "If your photo is much smaller or larger, rescale so defects look similar in size.") / 100
    show_crops = st.checkbox("Show zoomed crops of detections", value=True)

    weights_path = saved_weights(kind)
    if weights_path is None:
        st.divider()
        st.caption("No bundled model found. Upload trained weights to test the app:")
        up = st.file_uploader("Weights (.pt)", type=["pt"], key="weights_upload")
        if up is not None:
            weights_path = uploaded_weights_path(up)
    st.divider()
    st.caption("Images are processed in memory and are not stored.")

st.title("PCB Defect Detector")
st.caption("Upload a bare printed circuit board image. The model finds six defect types: "
           "missing hole, mouse bite, open circuit, short, spur, and spurious copper.")

tab_detect, tab_about = st.tabs(["Detect", "About the model"])


# ---------- detection ----------
def run_detection(file_bytes, name, predictor, weights_key):
    model_stamp = Path(str(weights_path)).stat().st_mtime_ns
    key = (hashlib.md5(file_bytes).hexdigest(), weights_key, model_stamp, rescale)
    cache = st.session_state.setdefault("results", {})
    if key not in cache:
        bgr = decode_image(file_bytes, rescale)
        bar = st.progress(0.0, text=f"Scanning {name}...")
        b, s, c = detect(predictor, bgr, representation=metrics.get("representation", "tiled"), tile=metrics.get("tile", 640), stride=metrics.get("stride", 480), progress=lambda i, n: bar.progress(i / n, text=f"Scanning {name}: tile {i}/{n}"))
        bar.empty()
        cache[key] = (bgr, b, s, c)
        while len(cache) > 4:  # keep memory bounded
            cache.pop(next(iter(cache)))
    return cache[key]


def show_result(idx, name, bgr, boxes, scores, cls):
    b, s, c = filter_conf(boxes, scores, cls, conf)
    n = len(b)
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.image(cv2.cvtColor(annotate(bgr, b, s, c), cv2.COLOR_BGR2RGB), use_container_width=True,
                 caption=f"{name} | {bgr.shape[1]}x{bgr.shape[0]} px analysed")
    with right:
        if n == 0:
            st.success("No defects detected at this confidence threshold.")
        else:
            st.error(f"{n} defect{'s' if n != 1 else ''} detected")
        cnt = counts(c)
        cols = st.columns(3)
        for i, cname in enumerate(CLASSES):
            cols[i % 3].metric(cname.replace("_", " "), cnt[cname])
        df = build_table(b, s, c)
        if n:
            st.dataframe(df, use_container_width=True, hide_index=True, height=min(35 * n + 38, 300))
        d1, d2 = st.columns(2)
        d1.download_button("Annotated image (PNG)", encode_png(annotate(bgr, b, s, c)),
                           file_name=f"{Path(name).stem}_detected.png", mime="image/png", key=f"png{idx}")
        d2.download_button("Detections (CSV)", df.to_csv(index=False), file_name=f"{Path(name).stem}_detections.csv",
                           mime="text/csv", key=f"csv{idx}", disabled=n == 0)
    if show_crops and n:
        with st.expander(f"Zoomed detections (top {min(n, 12)} by confidence)", expanded=False):
            items = crops_for(bgr, b, s, c)
            cols = st.columns(6)
            for j, (img, cap) in enumerate(items):
                cols[j % 6].image(img, caption=cap, use_container_width=True)


with tab_detect:
    if weights_path is None:
        st.warning("No trained model is available yet.")
        st.markdown(
            "1. Train the model with the Colab notebook.\n"
            "2. Run `python -m src.export_for_app --model yolo --weights weights/yolo_best.pt`.\n"
            "3. Commit the `models/` folder, or upload the `.pt` file in the sidebar to try it now."
        )
    else:
        files = st.file_uploader("Upload PCB image(s)", type=["jpg", "jpeg", "png", "bmp"],
                                 accept_multiple_files=True)
        if not files:
            st.info("Upload one or more images to start. Large board images work best "
                    "(the model scans them in overlapping 640 px tiles).")
        else:
            try:
                predictor = get_predictor(kind, str(weights_path), Path(weights_path).stat().st_mtime_ns)
            except (ValueError, RuntimeError, OSError) as error:
                st.error(f"Could not load the PCB model: {error}")
                st.stop()
            for i, f in enumerate(files):
                st.subheader(f.name)
                try:
                    bgr, b, s, c = run_detection(f.getvalue(), f.name, predictor, f"{kind}:{weights_path}")
                except ValueError as e:
                    st.error(str(e))
                    continue
                show_result(i, f.name, bgr, b, s, c)
                st.divider()

# ---------- about ----------
with tab_about:
    st.subheader("How it works")
    st.caption(f"Exported model representation: {metrics.get('representation', 'tiled')}. Whole-image models use direct inference.")
    st.markdown(
        "- The board image is cut into overlapping 640 px tiles so small defects keep their native resolution.\n"
        "- A fine-tuned detector (YOLO, with Faster R-CNN as a baseline) runs on each tile.\n"
        "- Tile results are shifted back to board coordinates, border-cut boxes are dropped, and duplicates are merged with NMS.\n"
        "- The confidence slider filters the results instantly; the model runs once per image."
    )
    st.subheader("Class colour legend")
    for class_name, (blue, green, red) in zip(CLASSES, PALETTE):
        st.markdown(f'<span style="color:rgb({red},{green},{blue})">●</span> {class_name.replace("_", " ")}', unsafe_allow_html=True)
    st.subheader("Defect types")
    st.table({"Defect": [c.replace("_", " ") for c in CLASSES], "Description": [DEFECT_INFO[c] for c in CLASSES]})

    m = read_json(MODEL_DIR / f"metrics_{kind}.json")
    if m:
        st.subheader(f"Test-set results ({m.get('model', model_name)})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("mAP@0.5", "N/A" if m["map50"] is None else f"{m['map50']:.3f}")
        c2.metric("mAP@0.5:0.95", "N/A" if m["map50_95"] is None else f"{m['map50_95']:.3f}")
        c3.metric("Precision", f"{m['precision']:.3f}")
        c4.metric("Recall", f"{m['recall']:.3f}")
        split_note = ("Board prefixes are disjoint across train, validation and test."
                      if m.get("split_mode") == "board" else
                      "Source images are held out; board layouts may overlap across splits.")
        st.caption(f"{split_note} Evaluation unit: {m.get('evaluation_unit', 'tile')}. "
                   f"Representation: {m.get('representation', 'unknown')}.")
        if m.get("subset_evaluation"):
            st.warning("Smoke-test/subset metrics: do not use these as final results.")
        st.bar_chart({k: v for k, v in m["per_class_ap50"].items()}, y_label="AP@0.5")
        cm = MODEL_DIR / f"confusion_matrix_{kind}.png"
        if cm.exists():
            st.image(str(cm), caption="Confusion matrix (test split)", width=520)
    else:
        st.info("Test-set metrics will appear here after you export them with `src.export_for_app`.")

    st.subheader("Limitations")
    st.markdown(
        "- Trained on PKU-Market-PCB: bare boards with synthetically produced defects. Real production lines may look different.\n"
        "- Works on bare-board images. It is not trained for boards with components mounted.\n"
        "- This is an inspection aid, not a replacement for a human or certified AOI system."
    )
