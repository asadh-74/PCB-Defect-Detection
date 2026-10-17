"""UI-independent logic for the Streamlit app (so it can be unit-tested without Streamlit)."""
import io

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .common import CLASSES
from .infer import draw, tiled_predict

Image.MAX_IMAGE_PIXELS = None
MAX_SIDE = 10000  # safety cap for very large uploads


def decode_image(file_bytes: bytes, rescale: float = 1.0) -> np.ndarray:
    """Bytes -> BGR uint8 array. Applies EXIF rotation and an optional rescale factor."""
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(file_bytes))).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not read this file as an image: {e}")
    w, h = img.size
    factor = rescale
    if max(w, h) * factor > MAX_SIDE:
        factor = MAX_SIDE / max(w, h)
    if abs(factor - 1.0) > 1e-6:
        img = img.resize((max(1, round(w * factor)), max(1, round(h * factor))), Image.LANCZOS)
    return np.ascontiguousarray(np.array(img)[:, :, ::-1])


def detect(predictor, bgr, floor=0.001, progress=None, representation="tiled", tile=640, stride=480):
    """Run tiled inference once at a low confidence floor; the UI slider filters afterwards."""
    if representation == "full":
        return predictor.predict(bgr, conf=floor)
    return tiled_predict(predictor, bgr, tile=tile, stride=stride, conf=floor, progress=progress)


def filter_conf(boxes, scores, cls, conf):
    keep = scores >= conf
    return boxes[keep], scores[keep], cls[keep]


def annotate(bgr, boxes, scores, cls):
    scale = max(bgr.shape[:2]) / 1600.0
    return draw(bgr, boxes, scores, cls, thickness=max(2, int(round(2 * scale))),
                font_scale=0.45 * max(1.0, scale))


def build_table(boxes, scores, cls) -> pd.DataFrame:
    rows = []
    for b, s, c in zip(boxes, scores, cls):
        rows.append({"defect": CLASSES[int(c)], "confidence": round(float(s), 3),
                     "x": int(b[0]), "y": int(b[1]),
                     "width": int(b[2] - b[0]), "height": int(b[3] - b[1])})
    df = pd.DataFrame(rows, columns=["defect", "confidence", "x", "y", "width", "height"])
    return df.sort_values("confidence", ascending=False).reset_index(drop=True)


def counts(cls) -> dict:
    return {name: int((np.asarray(cls) == i).sum()) for i, name in enumerate(CLASSES)}


def crops_for(bgr, boxes, scores, cls, top=12, pad=24, min_side=160):
    """Zoomed crops of the most confident detections: list of (RGB array, caption)."""
    H, W = bgr.shape[:2]
    out = []
    for i in np.argsort(-np.asarray(scores))[:top]:
        x0, y0, x1, y1 = boxes[i]
        x0, y0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
        x1, y1 = min(W, int(x1) + pad), min(H, int(y1) + pad)
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        f = max(1.0, min_side / max(crop.shape[:2]))
        if f > 1.0:
            crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        out.append((cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), f"{CLASSES[int(cls[i])]} {scores[i]:.2f}"))
    return out


def encode_png(bgr) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()
