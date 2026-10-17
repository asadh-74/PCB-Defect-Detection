"""Shared constants and small helpers."""
from pathlib import Path
import numpy as np

CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SIZE_EDGES = (16, 32, 64)  # box size = sqrt(w*h) in pixels, on the training tile scale
SIZE_LABELS = ["<16px", "16-32px", "32-64px", ">=64px"]


def norm_name(s: str) -> str:
    return s.strip().lower().replace("-", "_").replace(" ", "_")


def positions(length: int, tile: int, stride: int):
    """Start offsets for sliding-window tiles. The last tile is aligned to the far edge."""
    if length <= 0 or tile <= 0 or stride <= 0 or stride > tile:
        raise ValueError("Require length > 0 and 0 < stride <= tile")
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, stride))
    if pos[-1] != length - tile:
        pos.append(length - tile)
    return pos


def read_yolo_labels(path: Path, w: int, h: int):
    """Read a YOLO .txt label file. Returns (boxes Nx4 absolute xyxy float32, cls N int)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    rows = np.loadtxt(path, ndmin=2)
    cls = rows[:, 0].astype(np.int64)
    cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1).astype(np.float32)
    return boxes, cls
