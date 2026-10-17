"""Tiled inference on full-size PCB images, plus drawing helpers.

    python -m src.infer --model yolo --weights weights/yolo_best.pt --image board.jpg --out out.jpg
"""
import argparse

import cv2
import numpy as np

from .common import CLASSES, positions

PALETTE = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (255, 0, 255), (255, 0, 0), (0, 200, 0)]


def nms(boxes, scores, cls, iou_thr=0.5):
    """Class-aware NMS in numpy."""
    from .metrics import iou_matrix
    keep = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        order = idx[np.argsort(-scores[idx])]
        while len(order):
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
            order = order[1:][ious < iou_thr]
    return np.array(sorted(keep), dtype=np.int64)


def tiled_predict(predictor, img_bgr, tile=640, stride=480, conf=0.25, nms_iou=0.5, edge=2, progress=None):
    """progress: optional callable(done_tiles, total_tiles), used by the Streamlit progress bar."""
    if img_bgr is None or img_bgr.ndim != 3:
        raise ValueError("Expected a decoded colour image")
    if edge > 0 and stride >= tile:
        raise ValueError("Border rejection requires overlapping tiles (stride < tile)")
    H, W = img_bgr.shape[:2]
    B, S, C = [], [], []
    ys, xs = positions(H, tile, stride), positions(W, tile, stride)
    total, done = len(ys) * len(xs), 0
    for y0 in ys:
        for x0 in xs:
            done += 1
            if progress:
                progress(done, total)
            crop = img_bgr[y0:y0 + tile, x0:x0 + tile]
            ch, cw = crop.shape[:2]
            # Match preparation: images smaller than a tile are padded, not upscaled.
            if ch < tile or cw < tile:
                crop = cv2.copyMakeBorder(crop, 0, tile-ch, 0, tile-cw, cv2.BORDER_CONSTANT, value=0)
            b, s, c = predictor.predict(crop, conf=conf)
            b = b.copy()
            b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, cw)
            b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, ch)
            valid = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
            b, s, c = b[valid], s[valid], c[valid]
            if len(b) == 0:
                continue
            # drop boxes cut by an interior tile border; the neighbouring tile sees them whole
            ok = np.ones(len(b), bool)
            if x0 > 0:
                ok &= b[:, 0] > edge
            if y0 > 0:
                ok &= b[:, 1] > edge
            if x0 + cw < W:
                ok &= b[:, 2] < cw - edge
            if y0 + ch < H:
                ok &= b[:, 3] < ch - edge
            b, s, c = b[ok] + np.array([x0, y0, x0, y0], np.float32), s[ok], c[ok]
            B.append(b); S.append(s); C.append(c)
    if not B:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    B, S, C = np.concatenate(B), np.concatenate(S), np.concatenate(C)
    k = nms(B, S, C, nms_iou)
    return B[k], S[k], C[k]


def draw(img_bgr, boxes, scores=None, cls=None, color=None, thickness=2, names=CLASSES, font_scale=0.45):
    out = img_bgr.copy()
    for i, b in enumerate(np.asarray(boxes).reshape(-1, 4)):
        c = int(cls[i]) if cls is not None else 0
        col = color or PALETTE[c % len(PALETTE)]
        x0, y0, x1, y1 = (int(round(v)) for v in b)
        cv2.rectangle(out, (x0, y0), (x1, y1), col, thickness)
        label = names[c] if scores is None else f"{names[c]} {scores[i]:.2f}"
        cv2.putText(out, label, (x0, max(y0 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, max(1, int(font_scale + 0.5)), cv2.LINE_AA)
    return out


def main():
    from .predictors import load_predictor
    ap = argparse.ArgumentParser()
    ap.add_argument("--representation", choices=["tiled", "full"], default="tiled")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--stride", type=int, default=480)
    ap.add_argument("--model", choices=["yolo", "frcnn"], default="yolo")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="prediction.jpg")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    pred = load_predictor(args.model, args.weights)
    img = cv2.imread(args.image)
    if img is None:
        raise ValueError(f"Could not read image: {args.image}")
    b, s, c = (pred.predict(img, conf=args.conf) if args.representation == "full" else
               tiled_predict(pred, img, tile=args.tile, stride=args.stride, conf=args.conf))
    if not cv2.imwrite(args.out, draw(img, b, s, c, thickness=3)):
        raise OSError(f"Could not write image: {args.out}")
    print(f"{len(b)} defects found -> {args.out}")
    for k in sorted(set(c.tolist())):
        print(f"  {CLASSES[k]}: {(c == k).sum()}")


if __name__ == "__main__":
    main()
