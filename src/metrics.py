"""Detection metrics in plain numpy, so both models are scored by identical code.

Data layout (one entry per image):
    gts[i]   = {"boxes": (N,4) xyxy, "cls": (N,)}
    preds[i] = {"boxes": (M,4) xyxy, "scores": (M,), "cls": (M,)}
"""
import numpy as np

from .common import SIZE_EDGES


def iou_matrix(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match_image(gt_boxes, gt_cls, p_boxes, p_scores, p_cls, iou_thr, class_aware=True):
    """Greedy matching, highest score first. Returns array of matched GT index per pred (-1 = none)."""
    n_p = len(p_boxes)
    match = -np.ones(n_p, dtype=np.int64)
    if n_p == 0 or len(gt_boxes) == 0:
        return match
    ious = iou_matrix(p_boxes, gt_boxes)
    gt_cls = np.asarray(gt_cls)
    p_cls = np.asarray(p_cls)
    taken = np.zeros(len(gt_boxes), dtype=bool)
    for i in np.argsort(-np.asarray(p_scores), kind="stable"):
        cand = ious[i].copy()
        cand[taken] = -1
        if class_aware:
            cand[gt_cls != p_cls[i]] = -1
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            match[i] = j
            taken[j] = True
    return match


def ap_from_pr(recall, precision):
    """COCO-style 101-point interpolated AP (recall/precision sorted by descending score)."""
    recall, precision = np.asarray(recall, float), np.asarray(precision, float)
    if len(recall) == 0:
        return 0.0
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    q = np.where(idx < len(envelope), envelope[np.minimum(idx, len(envelope) - 1)], 0.0)
    return float(q.mean())


def compute_map(gts, preds, n_classes, iou_thrs=None, max_det=100):
    """Returns ap array of shape [n_thr, n_classes] (nan where a class has no GT)."""
    if iou_thrs is None:
        iou_thrs = np.linspace(0.5, 0.95, 10)
    n_gt = np.zeros(n_classes)
    for g in gts:
        for c in np.asarray(g["cls"]):
            n_gt[int(c)] += 1
    ap = np.full((len(iou_thrs), n_classes), np.nan)
    for ti, thr in enumerate(iou_thrs):
        scores, classes, tps = [], [], []
        for g, p in zip(gts, preds):
            order = np.argsort(-np.asarray(p["scores"]), kind="stable")[:max_det]
            pb = np.asarray(p["boxes"])[order]
            ps = np.asarray(p["scores"])[order]
            pc = np.asarray(p["cls"])[order]
            m = match_image(g["boxes"], g["cls"], pb, ps, pc, thr)
            scores.append(ps)
            classes.append(pc)
            tps.append(m >= 0)
        if not scores:
            continue
        scores, classes, tps = np.concatenate(scores), np.concatenate(classes), np.concatenate(tps)
        for c in range(n_classes):
            if n_gt[c] == 0:
                continue
            sel = classes == c
            if not sel.any():
                ap[ti, c] = 0.0
                continue
            o = np.argsort(-scores[sel], kind="stable")
            tp = tps[sel][o].astype(float)
            ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
            ap[ti, c] = ap_from_pr(ctp / n_gt[c], ctp / np.maximum(ctp + cfp, 1e-9))
    return ap


def evaluate_at_conf(gts, preds, n_classes, conf, iou_thr=0.5, size_edges=SIZE_EDGES):
    """Precision/recall/F1 per class at a fixed confidence, recall by object size, confusion matrix,
    and the list of errors (for visual error analysis)."""
    tp = np.zeros(n_classes)
    fp = np.zeros(n_classes)
    fn = np.zeros(n_classes)
    n_buckets = len(size_edges) + 1
    b_total = np.zeros(n_buckets)
    b_found = np.zeros(n_buckets)
    cm = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int64)  # [true, pred], last = background
    fn_list, fp_list = [], []

    for idx, (g, p) in enumerate(zip(gts, preds)):
        keep = np.asarray(p["scores"]) >= conf
        pb, ps, pc = np.asarray(p["boxes"])[keep], np.asarray(p["scores"])[keep], np.asarray(p["cls"])[keep]
        gb, gc = np.asarray(g["boxes"]).reshape(-1, 4), np.asarray(g["cls"])

        m = match_image(gb, gc, pb, ps, pc, iou_thr, class_aware=True)
        gt_found = np.zeros(len(gb), dtype=bool)
        for i, j in enumerate(m):
            if j >= 0:
                tp[pc[i]] += 1
                gt_found[j] = True
            else:
                fp[pc[i]] += 1
                fp_list.append((idx, i))
        for j in np.where(~gt_found)[0]:
            fn[gc[j]] += 1
            fn_list.append((idx, int(j)))

        if len(gb):
            sizes = np.sqrt((gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1]))
            bk = np.digitize(sizes, size_edges)
            for b, found in zip(bk, gt_found):
                b_total[b] += 1
                b_found[b] += int(found)

        ma = match_image(gb, gc, pb, ps, pc, iou_thr, class_aware=False)
        used = set()
        for i, j in enumerate(ma):
            if j >= 0:
                cm[gc[j], pc[i]] += 1
                used.add(int(j))
            else:
                cm[n_classes, pc[i]] += 1
        for j in range(len(gb)):
            if j not in used:
                cm[gc[j], n_classes] += 1

    prec = tp / np.maximum(tp + fp, 1e-9)
    rec = tp / np.maximum(tp + fn, 1e-9)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    present = (tp + fn) > 0
    return {
        "conf": float(conf),
        "per_class": {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn},
        "precision_micro": float(tp.sum() / max(tp.sum() + fp.sum(), 1e-9)),
        "recall_micro": float(tp.sum() / max(tp.sum() + fn.sum(), 1e-9)),
        "f1_macro": float(f1[present].mean()) if present.any() else 0.0,
        "size_recall": np.where(b_total > 0, b_found / np.maximum(b_total, 1), np.nan),
        "size_counts": b_total,
        "confusion": cm,
        "fn_list": fn_list,
        "fp_list": fp_list,
    }


def sweep_conf(gts, preds, n_classes, grid=None, iou_thr=0.5):
    """Pick the confidence threshold with the best macro-F1 (use on VALIDATION data only)."""
    grid = np.arange(0.05, 0.95, 0.05) if grid is None else grid
    best_c, best_f1, curve = float(grid[0]), -1.0, []
    for c in grid:
        f1 = evaluate_at_conf(gts, preds, n_classes, c, iou_thr)["f1_macro"]
        curve.append((float(c), f1))
        if f1 > best_f1:
            best_c, best_f1 = float(c), f1
    return best_c, curve
