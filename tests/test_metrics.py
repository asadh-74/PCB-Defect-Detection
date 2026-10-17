import numpy as np
from src.metrics import compute_map, evaluate_at_conf, iou_matrix, match_image, sweep_conf


def _gt(boxes, cls):
    return {"boxes": np.array(boxes, float).reshape(-1, 4), "cls": np.array(cls)}


def _pr(boxes, scores, cls):
    return {"boxes": np.array(boxes, float).reshape(-1, 4), "scores": np.array(scores, float), "cls": np.array(cls)}


def test_iou():
    a = [[0, 0, 10, 10]]
    assert np.isclose(iou_matrix(a, a)[0, 0], 1.0)
    assert np.isclose(iou_matrix(a, [[5, 0, 15, 10]])[0, 0], 1 / 3)
    assert iou_matrix(a, [[20, 20, 30, 30]])[0, 0] == 0


def test_perfect_predictions_give_map_one():
    gts = [_gt([[10, 10, 30, 30], [50, 50, 70, 70]], [0, 1]), _gt([[5, 5, 25, 25]], [1])]
    preds = [_pr([[10, 10, 30, 30], [50, 50, 70, 70]], [0.9, 0.8], [0, 1]), _pr([[5, 5, 25, 25]], [0.7], [1])]
    ap = compute_map(gts, preds, 3)
    assert np.isclose(np.nanmean(ap[0]), 1.0)
    assert np.isnan(ap[0, 2])  # class 2 has no GT, excluded from the mean


def test_shifted_boxes_drop_at_strict_iou():
    gts = [_gt([[0, 0, 20, 20]], [0])]
    preds = [_pr([[3, 0, 23, 20]], [0.9], [0])]  # IoU = 17/23 ~ 0.74
    ap = compute_map(gts, preds, 1)
    assert np.isclose(ap[0, 0], 1.0)           # passes at IoU 0.5
    assert ap[-1, 0] == 0.0                    # fails at IoU 0.95
    assert 0 < np.nanmean(ap) < 1


def test_wrong_class_is_fp_and_fn_and_confusion():
    gts = [_gt([[0, 0, 20, 20]], [0])]
    preds = [_pr([[0, 0, 20, 20]], [0.9], [1])]
    r = evaluate_at_conf(gts, preds, 2, conf=0.5)
    assert r["per_class"]["fn"][0] == 1 and r["per_class"]["fp"][1] == 1
    assert r["confusion"][0, 1] == 1  # true class 0 predicted as class 1


def test_missed_and_spurious():
    gts = [_gt([[0, 0, 10, 10]], [0]), _gt([], [])]
    preds = [_pr([], [], []), _pr([[0, 0, 10, 10]], [0.9], [0])]
    r = evaluate_at_conf(gts, preds, 1, conf=0.5)
    assert r["per_class"]["fn"][0] == 1 and r["per_class"]["fp"][0] == 1
    assert r["confusion"][1, 0] == 1 and r["confusion"][0, 1] == 1  # bg->class0, class0->bg


def test_duplicate_detection_counts_as_fp():
    gts = [_gt([[0, 0, 20, 20]], [0])]
    m = match_image(gts[0]["boxes"], gts[0]["cls"], np.array([[0, 0, 20, 20], [1, 1, 20, 20]]),
                    np.array([0.9, 0.8]), np.array([0, 0]), 0.5)
    assert list(m) == [0, -1]


def test_size_buckets_and_sweep():
    gts = [_gt([[0, 0, 10, 10], [0, 0, 100, 100]], [0, 0])]  # sizes 10 and 100
    preds = [_pr([[0, 0, 100, 100]], [0.9], [0])]
    r = evaluate_at_conf(gts, preds, 1, conf=0.5)
    assert r["size_recall"][0] == 0.0 and r["size_recall"][3] == 1.0
    conf, curve = sweep_conf(gts, preds, 1)
    assert 0.05 <= conf < 0.95 and len(curve) > 5
