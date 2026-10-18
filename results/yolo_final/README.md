# Recorded YOLO test evaluation

This folder is the user's actual Colab evaluation export, not a synthetic smoke test.
The checkpoint used is bundled at `models/yolo_best.pt`; its SHA256 matches `provenance.json`.
The app's exported metrics match `metrics.json`. Ground-truth and prediction files independently
reproduce the reported mAP, precision and recall using this repository's numpy scorer.

## Protocol and outcome

- Evaluation unit: original full board; tiled prediction with coordinate merging and class-aware NMS.
- Source split: board-prefix groups, seed 42. Train: 01/04/07/08/09/12; val: 06/10; test: 05/11.
- Source images: 480 train, 93 validation, 120 test. Prefix-to-physical-template mapping is not independently verified.
- Test ground-truth defects: 562. Operating confidence: 0.60, selected on validation macro-F1.
- mAP50: 0.9473501714192588; mAP50:95: 0.4614766542046836.
- Precision: 0.9602888086642599; recall: 0.9466192170818505; macro-F1: 0.9520856308043454.
- Class-aware counts: 532 true positives, 22 false positives, 30 false negatives.
- Mean board inference time: 219.74270589166355 ms on the recorded Tesla T4; CPU time is not measured here.

## Supported error observations

`per_class.csv` shows missing_hole has the lowest AP50 (0.859804) and short the lowest F1 (0.900000).
Spurious copper has 12 misses and no false positives at this threshold. Mouse bite and open circuit
have recall 1.0 on this test set, each with two false positives; this is not a claim of perfect generalization.
The class-agnostic confusion matrix has one short-to-spur match and one spurious-copper-to-spur match.
Its matching pass differs from the class-aware FP/FN calculation; do not sum them interchangeably.

`size_by_class.csv` and metrics.json show 176 ground-truth boxes in the 32–64 pixel sqrt(area) bucket
with recall 1.0, and 386 boxes in the >=64 bucket with recall 0.9222797927461139. There are no boxes
below 32 pixels in this test set, so recall in those buckets is undefined, not zero. All 30 class-aware
misses fall in the >=64 bucket. Size is measured in original board coordinates, not resized inputs.

All 120 evaluated images contain ground-truth defects. Clean-board false-positive performance and
specificity are not measured. The observed 22 false detections correspond to 0.18333333333333332 FP
per evaluated defect-containing board. This rate must not be described as a clean-board false-alarm rate.
Board-mode errors have no contributing tile coordinates; separate tile diagnostics are still needed
to attribute errors to tile contexts. Example selection is sampled by the evaluator; images are illustrative.

## Reproducibility and limits

`training_args.yaml` and `training_history.csv` are the original submitted run files. The history has
60 rows. Training was resumed from a Drive checkpoint; the saved YAML describes the resumed settings,
not a complete audit of every earlier training invocation. The pasted training log identified Ultralytics
8.4.182, while evaluation provenance records 8.4.181. Retain that distinction rather than claiming an
identical training/evaluation environment. The model loads using the repository's pinned evaluation version.

No dataset was included in this archive. Original image hashes, annotations aggregate hash and membership
are retained in manifest.json; full image-based reruns require obtaining that same dataset and restoring
its recorded split. This is one partition of one dataset; no production-line, multi-seed, image-split,
augmentation-ablation or full-image-ablation result is claimed. The shared scorer is COCO-style, not the
complete official COCO evaluator. Do not tune on this test result and then call the same test independent.
