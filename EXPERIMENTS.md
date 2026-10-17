# Feasible experiment plan

**Status: NOT RUN on PKU-Market-PCB. No performance claims are made.**

Use one free Colab/Kaggle GPU. Run the quick profile first, then the three core experiments below.
A free session's available duration and GPU model vary; measure the pilot before budgeting the rest.
The commands use 20 epochs and YOLOv8n as an initial fixed budget, not a guarantee of convergence.
Use the same budget for all core comparisons. If the pilot is too slow, decide a shorter common
budget before inspecting test results. Save the actual epochs completed and wall time.

## Core runs (PowerShell, project root)

Replace the raw path with the actual dataset root. The quotes support spaces and parentheses.

```powershell
python run_pipeline.py --profile gpu --run-name grouped_tiles --raw "C:\PCB Data (Bootcamp)\PCB_DATASET" --split-mode board --representation tiled --model yolov8n.pt --epochs 20 --batch 8 --skip-frcnn
python run_pipeline.py --profile gpu --run-name grouped_full --raw "C:\PCB Data (Bootcamp)\PCB_DATASET" --split-mode board --representation full --model yolov8n.pt --epochs 20 --batch 8 --skip-frcnn
python run_pipeline.py --profile gpu --run-name grouped_noaug --raw "C:\PCB Data (Bootcamp)\PCB_DATASET" --split-mode board --representation tiled --augmentation off --model yolov8n.pt --epochs 20 --batch 8 --skip-frcnn
```

`--representation full` preserves complete board files and normalized labels. YOLO performs its
normal aspect-ratio-preserving resizing/letterboxing to imgsz=640; it does not crop these into tiles.
Both representations are evaluated on the SAME original-resolution boards and original boxes:
the tiled model uses tiled inference plus NMS; the full-image model uses direct whole-board prediction.
Both tune their operating confidence on their validation board predictions only.
Compare `manifest.json` source memberships before comparing runs. Do not compare tile mAP against board mAP.

Equal epochs mean equal passes over each representation, **not equal optimizer updates, pixels, or
compute**: tiling creates more examples. State this explicitly. Retain wall time from training logs,
resolved settings, and hardware from `provenance.json`. Treat this as a practical pipeline ablation.

Optional runs, after the core set:

```powershell
python run_pipeline.py --profile gpu --run-name image_tiles --raw "C:\PCB Data (Bootcamp)\PCB_DATASET" --split-mode image --model yolov8n.pt --epochs 20 --batch 8 --skip-frcnn
python run_pipeline.py --profile gpu --run-name grouped_small --raw "C:\PCB Data (Bootcamp)\PCB_DATASET" --split-mode board --model yolov8s.pt --epochs 20 --batch 8 --skip-frcnn
```

The image-split experiment answers a different generalization question and has different held-out
examples. Its gap from the board split is NOT a controlled causal estimate of leakage.
A second board seed, if affordable, shows sensitivity to the small number of layouts; do not select
the best seed using test results. Freeze the plan and validation decisions before final test evaluation.
Faster R-CNN is optional for the ablations: the original `--with-frcnn` / GPU baseline remains available.
Its own resize, augmentation and optimization settings differ from YOLO; describe those differences.

## Generate a comparison from actual runs

```powershell
python -m src.compare --names grouped_tiles_yolo grouped_full_yolo grouped_noaug_yolo --output results/ablations
```

The image split may be included in a clearly labelled protocol comparison, with separate sample counts.
`src.compare` labels run names/protocols, shows missing AP as N/A, and keeps size counts per run.

## Results table template — replace only from actual results files

| Run | Split | Input | Augmentation | Model | Actual epochs | mAP50 | mAP50:95 | P | R | Macro-F1 | ms/board | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grouped_tiles | board | tiled | on | YOLOv8n | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | NOT RUN |
| grouped_full | board | full | on | YOLOv8n | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | NOT RUN |
| grouped_noaug | board | tiled | off | YOLOv8n | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | NOT RUN |
| image_tiles (optional) | image | tiled | on | YOLOv8n | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | NOT RUN |
| grouped_small (optional) | board | tiled | on | YOLOv8s | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | NOT RUN |

Use `results/<run>_yolo/metrics.json` for metrics, `training_config.json` for training settings,
`training_history.csv` for epochs, and `provenance.json` for environment/weight/source hashes.
Copy no values from the quick profile into final results. Its subset flag and training fraction make
it a smoke test even if the command completes successfully.

## Tile-context diagnostics (additional analysis, not the headline score)

```powershell
python -m src.evaluate --data data/pcb_grouped_tiles --weights weights/grouped_tiles_yolo_best.pt --model yolo --name grouped_tiles_context --unit tile
```

This tunes a separate tile operating threshold. Keep it separate from full-board results. Overlapping
tiles repeat some defects, so tile counts are not unique physical-defect counts. `errors.csv` records
FP/FN class, score, size, source, board prefix, tile origin, context and distance from image edges.
`error_context.json` reports FP per evaluated clean/defect-containing image with denominators.
`size_by_class.csv` reports size recall for each class. In board mode, tile coordinates are null:
merged board errors cannot be attributed to an individual source tile by this export.

## Honest reporting language (fill only after real runs)

- "The image split achieved [VALUE, metrics path] on held-out images; layouts may be shared with training."
- "The grouped split achieved [VALUE, metrics path] on [COUNT, stats path] held-out filename-prefix groups.
  We verified the prefix-to-template mapping by [ACTUAL CHECK]. This is a single group partition."
- "The most frequent off-diagonal confusion was [TRUE] predicted as [PRED], with [COUNT] matches at
  IoU >= 0.5 and validation-selected confidence [VALUE]. See metrics.json and confusion_matrix.png."
- "Recall for [CLASS] in [SIZE] was [FOUND]/[COUNT], from size_by_class.csv. Small support limits this estimate."
- "Clean tiles produced [COUNT] false detections across [N] evaluated clean tiles ([RATE] FP/tile).
  Manual inspection of [IDENTIFIED EXAMPLES] suggests [OBSERVED PATTERN], without establishing its cause."
- "The tiling pipeline changed board mAP50:95 by [DIFFERENCE computed from two actual metrics.json files].
  The comparison uses identical held-out boards and epoch budgets, with different compute costs."

## Verify assumptions

Inspect actual `splits.csv`, `stats.json` and `manifest.json`: there may be fewer or more than ten prefixes.
Do not claim exactly 693 images unless your run reports that count. The group allocation uses group
counts, not an impossible promise of exact image ratios or class balance. Missing classes trigger warnings;
undefined class AP becomes JSON null / table N/A. Never silently move a test board to improve metrics.
The scorer uses 101-point AP and a global max of 100 predictions per evaluation image. It is COCO-style,
not asserted identical to the official COCO evaluator (which has additional conventions).
