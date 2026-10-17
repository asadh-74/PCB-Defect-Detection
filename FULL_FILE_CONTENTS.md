# Complete pasteable project files

Each section gives the exact relative path and complete UTF-8 content. Empty placeholders and the unchanged binary COCO checkpoint are omitted. This generated listing excludes itself. See CHANGES.md for what changed and why.

## `.github/workflows/tests.yml`

````yaml
name: Python regression tests
on: [push, pull_request]
permissions:
  contents: read
jobs:
  tests:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: python -m pip install pytest numpy pandas matplotlib opencv-python-headless Pillow pyyaml tqdm "streamlit>=1.40,<2"
      - run: python -m pytest -q tests
      - run: python run_pipeline.py --dry-run --profile quick --raw "PCB Data (Bootcamp)" --run-name ci
````

## `.gitignore`

````text
data/pcb_*/
data/raw/
runs/
weights/*.pt
weights/*.config.json
weights/*.history.csv
results/pipeline_*.json
results/*/preds_*.pkl
__pycache__/
*.pyc
.pytest_cache/
.ipynb_checkpoints/
.venv/
.env
.env.*
!.env.example
*.zip
````

## `.streamlit/config.toml`

````toml
[server]
maxUploadSize = 100

[theme]
primaryColor = "#0E8A0E"
````

## `.vscode/extensions.json`

````json
{ "recommendations": ["ms-python.python", "ms-python.debugpy", "ms-toolsai.jupyter"] }
````

## `.vscode/launch.json`

````json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "1. Pipeline: quick test (minutes)",
      "type": "debugpy", "request": "launch",
      "program": "${workspaceFolder}/run_pipeline.py",
      "args": ["--profile", "quick"],
      "console": "integratedTerminal", "justMyCode": true
    },
    {
      "name": "2. Pipeline: full run, NVIDIA GPU",
      "type": "debugpy", "request": "launch",
      "program": "${workspaceFolder}/run_pipeline.py",
      "args": ["--profile", "gpu"],
      "console": "integratedTerminal", "justMyCode": true
    },
    {
      "name": "3. Pipeline: full run, CPU only",
      "type": "debugpy", "request": "launch",
      "program": "${workspaceFolder}/run_pipeline.py",
      "args": ["--profile", "cpu"],
      "console": "integratedTerminal", "justMyCode": true
    },
    {
      "name": "4. Streamlit app",
      "type": "debugpy", "request": "launch",
      "module": "streamlit",
      "args": ["run", "streamlit_app/app.py"],
      "console": "integratedTerminal"
    },
    {
      "name": "5. Run tests",
      "type": "debugpy", "request": "launch",
      "module": "pytest",
      "args": ["tests", "-q"],
      "console": "integratedTerminal"
    }
  ]
}
````

## `.vscode/settings.json`

````json
{
  "python.terminal.activateEnvironment": true,
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "files.exclude": { "**/__pycache__": true, "**/.pytest_cache": true },
  "terminal.integrated.scrollback": 20000
}
````

## `CHANGES.md`

````markdown
# Review changes and verification

This update restores the source, tests, notebook and app from the supplied ZIP into the repository,
which previously contained only seven root files and a COCO starting checkpoint. That checkpoint is
preserved unchanged. No dataset, trained PCB checkpoint, or synthetic performance results are added.

The table lists changes relative to the supplied ZIP. All replacement text is provided in
`FULL_FILE_CONTENTS.md`, including complete source files rather than diff fragments.

| Exact file | What changed | Why |
|---|---|---|
| src/prepare_data.py | Board-prefix grouped split option; tiled/full representation; original-board evaluation copies; source manifests/hashes, group/class counts; warnings; duplicate and ratio checks; quoted portable YAML; drop any tile with ambiguous unlabelled fragments | Prevent layout leakage, allow fair ablations, preserve provenance, avoid contaminated labels and ambiguous file matches |
| src/common.py | Validate positive dimensions/stride and disallow gaps between tiles | Fail clearly for invalid tiling arguments |
| src/train_yolo.py | Augmentation on/off; disable dataset augmentation pipeline when off; save resolved config, elapsed time and history beside weights | Make the ablation real and training reproducible, including optional library transforms |
| src/train_frcnn.py | Keep final partial batch, validate nonempty datasets and numeric settings, explicit device support, save training config/history | Avoid zero batches on small datasets and retain baseline provenance |
| src/frcnn.py | Explicit weights_backbone=None when constructing the architecture | Avoid an unnecessary ImageNet download when loading a saved baseline; full COCO pretraining remains selected for training |
| src/predictors.py | Validate six YOLO PCB class names/order; normalize numeric Faster R-CNN device strings | Reject an accidental COCO checkpoint and support device arguments consistently |
| src/infer.py | Match small-image padding to training; clip/remove padding-only boxes; require overlap for border rejection; validate image reads/writes; selectable full-image inference | Avoid resolution mismatch, invalid detections, and silent I/O errors; retain class-aware NMS |
| src/evaluate.py | Optional board evaluation with tiled/direct dispatch; robust tiny-split speed timing and sampling; validation-empty guard; strict JSON nulls; saved confidence sweep, predictions/GT, size-by-class recall, FP/FN records and context rates; unique error filenames; source/config/history/environment/hardware evidence | Score both representations on common original boxes and support defensible error analysis |
| src/compare.py | Label run and split; guard mixed units; optional isolated output directory; preserve missing AP as N/A and missing plot values as gaps; size counts per run | Avoid misleading comparisons and overwrites |
| run_pipeline.py | Board split default; run isolation; experiment overrides; propagate device; command/source/raw metadata signatures; per-step completion records; invalidate downstream steps before reruns; always export the selected run's model | Preserve fail-fast behavior while avoiding stale experiments and incorrect demo exports |
| src/app_logic.py | Cached inference floor 0.001; configurable direct/tiled prediction and tile settings | Match the exported validation operating point and representation |
| streamlit_app/app.py | Class-colour legend; honest split/unit/subset labels; handle undefined AP; model-file cache invalidation; friendly model-load errors; optional PCB_MODEL_DIR | Improve demo clarity without adding dependencies; prevent overstating unseen-layout results |
| requirements.txt | Include Streamlit; raise torch/torchvision minimums for used APIs; pin exercised Ultralytics version | Make root installation include the app and stabilize augmentation behavior |
| streamlit_app/requirements.txt | Same Ultralytics pin | Keep training and demo model handling aligned |
| setup.ps1 | New fail-fast Python environment/install/test helper; CPU default with configurable Torch index; quoted paths | One setup path for PowerShell without activation |
| LOCAL_SETUP.md | Complete Windows/CPU/GPU/data/training/results/app instructions; explicit resume limits | Make installation and execution repeatable with spaces/parentheses |
| notebooks/colab_run.ipynb | Use the same checked argument-list pipeline; quick default; selectable ablations; artifact download | Avoid a second divergent training procedure |
| EXPERIMENTS.md | Three core runs plus optional comparisons, fair evaluation units, placeholder table and conditional report wording | Strengthen methodology and analysis within a free-GPU budget |
| REPORT_SKELETON.md | Seven planned pages with explicit pending fields and evidence paths | Cover grading criteria without fabricated findings |
| PRESENTATION_OUTLINE.md | Ten-slide evidence-based plan and demo notes | Support a concise final presentation |
| README.md | Update entry points, protocols, artifacts, limitations and pending status | Give a truthful top-level project description |
| DEPLOY.md | Align selected-run export and local demo with the new paths; link current provider instructions | Avoid mismatched models/metrics and unsupported hosting assumptions |
| tests/test_improvements.py | New group/split/path/representation/error/speed/threshold/NMS/padding/fragment/fail-fast/resume regressions | Exercise new logic and important edge cases |
| tests/test_app.py | Isolate no-model boot from local trained weights; test exported protocol/undefined metrics rendering | Make tests valid before and after training |
| .github/workflows/tests.yml | Python 3.12 lightweight tests on Windows and Linux | Check portability on actual CI runners |
| .gitignore | Ignore environments, secrets, run datasets, transient completion state and training artifacts | Keep the public repository focused on source and intentional evidence |
| CHANGES.md | This file-by-file record | Explain exact changes and validation |
| FULL_FILE_CONTENTS.md | Full pasteable text of project files | Satisfy the complete replacement-content requirement |

## Validation performed

- Original 17 tests passed after the initial implementation.
- Final expanded suite: **29 tests passed** on Linux Python 3.12.
- A one-epoch YOLO training run on tiny synthetic images completed with augmentation disabled, and
  its saved checkpoint completed full-image evaluation.
- The actual one-command pipeline completed preparation, COCO-initialized YOLO fine-tuning,
  validation-selected board evaluation, comparison and app export on synthetic fixture images.
- Faster R-CNN architecture construction and forward inference executed without pretrained downloads.
- Streamlit was checked both without weights and with exported-metrics metadata via AppTest.
- A dry run printed quoted arguments for a dataset path containing spaces and parentheses.
- The Windows PowerShell script and CUDA training were not executed in this Linux review environment.
  Windows CI and a local GPU/driver check remain necessary.

Synthetic runs are software checks only. Their metrics/checkpoints are excluded from the deliverable
and must never be used in the bootcamp report. Real PKU-Market-PCB performance remains **NOT RUN**.

## Deliberate limits

Board grouping follows the requested filename convention; verify the prefix-to-layout mapping.
Group balance is not guaranteed with few layouts. Source-image stratification still uses the first
class, suitable only if that represents the image; inspect multiclass variants before relying on it.
VOC coordinates retain the supplied dataset convention; verify its endpoint/index convention against
sample images rather than assuming one universal XML convention. The scorer is COCO-style, not an
asserted byte-for-byte replacement for pycocotools. Border rejection may miss larger objects; class-aware
NMS can retain cross-class duplicates. Tile exclusions and overlapping samples must be reported.
Raw resume detection uses path/size/mtime; use --force after content changes that preserve those values.
The app's timing/metrics do not certify a production inspection system.
````

## `DEPLOY.md`

````markdown
# Demo setup and optional hosting

First complete an actual PCB training/evaluation run. The app opens before training but does not
include a working six-class checkpoint until you export one. A COCO starting checkpoint is not enough.

## Export a selected run

```powershell
& ".\.venv\Scripts\python.exe" -m src.export_for_app --model yolo --weights weights/grouped_tiles_yolo_best.pt --results results/grouped_tiles_yolo
& ".\.venv\Scripts\python.exe" -m streamlit run streamlit_app/app.py
```

The pipeline also exports automatically. `models/` contains `yolo_best.pt`, `metrics_yolo.json` and,
when available, the confusion matrix. Keep the checkpoint and its corresponding results together.
From Colab, download the artifacts archive in the final notebook cell and extract it into this project.

## Optional Streamlit hosting

Use the provider's current instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app
Choose this repository, the desired branch, and `streamlit_app/app.py` as the entry point.
`streamlit_app/requirements.txt` is the app-specific dependency list; `packages.txt` is at the root.
Verify which requirements file the provider selects in its build logs and use Python 3.12 if supported.
CPU inference works locally; provider memory/runtime limits and available Python versions may change.
No hosted app is created by this repository update.

Before publishing, explicitly add only your trained model and matching exported metrics:

```bash
git add models/yolo_best.pt models/metrics_yolo.json models/confusion_matrix_yolo.png
git commit -m "Add trained PCB demo model and evaluation evidence"
git push
```

If the confusion matrix is absent, omit that filename. Check checkpoint size against GitHub's current
file limit before committing large weights; a smaller detector may be more practical for a CPU demo.
Exclude raw datasets, credentials, local environments and unrelated files. Keep dataset licensing
requirements when sharing samples. No sample image is bundled because no verified dataset sample
was supplied with the project archive.

## Demo checklist

1. Open locally and upload a genuine bare-PCB image.
2. Confirm the model has the six project classes and metrics describe the chosen protocol.
3. Check the colour legend, confidence control, zoomed crops and PNG/CSV downloads.
4. Confirm full-image and tiled variants use their matching inference paths.
5. If there are no detections, say "none detected at this threshold"; do not claim the board is defect-free.
````

## `EXPERIMENTS.md`

````markdown
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
````

## `LOCAL_SETUP.md`

````markdown
# Windows 10/11, VS Code, PowerShell, Python 3.12

Open the extracted project folder in VS Code. Run all commands from that folder.
Paths may contain spaces and parentheses; quote paths and use the PowerShell call operator `&`.
No virtual-environment activation or permanent execution-policy change is needed.

## 1. Install and test (CPU default)

Install Python 3.12 first and verify `python --version`. Then:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
```

This creates `.venv`, installs CPU PyTorch and dependencies, and runs the tests; every failed command
stops setup. The execution-policy setting applies only to this PowerShell process. If your managed
computer blocks scripts, use the equivalent commands:

```powershell
python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" -m pytest -q tests
```

For an NVIDIA GPU, use the compatible PyTorch install command generated by
https://pytorch.org/get-started/locally/ rather than guessing a CUDA version. Alternatively pass that
command's index URL to `setup.ps1 -TorchIndexUrl "https://download.pytorch.org/whl/ACTUAL_SELECTION"`.
`ACTUAL_SELECTION` is a placeholder, not a usable URL. Setup installs the selected builds into the
project environment. If switching an existing CPU installation, follow the official reinstall steps.
Check:

```powershell
& ".\.venv\Scripts\python.exe" -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Select `.venv\Scripts\python.exe` as the VS Code interpreter. On Linux/Colab use `python` instead
of the Windows environment path and install requirements after selecting the appropriate PyTorch build.

## 2. First smoke run

```powershell
& ".\.venv\Scripts\python.exe" run_pipeline.py --profile quick --run-name smoke --skip-frcnn --raw "C:\PCB Data (Bootcamp)\PCB_DATASET"
```

The raw folder contains `Annotations/` and `images/`; nested folders are located automatically.
Omit `--raw` to try the existing downloader. It may need network access or credentials from the data
provider; manual download is the fallback. The first pretrained-model run also needs network access
unless the model file is already available. The repository's root `yolov8n.pt`, if present, is a
COCO starting checkpoint, **not a trained six-class PCB detector**. Never export it as the final model.

`quick` is deliberately weak: two epochs, a training fraction, and capped evaluation. It validates
execution, not accuracy or grading results. Full-board scanning may still take time on a CPU.

## 3. Real training (prefer one free Colab/Kaggle GPU)

Start with the budgeted main run; see `EXPERIMENTS.md` for controlled alternatives:

```powershell
& ".\.venv\Scripts\python.exe" run_pipeline.py --profile gpu --run-name grouped_tiles --model yolov8n.pt --epochs 20 --batch 8 --skip-frcnn --raw "C:\PCB Data (Bootcamp)\PCB_DATASET"
```

Without a GPU use `--profile cpu`; training may take many hours. The original `--profile gpu` defaults
to YOLOv8s with 100 epochs and a 20-epoch Faster R-CNN baseline. `--skip-frcnn` avoids that cost;
`--with-frcnn` explicitly enables the baseline for another profile.

The pipeline defaults to board-prefix grouping. Use `--split-mode image` only for the labelled
comparison. Standalone `src.prepare_data` keeps its original image-split default for compatibility.
Use unique `--run-name` values for experiments. Data, weights, metrics and completion records are
isolated by run. Results from changed source/config/dataset metadata are rebuilt. Rerunning an unchanged
command skips completed steps; a failed step stops the pipeline immediately. A missing earlier output
invalidates later completion records. `--force` rebuilds everything for that run. If raw content is
changed while preserving both size and timestamps, use `--force`; metadata-based resume cannot detect it.
Do not run concurrent pipelines with the same run name. Epoch resume within a failed training step
is not implemented: that step restarts from the configured starting checkpoint.

## 4. Results and demo

The pipeline exports its selected YOLO model automatically. Outputs for `grouped_tiles`:

- `results/grouped_tiles_yolo/`: metrics, predictions, ground truth, confidence sweep, per-class and
  size tables, errors/crops, dataset snapshot, training configuration/history, provenance.
- `results/grouped_tiles/comparison.md` and `comparison.png`.
- `weights/grouped_tiles_yolo_best.pt` and `models/yolo_best.pt`.

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run streamlit_app/app.py
```

The app opens without weights and explains what is missing. Detection requires a six-class PCB model
produced by training. The latest pipeline run becomes the exported demo model. To restore a selected run:

```powershell
& ".\.venv\Scripts\python.exe" -m src.export_for_app --model yolo --weights weights/grouped_tiles_yolo_best.pt --results results/grouped_tiles_yolo
```

Use the colour legend, confidence control, zoomed crops and PNG/CSV downloads. Confidence defaults to
the exported validation operating point; changing it is exploratory and does not update saved metrics.
A full-image variant uses direct inference; the tiled variant scans overlapping tiles. Results indicate
the split, evaluation unit and whether evaluation was capped. See `DEPLOY.md` for optional hosting.

## 5. Reproducibility and troubleshooting

Save the actual environment with:

```powershell
& ".\.venv\Scripts\python.exe" -m pip freeze | Set-Content -Encoding utf8 results/environment.txt
```

| Problem | Action |
|---|---|
| GPU profile refuses to run | Verify CUDA PyTorch or use CPU/Colab; there is no silent CPU fallback |
| CUDA out of memory | Lower `--batch 4` or `--batch 2` |
| Too few board prefixes | Check dataset layout/names; at least three groups are required |
| Missing class warning | Inspect saved membership; do not invent AP for a class with no ground truth |
| Duplicate image/XML stem | Resolve duplicates rather than silently selecting a file |
| Module `src` not found | Run from project root; confirm the complete repository was downloaded |
| Dataset/model download unavailable | Download manually and pass `--raw` / a local `--model` checkpoint |
| Windows multiprocessing issue | For individual training scripts use `--workers 0`; pipeline uses two on Windows |
| Cannot infer without weights | Complete real training; COCO weights cannot classify these six PCB classes |

The automated setup script is provided for Windows; validation in this review ran on Linux Python 3.12.
A local PowerShell run remains necessary to verify your installed Python/GPU/driver combination.
````

## `PRESENTATION_OUTLINE.md`

````markdown
# Final presentation — 10 slides

**All measured results are PENDING. Do not replace placeholders with illustrative numbers.**

| Slide | Title | Content / visual | Evidence |
|---|---|---|---|
| 1 | Detecting bare-PCB defects | Problem, author, scope, one real annotated image | Dataset source and example ID |
| 2 | Why this is difficult | Tiny defects, similar trace shapes, misses versus false alarms | Actual size statistics saved in results/ |
| 3 | Data and leakage controls | Classes, actual source/group counts, image versus board split | stats.json, splits.csv, manifest.json |
| 4 | Pipeline | Source split → representation → pretrained detector → validation threshold → test | Code and saved training_config.json |
| 5 | Preserving small defects | Tiled versus full-image inputs; overlap, border handling and NMS | Actual prepared example plus configuration |
| 6 | Controlled experiments | Three core variants, fixed source split, equal epochs with unequal compute caveat | EXPERIMENTS.md and actual run configs |
| 7 | Results | [PENDING] board-level comparison, class AP, explicit N/A/NOT RUN | Generated comparison.md/png, metrics.json |
| 8 | What failed and why it may fail | Real confusion pair, missed size bucket, FP contexts and labelled crops | size_by_class.csv, errors.csv, error_context.json |
| 9 | Live demonstration | Upload a bare board, explain legend/confidence, zoom defects, export CSV | Trained exported model; say if changing the validated threshold |
| 10 | Conclusions and limitations | One supported finding, one limitation, one evidence-led next step | Actual results paths and provenance |

Speaker notes:
- Describe bootcamp objectives and the bounded task; avoid production-ready claims.
- Do not describe source-image holdout as unseen layouts. Verify board prefix mapping first.
- Never compare tile mAP directly against board mAP. Identify the evaluation unit on every chart.
- State that smoke tests verify software execution, not detector quality.
- Explain which settings were chosen from validation and keep test results out of tuning decisions.
- Demo fallback: retain a previously exported annotated image and CSV from an actual inference run;
  label it as a saved run if the live environment is unavailable.
````

## `README.md`

````markdown
# PCB Defect Detection

AI Bootcamp final project — Computer Vision track. Author: Asad Hussain.

Detect six bare-PCB defect classes: `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`,
`spurious_copper`, using COCO-pretrained YOLO with a Faster R-CNN baseline.

**Real PKU-Market-PCB training and final results are pending.** Software tests and synthetic smoke
runs verify implementation, not detection accuracy. No trained PCB weights or invented metrics are
bundled. A root `yolov8n.pt`, if present, is only a COCO starting checkpoint.

## Start here

- [Windows / VS Code setup](LOCAL_SETUP.md) — Python 3.12, quoted paths, fail-fast setup.
- [Colab notebook](notebooks/colab_run.ipynb) — run the same pipeline on a free GPU.
- [Experiment plan and honest interpretation](EXPERIMENTS.md).
- [Seven-page report skeleton](REPORT_SKELETON.md) and [10-slide outline](PRESENTATION_OUTLINE.md).
- [File-by-file changes](CHANGES.md) and [complete replacement file contents](FULL_FILE_CONTENTS.md).
- [Streamlit deployment](DEPLOY.md).

Windows first setup:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
& ".\.venv\Scripts\python.exe" run_pipeline.py --profile quick --run-name smoke --skip-frcnn --raw "C:\PCB Data (Bootcamp)\PCB_DATASET"
& ".\.venv\Scripts\python.exe" -m streamlit run streamlit_app/app.py
```

The app can open before training; actual defect inference needs the trained checkpoint. Use a free
GPU for the real runs. See LOCAL_SETUP.md before choosing a CUDA PyTorch installation.

## Dataset and split protocols

PKU-Market-PCB official resource: https://robotics.pkusz.edu.cn/resources/datasetENG
The project expects `Annotations/` and `images/`. It uses annotated source images with recognized
classes; inspect the actual run's counts instead of assuming an exact dataset size. Dataset variants
and directory structures differ; `--ann_dir` and `--img_dir` override discovery in the preparation script.

- **Image split:** deterministic, stratified by first annotated class, before tiling. Layouts may overlap.
- **Board split:** deterministic assignment of complete filename-prefix groups before tiling.
  `01_missing_hole_01` maps to group `01`. Verify that convention against actual template boards.
- Pipeline default: `--split-mode board`. Standalone preparation retains `--split-mode image` by default.
- `splits.csv`, `stats.json` and content-hashed `manifest.json` document actual membership and class support.
  Group counts approximate requested ratios; image counts and class balance are not guaranteed.

Report both protocols separately: held-out images of potentially familiar layouts versus held-out
board-prefix groups. They have different test images; their score difference does not isolate leakage.

## Methods and experiments

Native-resolution 640-pixel tiles, stride 480, retain small defects. Far-edge alignment covers borders;
small inputs are black-padded consistently in preparation and inference. Partially visible boxes
retaining at least 50% area and two pixels per side are kept. Tiles with an unlabelled partial defect
of at least 5% visibility are discarded, even beside a kept box. Train retains a seeded 10% of clean
tiles; validation/test retain all eligible clean tiles. These exclusions affect the tile evaluation domain.

`--representation full` retains original whole-board images for normal detector resizing/letterboxing.
`--augmentation off` disables the YOLO training augmentation pipeline, including optional Albumentations.
Model size, epochs, batch and seed are selectable. Use EXPERIMENTS.md's three-run core plan.

The one-command pipeline evaluates both representations on **original full boards**, using original
boxes once per board. The tiled model merges overlapping predictions with class-aware NMS; the full
variant predicts directly. A separate `--unit tile` evaluation remains available for context diagnostics.
The baseline uses 16-pixel smallest anchors and its own 800-pixel internal resize.

The shared numpy scorer implements COCO-style 101-point AP across IoU 0.50:0.95, not the complete official
COCO evaluator. It caps predictions at 100 per evaluation image. Operating confidence maximizes macro-F1
on validation only and is then fixed for test. Missing-class AP is undefined (JSON null / table N/A).

## Run commands (activated environment / Linux / Colab)

```bash
python -m pytest -q tests
python run_pipeline.py --profile quick --run-name smoke --skip-frcnn
python run_pipeline.py --profile gpu --run-name grouped_tiles --model yolov8n.pt --epochs 20 --batch 8 --skip-frcnn
python run_pipeline.py --profile cpu --run-name cpu_run
python -m streamlit run streamlit_app/app.py
```

Use `--raw "YOUR DATASET PATH"` for manual data. `--dry-run` prints safely quoted command arguments
without modifying outputs. Unique run names isolate experiments. The pipeline stops on failed steps
or missing outputs; completion records are invalidated after source/config/raw metadata changes.
`--force` rebuilds. An interrupted training step restarts; this is step-level resume, not epoch resume.

Individual scripts remain available:

```bash
python -m src.prepare_data --raw "YOUR DATASET PATH" --out data/pcb_yolo --split-mode board --clean
python -m src.train_yolo --data data/pcb_yolo/pcb.yaml --model yolov8n.pt --epochs 20
python -m src.train_frcnn --data data/pcb_yolo --epochs 20
python -m src.evaluate --model yolo --weights weights/yolo_best.pt --data data/pcb_yolo --unit board --name yolo
python -m src.evaluate --model frcnn --weights weights/frcnn_best.pt --data data/pcb_yolo --unit board --name frcnn
python -m src.compare --names yolo frcnn
python -m src.export_for_app --model yolo --weights weights/yolo_best.pt --results results/yolo
```

## Results and error evidence

**NOT RUN on the real dataset.** Use the results table template in EXPERIMENTS.md.
After an actual run, `results/<run>_yolo/` contains:

| Artifact | Purpose |
|---|---|
| metrics.json / per_class.csv | Aggregate and class metrics, threshold, protocol and unit |
| confidence_sweep.json / conf_sweep.png | Validation-only operating-point evidence |
| size_by_class.csv | Recovered/total and recall by class and size |
| confusion_matrix.png | Class confusion and unmatched detections |
| errors.csv / errors.json | FP/FN observations, sizes, edge distances and source context |
| error_context.json | FP counts and rates with clean/defect-containing input denominators |
| error_examples/ | Real error illustrations with unique filenames |
| predictions.json / ground_truth.json | Inputs to reproduce reported scorer results |
| stats.json / splits.csv / manifest.json | Dataset statistics, membership and source hashes |
| training_config.json / training_history.csv | Actual training settings and completed epochs |
| provenance.json | Environment, hardware, source and checkpoint hashes |

Board-mode errors do not identify contributing tiles. Use the additional tile diagnostic command in
EXPERIMENTS.md. Proximity to an edge is an observation, not proof that tiling caused an error.

## Repository contents

`src/`: download, prepare, train, predict, infer, score, evaluate, compare and export modules.
`run_pipeline.py`: fail-fast orchestration and isolated run completion records.
`streamlit_app/`: fast upload demo with cached inference, class legend, zooms and PNG/CSV downloads.
`tests/`: original regression tests plus grouped splitting, representation, error export, NMS/border,
confidence and resume checks. `.github/workflows/tests.yml`: Python 3.12 checks on Linux and Windows.
`setup.ps1`: Windows dependency installation and tests without environment activation.

## Limitations and verification

Only a small number of layout groups; verify actual group counts and prefix mapping. Dataset-domain
performance does not establish production-line performance. Confirm synthetic-defect provenance and
licensing from the source used. Tiny fragments below the exclusion threshold may remain in tiles;
overlap duplicates some tile ground truth. Border rejection assumes overlap and sufficiently small
objects. NMS merges same-class duplicates only; different-class duplicates can survive. One partition
is not statistical validation. Free GPU availability/runtime varies.

Ultralytics is pinned to the version exercised during this review; capture the actual full dependency
environment for your real run. Setup was exercised on Linux Python 3.12; run the provided Windows CI
and local setup to verify PowerShell/driver behavior on your machine. No real dataset or GPU result
is claimed by the repository until its corresponding results files are produced.
````

## `REPORT_SKELETON.md`

````markdown
# PCB Defect Detection — seven-page report skeleton

Author: Asad Hussain | AI Bootcamp, Computer Vision track

**DRAFT — REAL TRAINING/EVALUATION PENDING.** All [PENDING: ...] fields must be filled only from
actual `results/` artifacts. Seven page sections are a layout guide; final pagination depends on
font, figures and your submission template. Delete prompts after replacing them with supported text.
Do not present planned work as completed work.

## Page 1 — Problem, motivation and objectives

### Abstract (write last)

This project investigates detection and classification of bare-PCB surface defects using a pretrained
YOLO detector and a Faster R-CNN baseline. Small defects motivate native-resolution tiling. Evaluation
separates source images before preprocessing and includes a board-prefix-grouped protocol to assess
unseen layouts. [PENDING: actual models, results, and the strongest supported conclusion, with paths.]

### Problem identification

A bare PCB can contain absent drill holes, notched traces, interrupted paths, unintended connections,
copper protrusions, or isolated copper residue. The objective is to localize defects with bounding
boxes and classify them into the six dataset categories. Explain how missed defects and false alarms
matter for inspection, without claiming verified electrical failure or production-line performance.

### Objectives and scope

1. Reproduce a data-to-evaluation pipeline with leakage controls.
2. Compare detectors and a small number of practical preprocessing/training choices.
3. Analyse class confusion, missed sizes, and false-positive contexts.
4. Demonstrate uploaded-image inference in Streamlit.

Scope: bare-board images from the chosen dataset distribution. Exclude assembled-PCB inspection,
electrical continuity verification, manufacturing deployment and safety certification.

Suggested figure: one actual dataset board with annotated examples; record its source and licence.

<!-- PAGE BREAK -->
## Page 2 — Dataset and evaluation protocols

Dataset source/version/download date: [PENDING: provenance or run notes saved under results/].
Consult the official dataset description before finalizing claims about defect generation or licensing.
Annotation format: Pascal VOC XML; class order follows `src/common.py`.

| Quantity | Actual value | Evidence under results/ |
|---|---|---|
| Annotated source images used | [PENDING] | <run>_yolo/stats.json |
| Distinct board-prefix groups | [PENDING] | <run>_yolo/stats.json |
| Train / val / test source counts | [PENDING] | <run>_yolo/stats.json |
| Per-class source box counts | [PENDING] | <run>_yolo/stats.json |
| Prepared image / clean tile counts | [PENDING] | <run>_yolo/stats.json |
| Dropped ambiguous tiles | [PENDING] | <run>_yolo/stats.json |

Explain image splitting (stratified by first annotated class) versus grouped splitting (prefix before
first underscore, verified against templates by [PENDING: actual check]). Both occur before tiling.
Show actual group membership from `splits.csv`; do not assert exact 70/15/15 image ratios for a small
number of groups. Discuss missing-class warnings and undefined AP.

Image-split results concern held-out images of potentially familiar layouts. Grouped results concern
held-out filename-prefix groups; template-level interpretation depends on verifying that mapping.
Do not pool these results. Different test sets prevent interpreting their difference as solely leakage.

Suggested figure: actual source box-size distribution or source count table. If deriving a figure,
write its underlying numbers and script output into `results/` before citing it.

<!-- PAGE BREAK -->
## Page 3 — Methodology and implementation

Describe VOC parsing, image matching, source split, tiled/full preparation and training.
Tiled defaults: 640-pixel square crops with 480-pixel stride; far-edge alignment and black padding for
small images. A box must retain at least half its area and two pixels in each dimension. Tiles with
unlabelled partial defects of at least 5% visibility are dropped, even when another labelled box is
present. Below-5% fragments remain a limitation. Explain training background sampling and its seed.

Explain COCO-pretrained YOLO fine-tuning, actual architecture and actual resolved settings using
`training_config.json`. Describe Faster R-CNN's replacement class head, anchor sizes and internal
800-pixel resize if actually trained; mark NOT RUN if omitted. The two models have different training
recipes, so a comparison is of configured pipelines.

Explain tiled inference: native-scale padding, coordinate offsets, interior-edge rejection and
class-aware NMS. Edge rejection assumes overlap and sufficiently small objects. Neighbouring tiles
are not guaranteed to recover every large or poorly localized defect. Cross-class duplicates can remain.

Define AP, precision, recall and macro-F1. The shared numpy scorer uses IoU thresholds from 0.50 to
0.95 and 101 recall samples, with a 100-detection limit per evaluation image. Call it COCO-style.
Confidence is chosen on validation only, then applied unchanged to test. Save and show the sweep.

<!-- PAGE BREAK -->
## Page 4 — Experimental design and reproducibility

State the predeclared plan in `EXPERIMENTS.md`: grouped tiled baseline, full-image alternative and
augmentation-off alternative. Add model size/image protocol/second seed only if resources allow.
All core variants share source membership, seed, model and epoch budget, and use full-board evaluation.
Equal epochs are not equal training compute when tiling expands the dataset.

| Setting | Recorded value | Evidence |
|---|---|---|
| Seed / split / representation | [PENDING] | stats.json / manifest.json |
| Actual epochs, batch, training fraction | [PENDING] | training_config.json / training_history.csv |
| Python, package versions and hardware | [PENDING] | provenance.json |
| Weight and code hashes | [PENDING] | provenance.json |
| Validation confidence | [PENDING] | confidence_sweep.json / metrics.json |
| Runtime / GPU model | [PENDING] | training_config.json / provenance.json |

Include exact commands, repository commit, and dataset source in a run note saved in results/.
Keep quick/subset smoke runs separate. Report failures, early stopping and omitted experiments rather
than hiding them. Record a frozen dependency snapshot with `python -m pip freeze` on the actual machine.

<!-- PAGE BREAK -->
## Page 5 — Results and supported analysis

**No values are available until real training and evaluation finish.**

| Run / protocol | Unit | mAP50 | mAP50:95 | Precision | Recall | Macro-F1 | ms/image |
|---|---|---|---|---|---|---|---|
| Grouped tiled YOLO | board | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| Grouped full YOLO | board | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| Grouped no augmentation | board | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| Faster R-CNN (if run) | board | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] | [PENDING] |

Insert the generated comparison and per-class AP table. Cite exact metrics paths beside each row.
Report denominators and absent-class N/A values. State whether timing includes tile merging and model
preprocessing; speed uses already-decoded images, sequential inference after warm-up, excludes image
file reads, and is specific to the recorded hardware. Board FPS is not tile FPS.

Discuss whether evidence supports tiling or augmentation for this partition. Do not claim statistical
significance from a single seed, guaranteed free-GPU runtimes, or superiority on unseen production data.
Present image-split results separately if run and label their different generalization question.

<!-- PAGE BREAK -->
## Page 6 — Error analysis

Insert `confusion_matrix.png`; rows are true class, columns are prediction, with background for misses
and unmatched detections. Off-diagonal matches use class-agnostic IoU matching; class-aware FP/FN
counts come from a separate matching pass and need not equal off-diagonal counts one-for-one.

1. Class confusion: [PENDING: pair, count, threshold, and actual example identifiers].
2. Size misses: [PENDING: class, size bin, recovered/total, `size_by_class.csv`].
3. False-positive context: [PENDING: clean versus defect-containing input FP rates and denominators,
   `error_context.json`]. For tile context use a separate tile diagnostic run.
4. Border examples: [PENDING: observed errors, distances, tile origins from `errors.csv`]. Do not infer
   a higher border error rate from counts alone; an exposure denominator would be required.

Use two or three real, labelled crops from `error_examples/`: green GT, red predictions, yellow
highlight. State how examples were selected. Manually describe visible patterns without pretending
the automatic exporter diagnoses physical causes. Overlapping tile examples may repeat one defect.
Use wording from `EXPERIMENTS.md`, replacing every bracket with actual evidence.

<!-- PAGE BREAK -->
## Page 7 — Limitations, conclusions and next steps

Limitations to discuss: small number of layouts; filename-prefix grouping assumption; single partition;
missing classes or small size buckets; dataset-domain shift; possible synthetic-defect bias (verify
source); ambiguous tile exclusions; training negative sampling; differences in compute and model
recipes; confidence/NMS sensitivity; no independent production-line validation.

Conclusion template: "Under [ACTUAL PROTOCOL], [MODEL/PIPELINE] achieved [ACTUAL METRIC from PATH].
The [ACTUAL ABLATION] suggests [SUPPORTED OBSERVATION] on this held-out partition. The error analysis
identified [OBSERVED PATTERN with evidence]. These findings support [LIMITED CONCLUSION], while
[ACTUAL LIMITATIONS] prevent broader claims."

Next steps should follow actual errors: collect independent layouts, annotate real acquisition data,
repeat group partitions, test lighting/scale robustness, validate board-level operating thresholds,
or investigate edge-aware merging. Do not promise improvement without measurement.

References: official dataset page, Ultralytics training documentation, torchvision Faster R-CNN
documentation, and original model papers actually consulted. Include access/version details.

Submission checklist: every bracket resolved or explicitly NOT RUN; every numeric finding backed by
an actual file in results/; exact commands/configs retained; no credentials or private paths in shared
screenshots; source, tests, limitations and demo instructions included.
````

## `models/README.md`

````markdown
Put the exported weights here (created by `python -m src.export_for_app`):

- yolo_best.pt (about 22 MB for yolov8s, fine for GitHub)
- metrics_yolo.json, confusion_matrix_yolo.png (shown on the app's "About" tab)

Commit this folder so Streamlit Community Cloud can load the model.
````

## `notebooks/colab_run.ipynb`

````json
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# PCB Defect Detection \u2014 reproducible GPU run\n",
        "Select Runtime \u2192 Change runtime type \u2192 GPU. This notebook calls the same fail-fast pipeline used locally.\n",
        "Real training results are pending. Keep QUICK=True for the first execution; smoke results are not final results.\n",
        "Download results before the runtime disconnects. See EXPERIMENTS.md for the controlled core runs.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "import os, sys, subprocess\n",
        "from pathlib import Path\n",
        "\n",
        "def run(args):\n",
        "    print(subprocess.list2cmdline([str(a) for a in args]), flush=True)\n",
        "    subprocess.run([str(a) for a in args], check=True)\n",
        "\n",
        "PROJECT = Path('/content/PCB-Defect-Detection')\n",
        "if not PROJECT.exists():\n",
        "    run(['git', 'clone', 'https://github.com/asadh-74/PCB-Defect-Detection.git', PROJECT])\n",
        "os.chdir(PROJECT)\n",
        "assert (PROJECT / 'src').exists(), 'Download the complete updated repository'\n",
        "run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])\n",
        "run([sys.executable, '-m', 'pytest', '-q', 'tests'])\n",
        "import torch\n",
        "assert torch.cuda.is_available(), 'Select a GPU runtime before training'\n",
        "print(torch.cuda.get_device_name(0))\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Choose the data and run\n",
        "Upload and extract the dataset (Annotations/ and images/), then set RAW to its root.\n",
        "Leave RAW=None to try the automatic downloader. Data-provider access can vary.\n",
        "Use unique RUN_NAME values for variants and keep source split/seed fixed for the core ablations.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "QUICK = True\n",
        "RAW = None  # Example: '/content/PCB_DATASET' or a Kaggle input path\n",
        "RUN_NAME = 'smoke' if QUICK else 'grouped_tiles'\n",
        "SPLIT_MODE = 'board'\n",
        "REPRESENTATION = 'tiled'  # 'full' for whole-image resizing\n",
        "AUGMENTATION = 'on'      # 'off' for the augmentation ablation\n",
        "EPOCHS = 2 if QUICK else 20\n",
        "\n",
        "args = [sys.executable, 'run_pipeline.py', '--profile', 'quick' if QUICK else 'gpu',\n",
        "        '--run-name', RUN_NAME, '--split-mode', SPLIT_MODE, '--representation', REPRESENTATION,\n",
        "        '--augmentation', AUGMENTATION, '--model', 'yolov8n.pt', '--epochs', str(EPOCHS),\n",
        "        '--batch', '8', '--skip-frcnn']\n",
        "if RAW is not None:\n",
        "    args += ['--raw', RAW]\n",
        "run(args)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Inspect actual outputs\n",
        "Quick runs are execution checks. Do not paste their metrics into the report.\n",
        "Full-board scores compare like evaluation units; tile diagnostics are separate.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "import json\n",
        "from IPython.display import display, Markdown, Image\n",
        "result = Path('results') / f'{RUN_NAME}_yolo'\n",
        "display(Markdown((Path('results') / RUN_NAME / 'comparison.md').read_text()))\n",
        "display(Image(filename=str(result / 'confusion_matrix.png')))\n",
        "print(json.loads((result / 'stats.json').read_text()))\n",
        "for image in sorted((result / 'error_examples').glob('*.jpg'))[:4]:\n",
        "    display(Image(filename=str(image)))\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Download and preserve the trained model and evidence\n",
        "The pipeline exports models/yolo_best.pt and its metrics. Unzip the downloaded archive into your local\n",
        "project to run Streamlit. These are your actual run artifacts; do not substitute COCO starting weights.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "import zipfile\n",
        "from google.colab import files\n",
        "archive = Path('/content') / f'pcb_{RUN_NAME}_artifacts.zip'\n",
        "with archive.open('wb') as target:\n",
        "    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:\n",
        "        for directory in ['results', 'weights', 'models']:\n",
        "            for path in Path(directory).rglob('*'):\n",
        "                if path.is_file():\n",
        "                    z.write(path, path.as_posix())\n",
        "files.download(str(archive))\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.12"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
````

## `packages.txt`

````text
libgl1
libglib2.0-0
````

## `requirements.txt`

````text
torch>=2.3
torchvision>=0.18
ultralytics==8.4.181
opencv-python-headless>=4.8
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
pyyaml>=6.0
Pillow>=10.0
tqdm>=4.65
pytest>=7.0
kagglehub>=0.3
huggingface_hub>=0.23
streamlit>=1.40,<2
````

## `run_pipeline.py`

````python
"""Run the whole project with ONE command (Windows, macOS, Linux; works from the VS Code terminal).

    python run_pipeline.py --profile quick   # smoke test: checks every step works (minutes)
    python run_pipeline.py --profile gpu     # full run on an NVIDIA GPU
    python run_pipeline.py --profile cpu     # lighter run for machines without a GPU

Re-running skips steps whose outputs already exist, so you can stop and resume.
Use --force to redo everything. Switching profile automatically retrains.
"""
import hashlib
import re
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

PROFILES = {
    "quick": dict(model="yolov8n.pt", epochs=2, batch=8, fraction=0.1, frcnn_epochs=1, max_train=120, max_tiles=150),
    "cpu": dict(model="yolov8n.pt", epochs=30, batch=8, fraction=1.0, frcnn_epochs=0, max_train=None, max_tiles=None),
    "gpu": dict(model="yolov8s.pt", epochs=100, batch=16, fraction=1.0, frcnn_epochs=20, max_train=None, max_tiles=None),
}


def have_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="default", help="isolated experiment name (letters, digits, _ and -)")
    ap.add_argument("--split-mode", choices=["image", "board"], default="board")
    ap.add_argument("--representation", choices=["tiled", "full"], default="tiled")
    ap.add_argument("--augmentation", choices=["on", "off"], default="on")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--profile", choices=PROFILES, default="quick")
    ap.add_argument("--raw", default=None, help="folder containing Annotations/ and images/ (default: auto-download)")
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: automatic)")
    ap.add_argument("--with-frcnn", action="store_true", help="also train the Faster R-CNN baseline")
    ap.add_argument("--skip-frcnn", action="store_true", help="skip the Faster R-CNN baseline")
    ap.add_argument("--force", action="store_true", help="redo steps even if their outputs exist")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        ap.error("--run-name may contain only letters, digits, underscores and hyphens")
    cfg = dict(PROFILES[args.profile])
    for key in ("epochs", "batch", "model"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if cfg["epochs"] < 1 or cfg["batch"] < 1:
        ap.error("--epochs and --batch must be positive")
    cuda = have_cuda()
    use_frcnn = cfg["frcnn_epochs"] > 0
    if args.profile == "quick":
        use_frcnn = cuda  # on CPU the baseline makes the smoke test very slow
    if args.with_frcnn:
        use_frcnn = True
        cfg["frcnn_epochs"] = cfg["frcnn_epochs"] or 5
    if args.skip_frcnn:
        use_frcnn = False
    print(f"profile={args.profile} | CUDA GPU: {cuda} | Faster R-CNN baseline: {use_frcnn}")
    if args.profile == "gpu" and not cuda and not args.dry_run:
        sys.exit("Profile 'gpu' needs an NVIDIA GPU with a CUDA build of PyTorch (see LOCAL_SETUP.md). "
                 "Use --profile cpu, or train on Colab/Kaggle.")

    state_file = ROOT / "results" / f"pipeline_{args.run_name}.json"
    force = args.force
    workers = 2 if os.name == "nt" else 4
    dev = ["--device", args.device] if args.device else []

    if args.dry_run:
        raw = args.raw or "<auto-downloaded dataset folder>"
    elif args.raw:
        raw = args.raw
    else:
        sys.path.insert(0, str(ROOT))
        from src.download_data import get_dataset
        raw = get_dataset(str(ROOT / "data" / "raw"))
    print("dataset:", raw)

    data = f"data/pcb_{args.run_name}"
    yn, fn = f"{args.run_name}_yolo", f"{args.run_name}_frcnn"
    yw, fw = f"weights/{yn}_best.pt", f"weights/{fn}_best.pt"
    steps = [
        ("Prepare dataset",
         [PY, "-m", "src.prepare_data", "--raw", raw, "--out", data, "--clean", "--split-mode", args.split_mode, "--representation", args.representation, "--seed", str(args.seed)],
         [f"{data}/pcb.yaml", f"{data}/stats.json", f"{data}/manifest.json"]),
        ("Train YOLO",
         [PY, "-m", "src.train_yolo", "--data", f"{data}/pcb.yaml", "--model", cfg["model"],
          "--epochs", str(cfg["epochs"]), "--batch", str(cfg["batch"]), "--fraction", str(cfg["fraction"]),
          "--workers", str(workers), "--name", yn, "--out", yw, "--seed", str(args.seed), "--augmentation", args.augmentation] + dev,
         [yw]),
    ]
    ev = ["--max_tiles", str(cfg["max_tiles"])] if cfg["max_tiles"] else []
    if use_frcnn:
        mt = ["--max_train", str(cfg["max_train"])] if cfg["max_train"] else []
        steps.append(("Train Faster R-CNN baseline",
                      [PY, "-m", "src.train_frcnn", "--data", data, "--epochs", str(cfg["frcnn_epochs"]),
                       "--val_every", "1", "--workers", str(workers), "--seed", str(args.seed), "--out", fw] + mt + dev,
                      [fw]))
    steps.append(("Evaluate YOLO",
                  [PY, "-m", "src.evaluate", "--model", "yolo", "--weights", yw, "--name", yn, "--data", data, "--unit", "board"] + ev + dev,
                  [f"results/{yn}/metrics.json"]))
    names = [yn]
    if use_frcnn:
        steps.append(("Evaluate Faster R-CNN",
                      [PY, "-m", "src.evaluate", "--model", "frcnn", "--weights", fw, "--name", fn, "--data", data, "--unit", "board"] + ev + dev,
                      [f"results/{fn}/metrics.json"]))
        names.append(fn)
    steps.append(("Compare models", [PY, "-m", "src.compare", "--output", f"results/{args.run_name}", "--names"] + names, [f"results/{args.run_name}/comparison.md"]))
    steps.append(("Export weights for the Streamlit app",
                  [PY, "-m", "src.export_for_app", "--model", "yolo", "--weights", yw, "--results", f"results/{yn}"],
                  ["models/yolo_best.pt"]))

    # Config and source changes invalidate completion records, not just the profile.
    # Raw metadata catches local dataset changes; preparation saves content hashes too.
    raw_path = Path(raw).resolve()
    source = [ROOT / "run_pipeline.py"] + sorted((ROOT / "src").glob("*.py"))
    fingerprint = {"options": vars(args) | {"force": False, "dry_run": False},
                   "raw": str(raw_path), "cfg": cfg, "frcnn": use_frcnn,
                   "source": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source}}
    if raw_path.exists():
        fingerprint["raw_files"] = [(str(p.relative_to(raw_path)), p.stat().st_size, p.stat().st_mtime_ns)
                                     for p in sorted(raw_path.rglob("*")) if p.is_file()]
    signature = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    state = {"signature": signature, "completed": []}
    if state_file.exists() and not force:
        old = json.loads(state_file.read_text())
        if old.get("signature") == signature:
            state = old
    if not args.dry_run:
        state_file.parent.mkdir(exist_ok=True)
        state_file.write_text(json.dumps(state))
    invalidated = False

    for i, (name, cmd, outputs) in enumerate(steps, 1):
        label = f"[{i}/{len(steps)}] {name}"
        if args.dry_run:
            print(f"\n{label}\n  $ {subprocess.list2cmdline([str(x) for x in cmd])}\n  outputs: {outputs}")
            continue
        # the dataset step must also rerun when forced; others skip when outputs exist
        if name != "Export weights for the Streamlit app" and not invalidated and not force and name in state["completed"] and all((ROOT / o).exists() for o in outputs):
            print(f"\n{label}: outputs exist, skipping (use --force to redo)")
            continue
        invalidated = True
        # Remove this and later completion records BEFORE executing a possibly failing step.
        earlier = {entry[0] for entry in steps[:i-1]}
        state["completed"] = [entry for entry in state["completed"] if entry in earlier]
        state_file.write_text(json.dumps(state))
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            sys.exit(f"\nSTEP FAILED: {name} (exit code {rc}). The real error is printed above. "
                     "Fix it and re-run the same command; finished steps are skipped.")
        missing = [o for o in outputs if not (ROOT / o).exists()]
        if missing:
            sys.exit(f"\nSTEP FINISHED but did not produce {missing}. Check the output above.")
        state["completed"].append(name)
        state_file.write_text(json.dumps(state))
        print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)

    if not args.dry_run:
        print(f"\nAll steps finished.\n  results: results/{yn}/ and results/{args.run_name}/comparison.md"
              "\n  the app:  streamlit run streamlit_app/app.py")


if __name__ == "__main__":
    main()
````

## `setup.ps1`

````powershell
param(
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cpu"
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Setup stopped: $Program exited with $LASTEXITCODE" }
}
Invoke-Checked -Program "python" -Arguments @("-c", "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required; Python 3.12 recommended'")
if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    Invoke-Checked -Program "python" -Arguments @("-m", "venv", ".venv")
}
$ProjectPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Invoke-Checked -Program $ProjectPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked -Program $ProjectPython -Arguments @("-m", "pip", "install", "torch", "torchvision", "--index-url", $TorchIndexUrl)
Invoke-Checked -Program $ProjectPython -Arguments @("-m", "pip", "install", "-r", "requirements.txt")
Invoke-Checked -Program $ProjectPython -Arguments @("-m", "pytest", "-q", "tests")
Write-Host "Setup and tests completed. See LOCAL_SETUP.md for training and the app."
````

## `src/app_logic.py`

````python
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
````

## `src/common.py`

````python
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
````

## `src/compare.py`

````python
"""Build the comparison table and chart from results/*/metrics.json.

    python -m src.compare --names yolo frcnn
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import CLASSES, SIZE_LABELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["yolo", "frcnn"])
    ap.add_argument("--results", default="results")
    ap.add_argument("--output", default=None, help="output directory; input remains --results")
    args = ap.parse_args()
    output = Path(args.output or args.results)
    output.mkdir(parents=True, exist_ok=True)
    ms = [json.loads((Path(args.results) / n / "metrics.json").read_text()) for n in args.names]

    units = {m.get("evaluation_unit", "tile") for m in ms}
    if len(units) != 1:
        raise ValueError("Cannot compare different evaluation units")
    def fmt(value, places=3):
        return "N/A" if value is None else f"{value:.{places}f}"
    lines = [f"Evaluation unit: {next(iter(units))}. N/A means unavailable, not zero.", "",
             "| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Macro-F1 | ms/image | FPS | Params (M) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m in ms:
        params = f"{m['n_params'] / 1e6:.1f}" if m.get("n_params") else "-"
        lines.append(f"| {m['name']} ({m['split_mode'] if 'split_mode' in m else 'unknown'}) | {fmt(m['map50'])} | {fmt(m['map50_95'])} | {fmt(m['precision'])} | "
                     f"{fmt(m['recall'])} | {fmt(m['f1_macro'])} | {fmt(m['ms_per_tile'], 1)} | {fmt(m['fps'], 1)} | {params} |")
    lines += ["", "Per-class AP@0.5:", "", "| Class | " + " | ".join(m["name"] for m in ms) + " |",
              "|---|" + "---|" * len(ms)]
    for c in CLASSES:
        lines.append(f"| {c} | " + " | ".join(fmt(m["per_class_ap50"][c]) for m in ms) + " |")
    lines += ["", "Recall by defect size (sqrt of box area, evaluation image pixels):", "",
              "| Size | " + " | ".join(m["name"] + " (n; recall)" for m in ms) + " |", "|---|" + "---|" * len(ms)]
    for s in SIZE_LABELS:
        vals = [m["size_recall"][s] for m in ms]
        lines.append(f"| {s} | " +
                     " | ".join(f"{m['size_counts'][s]}; {fmt(v)}" for m, v in zip(ms, vals)) + " |")
    out = output / "comparison.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    x, w = np.arange(len(CLASSES)), 0.8 / len(ms)
    for k, m in enumerate(ms):
        axes[0].bar(x + k * w, [np.nan if m["per_class_ap50"][c] is None else m["per_class_ap50"][c] for c in CLASSES], w, label=m["name"])
        axes[1].bar(np.arange(len(SIZE_LABELS)) + k * w,
                    [np.nan if m["size_recall"][s] is None else m["size_recall"][s] for s in SIZE_LABELS], w, label=m["name"])
    axes[0].set_xticks(x + w * (len(ms) - 1) / 2); axes[0].set_xticklabels(CLASSES, rotation=30, ha="right")
    axes[0].set_ylabel("AP@0.5"); axes[0].set_title("Per-class AP")
    axes[1].set_xticks(np.arange(len(SIZE_LABELS)) + w * (len(ms) - 1) / 2); axes[1].set_xticklabels(SIZE_LABELS)
    axes[1].set_ylabel("recall"); axes[1].set_title("Recall by defect size")
    for a in axes:
        a.set_ylim(0, 1.05); a.legend()
    fig.tight_layout(); fig.savefig(output / "comparison.png", dpi=150)
    print(f"\nsaved {out} and comparison.png")


if __name__ == "__main__":
    main()
````

## `src/download_data.py`

````python
"""Download PKU-Market-PCB automatically (Kaggle first, Hugging Face as fallback).

    python -m src.download_data
Prints the dataset root (the folder containing Annotations/ and images/).
"""
import os
from pathlib import Path


def find_dataset_root(root, depth=5):
    root = str(root)
    if not os.path.isdir(root):
        return None
    base = root.rstrip("/\\").count(os.sep)
    for d, dirs, _ in os.walk(root):
        if d.count(os.sep) - base >= depth:
            dirs[:] = []
        names = {x.lower() for x in dirs}
        if "annotations" in names and "images" in names:
            return d
    return None


def _kaggle():
    import kagglehub
    return kagglehub.dataset_download("akhatova/pcb-defects")


def _hf():
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id="KeenForgeAI/PKU-Market-PCB-corrected", repo_type="dataset")


SOURCES = (("Kaggle akhatova/pcb-defects (original, about 2 GB)", _kaggle),
           ("Hugging Face KeenForgeAI/PKU-Market-PCB-corrected", _hf))


def get_dataset(local_dir="data/raw"):
    root = find_dataset_root(local_dir)
    if root:
        return root
    errors = []
    for name, fn in SOURCES:
        try:
            print(f"Downloading from {name} ...", flush=True)
            root = find_dataset_root(fn())
            if root:
                print("Dataset ready at", root)
                return root
            errors.append(f"{name}: downloaded, but no Annotations/images folders found")
        except Exception as e:
            errors.append(f"{name}: {repr(e)[:200]}")
            print("  failed:", errors[-1], flush=True)
    raise SystemExit(
        "Automatic download failed:\n  " + "\n  ".join(errors) +
        "\nDownload it manually from https://robotics.pkusz.edu.cn/resources/datasetENG, unzip it, and pass "
        "the folder with --raw <folder containing Annotations and images>.")


if __name__ == "__main__":
    print(get_dataset())
````

## `src/evaluate.py`

````python
"""Evaluate a trained detector on the held-out test tiles.

    python -m src.evaluate --model yolo  --weights weights/yolo_best.pt  --name yolo
    python -m src.evaluate --model frcnn --weights weights/frcnn_best.pt --name frcnn

The operating confidence threshold is chosen on the VALIDATION split (best macro-F1) and then
applied unchanged to the test split. mAP uses all detections down to conf 0.001.
Outputs go to results/<name>/: metrics.json, per_class.csv, confusion_matrix.png,
conf_sweep.png, error_examples/.
"""
import hashlib
import re
import shutil
import platform
import importlib.metadata
import argparse
import json
import random
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import CLASSES, SIZE_LABELS, SIZE_EDGES, read_yolo_labels  # noqa: E402
from .infer import draw  # noqa: E402
from .metrics import compute_map, evaluate_at_conf, sweep_conf, match_image  # noqa: E402


def load_split(data_root, split, max_tiles=None):
    paths = sorted((Path(data_root) / "images" / split).glob("*.jpg"))
    if max_tiles and len(paths) > max_tiles:  # quick tests: half defect tiles, half clean tiles
        lbl = Path(data_root) / "labels" / split
        has = [p for p in paths if (lbl / f"{p.stem}.txt").stat().st_size > 0]
        has_set = set(has)
        clean = [p for p in paths if p not in has_set]
        rng = random.Random(0)
        k = max_tiles // 2
        chosen = rng.sample(has, min(k, len(has))) + rng.sample(clean, min(max_tiles - min(k, len(has)), len(clean)))
        selected = set(chosen)
        remaining = [p for p in paths if p not in selected]
        paths = sorted(chosen + rng.sample(remaining, min(max_tiles-len(chosen), len(remaining))))
    if not paths:
        raise ValueError(f"Empty {split} evaluation split under {data_root}")
    gts = []
    for p in paths:
        image = cv2.imread(str(p))
        if image is None:
            raise ValueError(f"Unreadable image: {p}")
        h, w = image.shape[:2]
        boxes, cls = read_yolo_labels(Path(data_root) / "labels" / split / f"{p.stem}.txt", w, h)
        gts.append({"boxes": boxes, "cls": cls})
    return paths, gts


def run_predictions(predictor, paths, conf=0.001, desc="predict"):
    preds = []
    for p in tqdm(paths, desc=desc):
        b, s, c = predictor.predict(cv2.imread(str(p)), conf=conf)
        preds.append({"boxes": b, "scores": s, "cls": c})
    return preds


def measure_speed(predictor, paths, n=200, warmup=10):
    if not paths:
        return None, None
    imgs = [cv2.imread(str(p)) for p in paths[:n]]
    for i in range(warmup):
        predictor.predict(imgs[i % len(imgs)], conf=0.25)
    t0 = time.perf_counter()
    for im in imgs:
        predictor.predict(im, conf=0.25)
    dt = max((time.perf_counter() - t0) / len(imgs), 1e-12)
    return dt * 1000, 1.0 / dt


class BoardPredictor:
    """Same original-board evaluation inputs for both representations."""
    def __init__(self, predictor, representation, tile=640, stride=480):
        self.base, self.representation = predictor, representation
        self.tile, self.stride = tile, stride
        self.name = predictor.name
        self.n_params = getattr(predictor, "n_params", None)

    def predict(self, image, conf=0.001):
        if self.representation == "full":
            return self.base.predict(image, conf=conf)
        from .infer import tiled_predict
        return tiled_predict(self.base, image, tile=self.tile, stride=self.stride, conf=conf)


def save_error_records(out_dir, paths, gts, preds, res, conf, unit="tile"):
    """Machine-readable observations, not inferred physical causes."""
    rows = []
    denominators = {"clean": 0, "contains_gt": 0}
    for gt in gts:
        denominators["contains_gt" if len(gt["boxes"]) else "clean"] += 1
    for kind, items in (("FN", res["fn_list"]), ("FP", res["fp_list"])):
        for idx, k in items:
            image = cv2.imread(str(paths[idx]))
            h, w = image.shape[:2]
            p, g = preds[idx], gts[idx]
            keep = p["scores"] >= conf
            boxes, classes = (g["boxes"], g["cls"]) if kind == "FN" else (p["boxes"][keep], p["cls"][keep])
            box = boxes[k]
            margin = float(min(box[0], box[1], w-box[2], h-box[3]))
            match = re.match(r"^(.*)_x(\d+)_y(\d+)$", paths[idx].stem) if unit == "tile" else None
            source = match[1] if match else paths[idx].stem
            rows.append({"kind": kind, "image": paths[idx].name, "source_image": source,
                         "board_id": source.split("_", 1)[0], "class": CLASSES[int(classes[k])],
                         "score": None if kind == "FN" else float(p["scores"][keep][k]),
                         "box": [float(v) for v in box],
                         "size_sqrt_area_px": float(np.sqrt(max(0, (box[2]-box[0])*(box[3]-box[1])))),
                         "edge_distance_px": margin, "near_edge_16px": margin < 16,
                         "tile_x": int(match[2]) if match else None,
                         "tile_y": int(match[3]) if match else None,
                         "context": "contains_gt" if len(g["boxes"]) else "clean"})
    out_dir = Path(out_dir)
    (out_dir / "errors.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    pd.DataFrame(rows, columns=["kind", "image", "source_image", "board_id", "class", "score", "box",
                               "size_sqrt_area_px", "edge_distance_px", "near_edge_16px", "tile_x", "tile_y", "context"]).to_csv(
        out_dir / "errors.csv", index=False)
    context = {c: {"n_images": n, "fp_count": sum(r["kind"] == "FP" and r["context"] == c for r in rows),
                   "fp_per_image": (sum(r["kind"] == "FP" and r["context"] == c for r in rows) / n if n else None)}
               for c, n in denominators.items()}
    (out_dir / "error_context.json").write_text(json.dumps({"unit": unit, "contexts": context,
        "note": "Near-edge counts describe error locations, not a causal error rate. Inspect crops to label visual causes."}, indent=2), encoding="utf-8")


def plot_confusion(cm, path, title):
    labels = CLASSES + ["background"]
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if norm[i, j] > 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_error_examples(out_dir, paths, gts, preds, res, conf, n=12, seed=0):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_image in out_dir.glob("*.jpg"):
        old_image.unlink()
    rng = random.Random(seed)
    for kind, items in (("FN", res["fn_list"]), ("FP", res["fp_list"])):
        for idx, k in rng.sample(items, min(n, len(items))):
            img = cv2.imread(str(paths[idx]))
            g, p = gts[idx], preds[idx]
            keep = p["scores"] >= conf
            img = draw(img, g["boxes"], cls=g["cls"], color=(0, 200, 0), thickness=2)       # GT green
            img = draw(img, p["boxes"][keep], p["scores"][keep], p["cls"][keep], color=(0, 0, 255))  # preds red
            hb = g["boxes"][k] if kind == "FN" else p["boxes"][keep][k]
            x0, y0, x1, y1 = (int(v) for v in hb)
            cv2.rectangle(img, (x0 - 6, y0 - 6), (x1 + 6, y1 + 6), (0, 255, 255), 2)         # the error, yellow
            cv2.imwrite(str(out_dir / f"{kind}_{paths[idx].stem}_{k}.jpg"), img)


def analyse(predictor, data_root, out_dir, conf="auto", name="model", do_speed=True, max_tiles=None, unit="tile", weights=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cls = len(CLASSES)
    data_root = Path(data_root)
    stats_path = data_root / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    evaluation_root = data_root / "boards" if unit == "board" else data_root
    if unit == "board":
        predictor = BoardPredictor(predictor, stats.get("representation", "tiled"),
                                   stats.get("tile", 640), stats.get("stride", 480))
    if conf != "auto" and (not np.isfinite(float(conf)) or not 0 <= float(conf) <= 1):
        raise ValueError("Confidence must be 'auto' or a finite number in [0,1]")

    conf_was_auto = conf == "auto"
    if not conf_was_auto:
        for filename in ("confidence_sweep.json", "conf_sweep.png"):
            (out_dir / filename).unlink(missing_ok=True)
    if conf_was_auto:
        v_paths, v_gts = load_split(evaluation_root, "val", max_tiles)
        if not any(len(g["cls"]) for g in v_gts):
            raise ValueError("Validation has no ground-truth defects; cannot tune confidence")
        v_preds = run_predictions(predictor, v_paths, desc="val")
        conf, curve = sweep_conf(v_gts, v_preds, n_cls)
        (out_dir / "confidence_sweep.json").write_text(json.dumps(curve), encoding="utf-8")
        plt.figure(figsize=(5, 3.5)); plt.plot(*zip(*curve), marker="o")
        plt.axvline(conf, color="red", ls="--"); plt.xlabel("confidence"); plt.ylabel("macro-F1 (val)")
        plt.title(f"Operating point: {conf:.2f}"); plt.tight_layout()
        plt.savefig(out_dir / "conf_sweep.png", dpi=150); plt.close()
    conf = float(conf)

    paths, gts = load_split(evaluation_root, "test", max_tiles)
    preds = run_predictions(predictor, paths, desc="test")
    ap = compute_map(gts, preds, n_cls)
    res = evaluate_at_conf(gts, preds, n_cls, conf)
    pc = res["per_class"]

    rows = []
    for i, c in enumerate(CLASSES):
        rows.append({"class": c, "AP50": ap[0, i], "AP50-95": np.nanmean(ap[:, i]) if np.isfinite(ap[:, i]).any() else float("nan"),
                     "precision": pc["precision"][i], "recall": pc["recall"][i], "f1": pc["f1"][i],
                     "TP": int(pc["tp"][i]), "FP": int(pc["fp"][i]), "FN": int(pc["fn"][i])})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_class.csv", index=False)
    totals = np.zeros((n_cls, len(SIZE_LABELS)), dtype=int)
    found = np.zeros_like(totals)
    for gt, pred in zip(gts, preds):
        selected = pred["scores"] >= conf
        matches = match_image(gt["boxes"], gt["cls"], pred["boxes"][selected],
                              pred["scores"][selected], pred["cls"][selected], 0.5)
        recovered = set(matches[matches >= 0].tolist())
        for j, (box, cls) in enumerate(zip(gt["boxes"], gt["cls"])):
            bucket = int(np.digitize(np.sqrt((box[2]-box[0])*(box[3]-box[1])), SIZE_EDGES))
            totals[int(cls), bucket] += 1
            found[int(cls), bucket] += int(j in recovered)
    pd.DataFrame([{"class": c, "size": label, "count": int(totals[i, k]), "found": int(found[i, k]),
                   "recall": float(found[i, k]/totals[i, k]) if totals[i, k] else None}
                  for i, c in enumerate(CLASSES) for k, label in enumerate(SIZE_LABELS)]).to_csv(
        out_dir / "size_by_class.csv", index=False)


    ms, fps = measure_speed(predictor, paths) if do_speed else (None, None)
    metrics = {
        "tile": stats.get("tile", 640), "stride": stats.get("stride", 480),
        "evaluation_unit": unit, "split_mode": stats.get("split_mode", "unknown"),
        "representation": stats.get("representation", "unknown"), "subset_evaluation": max_tiles is not None,
        "threshold_source": "validation" if conf_was_auto else "manual",
        "size_definition": "sqrt(box area) in original board pixels" if unit == "board" else "sqrt(box area) in prepared image pixels",
        "n_images": len(paths), "ms_per_image": ms,
        "name": name, "model": getattr(predictor, "name", name), "split": "test", "conf": conf,
        "n_tiles": len(paths), "n_gt": int(sum(len(g["cls"]) for g in gts)),
        "map50": float(np.nanmean(ap[0])) if np.isfinite(ap[0]).any() else float("nan"), "map50_95": float(np.nanmean(ap)) if np.isfinite(ap).any() else float("nan"),
        "precision": res["precision_micro"], "recall": res["recall_micro"], "f1_macro": res["f1_macro"],
        "per_class_ap50": {c: float(ap[0, i]) for i, c in enumerate(CLASSES)},
        "size_recall": {l: (None if np.isnan(v) else float(v)) for l, v in zip(SIZE_LABELS, res["size_recall"])},
        "size_counts": {l: int(v) for l, v in zip(SIZE_LABELS, res["size_counts"])},
        "confusion": res["confusion"].tolist(),
        "ms_per_tile": ms, "fps": fps, "n_params": getattr(predictor, "n_params", None),
    }
    def finite_json(obj):
        if isinstance(obj, dict):
            return {k: finite_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [finite_json(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    metrics = finite_json(metrics)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    save_error_records(out_dir, paths, gts, preds, res, conf, unit)
    for filename in ("stats.json", "manifest.json", "splits.csv"):
        if (data_root / filename).exists():
            shutil.copyfile(data_root / filename, out_dir / filename)
    if weights and Path(weights).with_suffix(".config.json").exists():
        shutil.copyfile(Path(weights).with_suffix(".config.json"), out_dir / "training_config.json")
    if weights and Path(weights).with_suffix(".history.csv").exists():
        shutil.copyfile(Path(weights).with_suffix(".history.csv"), out_dir / "training_history.csv")
    packages = {}
    for package in ("numpy", "torch", "torchvision", "ultralytics", "opencv-python-headless", "streamlit", "albumentations"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not installed"
    provenance = {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
                  "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
                  "weights_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest() if weights else None}
    try:
        import torch
        provenance["hardware"] = {"cuda_available": torch.cuda.is_available(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "cpu": platform.processor(), "predictor_device": str(getattr(getattr(predictor, "base", predictor), "device", "automatic"))}
    except ImportError:
        provenance["hardware"] = {"cpu": platform.processor()}
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    prediction_records = [{"image": p.name, "boxes": pred["boxes"].tolist(), "scores": pred["scores"].tolist(),
                           "cls": pred["cls"].tolist()} for p, pred in zip(paths, preds)]
    ground_truth = [{"image": p.name, "boxes": gt["boxes"].tolist(), "cls": gt["cls"].tolist()}
                    for p, gt in zip(paths, gts)]
    (out_dir / "ground_truth.json").write_text(json.dumps(ground_truth), encoding="utf-8")
    (out_dir / "predictions.json").write_text(json.dumps(prediction_records), encoding="utf-8")
    plot_confusion(res["confusion"], out_dir / "confusion_matrix.png", f"{name} (test, conf {conf:.2f})")
    save_error_examples(out_dir / "error_examples", paths, gts, preds, res, conf)

    print(f"\n{name}: mAP@0.5 {metrics['map50']} | mAP@0.5:0.95 {metrics['map50_95']} | "
          f"P {metrics['precision']:.3f} R {metrics['recall']:.3f} | FPS={fps} ({unit}) @ conf {conf:.2f}")
    print(df.round(3).to_string(index=False))
    print("recall by defect size:", metrics["size_recall"])
    return metrics


def main():
    from .predictors import load_predictor
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", choices=["tile", "board"], default="tile")
    ap.add_argument("--model", choices=["yolo", "frcnn"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/pcb_yolo")
    ap.add_argument("--conf", default="auto", help="'auto' (tuned on val) or a float")
    ap.add_argument("--name", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_tiles", type=int, default=None, help="evaluate on a subset (quick tests only)")
    args = ap.parse_args()
    name = args.name or args.model
    pred = load_predictor(args.model, args.weights, imgsz=args.imgsz, device=args.device)
    analyse(pred, args.data, Path("results") / name, args.conf, name, max_tiles=args.max_tiles, unit=args.unit, weights=args.weights)


if __name__ == "__main__":
    main()
````

## `src/export_for_app.py`

````python
"""Copy trained weights (and the evaluation summary) into models/ for the Streamlit app.

    python -m src.export_for_app --model yolo --weights weights/yolo_best.pt
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["yolo", "frcnn"], default="yolo")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--results", default=None, help="results/<name> folder (default results/<model>)")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.weights, out / f"{args.model}_best.pt")
    res = Path(args.results or f"results/{args.model}")
    for src, dst in (("metrics.json", f"metrics_{args.model}.json"),
                     ("confusion_matrix.png", f"confusion_matrix_{args.model}.png")):
        if (res / src).exists():
            shutil.copy(res / src, out / dst)
        else:
            print(f"[warn] {res / src} not found; the app will skip it")
    size = (out / f"{args.model}_best.pt").stat().st_size / 1e6
    print(f"exported to {out}/ ({size:.1f} MB weights). Commit this folder to GitHub for deployment.")
    if size > 90:
        print("[warn] file is near GitHub's 100 MB limit; use Git LFS or a smaller model.")


if __name__ == "__main__":
    main()
````

## `src/frcnn.py`

````python
"""Faster R-CNN baseline (torchvision) with anchors resized for small defects."""
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .common import CLASSES


def build_model(pretrained=True, num_classes=len(CLASSES)):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights, weights_backbone=None, min_size=800, max_size=800,  # 640px tiles are upscaled 1.25x
        box_score_thresh=0.001, box_detections_per_img=100,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # +1 background
    # Default anchors start at 32px; PCB defects are often 15-40px. Same anchor count per
    # location (3 ratios), so the pretrained RPN head shapes still load.
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    return model
````

## `src/infer.py`

````python
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
````

## `src/metrics.py`

````python
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
````

## `src/predictors.py`

````python
"""Unified predictor interface: predict(img_bgr, conf) -> (boxes xyxy, scores, cls)."""
import numpy as np


class YoloPredictor:
    def __init__(self, weights, imgsz=640, conf=0.001, iou=0.6, device=None, max_det=100):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        from .common import CLASSES, norm_name
        names = self.model.names
        labels = [names[i] for i in range(len(names))]
        if [norm_name(name) for name in labels] != CLASSES:
            raise ValueError("This checkpoint does not contain the six PCB classes in the expected order. "
                             "Train/export a PCB checkpoint; COCO starting weights are not a detector for this task.")
        self.imgsz, self.conf, self.iou, self.device, self.max_det = imgsz, conf, iou, device, max_det
        self.n_params = int(sum(p.numel() for p in self.model.model.parameters()))
        self.name = "YOLO"

    def predict(self, img_bgr, conf=None):
        r = self.model.predict(img_bgr, imgsz=self.imgsz, conf=self.conf if conf is None else conf,
                               iou=self.iou, device=self.device, max_det=self.max_det, verbose=False)[0]
        b = r.boxes
        return (b.xyxy.cpu().numpy().astype(np.float32), b.conf.cpu().numpy().astype(np.float32),
                b.cls.cpu().numpy().astype(np.int64))


class FRCNNPredictor:
    def __init__(self, weights, conf=0.001, device=None):
        import torch
        from .frcnn import build_model
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(self.device).isdigit():
            self.device = f"cuda:{self.device}"
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.to(self.device).eval()
        self.conf = conf
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))
        self.name = "Faster R-CNN"

    def predict(self, img_bgr, conf=None):
        import cv2
        torch = self.torch
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)
        with torch.no_grad():
            out = self.model([x])[0]
        boxes = out["boxes"].cpu().numpy().astype(np.float32)
        scores = out["scores"].cpu().numpy().astype(np.float32)
        cls = out["labels"].cpu().numpy().astype(np.int64) - 1  # torchvision labels are 1-based
        keep = scores >= (self.conf if conf is None else conf)
        return boxes[keep], scores[keep], cls[keep]


def load_predictor(kind, weights, **kw):
    if kind == "yolo":
        return YoloPredictor(weights, **kw)
    if kind == "frcnn":
        kw.pop("imgsz", None)
        return FRCNNPredictor(weights, **kw)
    raise ValueError(kind)
````

## `src/prepare_data.py`

````python
"""Convert PKU-Market-PCB (Pascal VOC XML) into a tiled YOLO-format dataset.

Key decisions
- Split by SOURCE IMAGE (stratified by the first defect class in each image), then tile.
  Tiling first and splitting after would leak overlapping pixels between train and test.
- Tiles are TILE x TILE with STRIDE overlap so small defects are not downscaled.
- A box is kept in a tile if >= 50% of its area is inside. A tile that cuts through a defect
  (5%-50% visible) with no fully kept box is dropped in every split, so it never teaches or
  scores a half-visible defect as background.
- Train: all tiles with defects + a small fraction of background tiles.
  Val/test: ALL clean tiles are kept so false positives are measured honestly.

Usage:
    python -m src.prepare_data --raw /path/to/PCB_DATASET --out data/pcb_yolo
"""
import argparse
import hashlib
import warnings
import csv
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .common import CLASSES, IMG_EXT, norm_name, positions

Image.MAX_IMAGE_PIXELS = None


def locate_dir(raw: Path, name: str) -> Path:
    cands = [p for p in raw.rglob("*") if p.is_dir() and p.name.lower() == name.lower()]
    if raw.name.lower() == name.lower():
        cands.append(raw)
    if not cands:
        raise FileNotFoundError(f"Could not find a '{name}' folder under {raw}. Use --ann_dir/--img_dir.")
    return min(cands, key=lambda p: len(p.parts))


def parse_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        name = norm_name(obj.findtext("name", ""))
        if name not in CLASSES:
            print(f"  [warn] unknown class '{name}' in {xml_path.name}, skipped")
            continue
        bb = obj.find("bndbox")
        x0, y0, x1, y1 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        if x1 > x0 and y1 > y0:
            boxes.append((CLASSES.index(name), x0, y0, x1, y1))
    return boxes


def split_samples(samples, ratios, seed):
    """samples: list of (stem, img_path, boxes). Stratified by first box class."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for s in samples:
        groups[s[2][0][0]].append(s)
    out = {"train": [], "val": [], "test": []}
    for _, items in sorted(groups.items()):
        items = sorted(items, key=lambda s: s[0])
        rng.shuffle(items)
        n = len(items)
        n_val = max(1, round(n * ratios[1])) if n >= 3 else 0
        n_test = max(1, round(n * ratios[2])) if n >= 3 else 0
        out["val"] += items[:n_val]
        out["test"] += items[n_val:n_val + n_test]
        out["train"] += items[n_val + n_test:]
    return out


def board_id(stem):
    """Dataset convention; verify prefixes against actual template boards."""
    prefix, sep, _ = stem.partition("_")
    if not sep or not prefix:
        raise ValueError(f"Cannot extract board prefix from {stem!r}")
    return prefix


def grouped_split(samples, ratios, seed):
    """Assign whole boards, deterministically. Ratios apply to group counts."""
    groups = defaultdict(list)
    for sample in sorted(samples, key=lambda s: s[0]):
        groups[board_id(sample[0])].append(sample)
    ids = sorted(groups)
    if len(ids) < 3:
        raise ValueError("Board splitting needs at least three distinct board prefixes")
    random.Random(seed).shuffle(ids)
    nv = max(1, round(len(ids) * ratios[1]))
    nt = max(1, round(len(ids) * ratios[2]))
    if nv + nt >= len(ids):
        raise ValueError("Split ratios leave no training boards")
    assignments = {"val": ids[:nv], "test": ids[nv:nv + nt], "train": ids[nv + nt:]}
    return {sp: [s for bid in bids for s in groups[bid]] for sp, bids in assignments.items()}


def save_board(split, stem, img, boxes, out):
    """Keep original-resolution boards and labels for a common evaluation unit."""
    image_dir = out / "boards" / "images" / split
    label_dir = out / "boards" / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    img.save(image_dir / f"{stem}.jpg", quality=95)
    w, h = img.size
    lines = [f"{c} {(a+x)/2/w:.8f} {(b+y)/2/h:.8f} {(x-a)/w:.8f} {(y-b)/h:.8f}"
             for c, a, b, x, y in boxes]
    (label_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def tile_boxes(boxes, x0, y0, tile, min_vis=0.5, min_px=2.0):
    """Return (kept boxes in tile coords, ambiguous flag)."""
    x1, y1 = x0 + tile, y0 + tile
    kept, ambiguous = [], False
    for c, bx0, by0, bx1, by1 in boxes:
        ix0, iy0, ix1, iy1 = max(bx0, x0), max(by0, y0), min(bx1, x1), min(by1, y1)
        iw, ih = ix1 - ix0, iy1 - iy0
        if iw <= 0 or ih <= 0:
            continue
        frac = (iw * ih) / max((bx1 - bx0) * (by1 - by0), 1e-6)
        if frac >= min_vis and iw >= min_px and ih >= min_px:
            kept.append((c, ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0))
        elif frac >= 0.05:
            ambiguous = True
    return kept, ambiguous


def process_split(split, items, out_root, args, rng, stats):
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for stem, img_path, boxes in items:
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        boxes = [(c, max(0, a), max(0, b), min(W, x), min(H, y)) for c, a, b, x, y in boxes
                 if min(W, x) > max(0, a) and min(H, y) > max(0, b)]
        save_board(split, stem, img, boxes, out_root)
        if args.representation == "full":
            img.save(img_dir / f"{stem}.jpg", quality=95)
            shutil.copyfile(out_root / "boards" / "labels" / split / f"{stem}.txt", lbl_dir / f"{stem}.txt")
            stats["tiles"][split] += 1
            for c, a, b, x, y in boxes:
                stats["boxes"][split][CLASSES[c]] += 1
                stats["sizes"].append(float(np.sqrt((x-a)*(y-b))))
            continue
        for y0 in positions(H, args.tile, args.stride):
            for x0 in positions(W, args.tile, args.stride):
                kept, ambiguous = tile_boxes(boxes, x0, y0, args.tile)
                # An unlabelled partial defect is unsafe even beside another kept box.
                if ambiguous:
                    stats.setdefault("dropped_ambiguous", Counter())[split] += 1
                    continue
                if not kept:
                    if split == "train" and rng.random() >= args.bg_ratio:
                        continue
                name = f"{stem}_x{x0}_y{y0}"
                img.crop((x0, y0, x0 + args.tile, y0 + args.tile)).save(img_dir / f"{name}.jpg", quality=95)
                lines = []
                for c, a, b, c2, d in kept:
                    a, b = max(a, 0), max(b, 0)
                    c2, d = min(c2, args.tile), min(d, args.tile)
                    lines.append(f"{c} {(a + c2) / 2 / args.tile:.6f} {(b + d) / 2 / args.tile:.6f} "
                                 f"{(c2 - a) / args.tile:.6f} {(d - b) / args.tile:.6f}")
                    stats["boxes"][split][CLASSES[c]] += 1
                    stats["sizes"].append(float(np.sqrt((c2 - a) * (d - b))))
                (lbl_dir / f"{name}.txt").write_text("\n".join(lines))
                stats["tiles"][split] += 1
                stats["bg_tiles"][split] += int(not kept)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="folder containing Annotations/ and images/")
    ap.add_argument("--ann_dir", default=None)
    ap.add_argument("--img_dir", default=None)
    ap.add_argument("--out", default="data/pcb_yolo")
    ap.add_argument("--split-mode", choices=["image", "board"], default="image")
    ap.add_argument("--representation", choices=["tiled", "full"], default="tiled")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--stride", type=int, default=480)
    ap.add_argument("--split", type=float, nargs=3, default=(0.70, 0.15, 0.15))
    ap.add_argument("--bg_ratio", type=float, default=0.10, help="fraction of clean train tiles to keep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean", action="store_true", help="delete --out first")
    args = ap.parse_args()

    if any(r <= 0 for r in args.split) or not np.isclose(sum(args.split), 1):
        ap.error("--split must contain three positive ratios summing to one")
    if not 0 <= args.bg_ratio <= 1:
        ap.error("--bg_ratio must be between zero and one")
    positions(1, args.tile, args.stride)  # validate before touching outputs
    raw, out = Path(args.raw), Path(args.out)
    if out.resolve() == raw.resolve() or out.resolve() in raw.resolve().parents or raw.resolve() in out.resolve().parents:
        ap.error("--out and --raw must be separate directories, not nested")
    ann_dir = Path(args.ann_dir) if args.ann_dir else locate_dir(raw, "Annotations")
    img_dir = Path(args.img_dir) if args.img_dir else locate_dir(raw, "images")
    print(f"annotations: {ann_dir}\nimages:      {img_dir}")

    index = {}
    for p in img_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            if p.stem in index:
                raise ValueError(f"Duplicate image stem {p.stem}: resolve ambiguity before splitting")
            index[p.stem] = p

    samples, missing = [], 0
    for xml_path in sorted(ann_dir.rglob("*.xml")):
        if xml_path.stem not in index:
            missing += 1
            continue
        boxes = parse_voc(xml_path)
        if boxes:
            samples.append((xml_path.stem, index[xml_path.stem], boxes))
    print(f"{len(samples)} annotated images found ({missing} XML files had no matching image)")
    if not samples:
        raise SystemExit("No samples found. Check the folder layout.")

    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    if len({s[0] for s in samples}) != len(samples):
        raise ValueError("Duplicate XML stems would duplicate samples across splits")
    splits = (grouped_split if args.split_mode == "board" else split_samples)(samples, args.split, args.seed)
    for sp, items in splits.items():
        present = {b[0] for sample in items for b in sample[2]}
        if not items:
            raise ValueError(f"Empty {sp} source split")
        absent = [c for i, c in enumerate(CLASSES) if i not in present]
        if absent:
            warnings.warn(f"{sp} has no source annotations for {absent}; AP is undefined for these classes")
    rng = random.Random(args.seed)
    stats = {"tiles": Counter(), "bg_tiles": Counter(), "boxes": defaultdict(Counter), "sizes": []}
    for split, items in splits.items():
        print(f"{split}: {len(items)} source images")
        process_split(split, items, out, args, rng, stats)

    with open(out / "splits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "split", "board_id"])
        for split, items in splits.items():
            for s in items:
                w.writerow([s[0], split, board_id(s[0])])

    yaml_text = f"path: {json.dumps(out.resolve().as_posix())}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    yaml_text += "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    (out / "pcb.yaml").write_text(yaml_text)

    sizes = np.array(stats["sizes"]) if stats["sizes"] else np.zeros(1)
    summary = {
        "split_mode": args.split_mode, "representation": args.representation,
        "board_ids": {sp: sorted({board_id(s[0]) for s in items}) for sp, items in splits.items()},
        "source_boxes_per_class": {sp: dict(Counter(CLASSES[b[0]] for s in items for b in s[2]))
                                   for sp, items in splits.items()},
        "dropped_ambiguous_tiles": dict(stats.get("dropped_ambiguous", {})),
        "source_images": {k: len(v) for k, v in splits.items()},
        "tiles": dict(stats["tiles"]),
        "background_tiles": dict(stats["bg_tiles"]),
        "boxes_per_class": {k: dict(v) for k, v in stats["boxes"].items()},
        "box_size_px_sqrt_area": {p: float(np.percentile(sizes, p)) for p in (5, 25, 50, 75, 95)},
        "tile": args.tile, "stride": args.stride, "seed": args.seed,
    }
    manifest = {"seed": args.seed, "split_mode": args.split_mode,
                "samples": [{"image": s[0], "split": sp, "board_id": board_id(s[0]),
                             "image_sha256": hashlib.sha256(s[1].read_bytes()).hexdigest()}
                            for sp, items in splits.items() for s in items]}
    manifest["annotations_sha256"] = hashlib.sha256(b"".join(
        p.read_bytes() for p in sorted(ann_dir.rglob("*.xml")))).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nDataset YAML: {out / 'pcb.yaml'}")


if __name__ == "__main__":
    main()
````

## `src/train_frcnn.py`

````python
"""Fine-tune the Faster R-CNN baseline on the same tiled dataset.

    python -m src.train_frcnn --data data/pcb_yolo --epochs 20
"""
import csv
import json
import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .common import CLASSES, read_yolo_labels
from .frcnn import build_model
from .metrics import compute_map


class TileDataset(Dataset):
    def __init__(self, root, split, train):
        self.root, self.split, self.train = Path(root), split, train
        self.paths = sorted((self.root / "images" / split).glob("*.jpg"))

    def __len__(self):
        return len(self.paths)

    def load(self, i):
        p = self.paths[i]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        boxes, cls = read_yolo_labels(self.root / "labels" / self.split / f"{p.stem}.txt", w, h)
        return img, boxes, cls

    def __getitem__(self, i):
        img, boxes, cls = self.load(i)
        h, w = img.shape[:2]
        if self.train:
            if random.random() < 0.5:
                img = img[:, ::-1]
                boxes = boxes.copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            if random.random() < 0.5:
                img = img[::-1]
                boxes = boxes.copy()
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
            img = np.ascontiguousarray(img)
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255)
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.as_tensor(cls + 1, dtype=torch.int64)}  # 0 is background
        return x, target


def collate(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def validate(model, ds, indices, device):
    model.eval()
    gts, preds = [], []
    for i in indices:
        img, boxes, cls = ds.load(i)
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(device)
        out = model([x])[0]
        gts.append({"boxes": boxes, "cls": cls})
        preds.append({"boxes": out["boxes"].cpu().numpy(), "scores": out["scores"].cpu().numpy(),
                      "cls": out["labels"].cpu().numpy() - 1})
    ap = compute_map(gts, preds, len(CLASSES), iou_thrs=[0.5])
    return float(np.nanmean(ap[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cpu, cuda, cuda:0, or 0")
    ap.add_argument("--data", default="data/pcb_yolo")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val_every", type=int, default=2)
    ap.add_argument("--val_max", type=int, default=600, help="tiles used for per-epoch validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_train", type=int, default=None, help="use only N training tiles (quick tests)")
    ap.add_argument("--out", default="weights/frcnn_best.pt")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.epochs < 1 or args.batch < 1 or args.val_every < 1 or args.val_max < 1:
        ap.error("epochs, batch, val_every and val_max must be positive")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).isdigit():
        device = f"cuda:{device}"
    train_ds = TileDataset(args.data, "train", True)
    val_ds = TileDataset(args.data, "val", False)
    if args.max_train and args.max_train < len(train_ds):
        train_ds.paths = sorted(random.Random(args.seed).sample(train_ds.paths, args.max_train))
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        collate_fn=collate, drop_last=False)
    if not len(train_ds) or not len(val_ds):
        raise ValueError("Training and validation datasets must be nonempty")
    val_idx = random.Random(args.seed).sample(range(len(val_ds)), min(args.val_max, len(val_ds)))

    model = build_model(pretrained=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    total = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / 200) * 0.5 * (1 + np.cos(np.pi * it / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.perf_counter()
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss = time.time(), 0.0
        for imgs, targets in loader:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type="cuda", enabled=str(device).startswith("cuda")):
                loss = sum(model(imgs, targets).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
        msg = f"epoch {epoch}/{args.epochs} loss {run_loss / len(loader):.4f} ({time.time() - t0:.0f}s)"
        if epoch % args.val_every == 0 or epoch == args.epochs:
            m50 = validate(model, val_ds, val_idx, device)
            msg += f" | val mAP@0.5 {m50:.4f}"
            if m50 > best:
                best = m50
                torch.save(model.state_dict(), args.out)
                msg += "  (saved best)"
        history.append({"epoch": epoch, "loss": run_loss / len(loader), "best_val_map50": best})
        print(msg, flush=True)
    Path(args.out).with_suffix(".config.json").write_text(json.dumps(vars(args) | {
        "resolved_device": device, "wall_time_seconds": time.perf_counter() - started}, indent=2), encoding="utf-8")
    with Path(args.out).with_suffix(".history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "best_val_map50"])
        writer.writeheader(); writer.writerows(history)
    print(f"best val mAP@0.5 = {best:.4f}; weights at {args.out}")


if __name__ == "__main__":
    main()
````

## `src/train_yolo.py`

````python
"""Fine-tune a COCO-pretrained YOLO on the tiled PCB dataset.

    python -m src.train_yolo --data data/pcb_yolo/pcb.yaml --model yolov8s.pt --epochs 100
"""
import time
import json
import argparse
import shutil
from pathlib import Path


def augmentation_options(enabled):
    if enabled:
        return dict(flipud=0.5, fliplr=0.5, degrees=0.0, mosaic=1.0, close_mosaic=10,
                    scale=0.25, translate=0.1, hsv_h=0.01, hsv_s=0.3, hsv_v=0.3)
    return dict(flipud=0.0, fliplr=0.0, degrees=0.0, mosaic=0.0, close_mosaic=0,
                scale=0.0, translate=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                mixup=0.0, copy_paste=0.0, shear=0.0, perspective=0.0,
                auto_augment=None, erasing=0.0)


def no_augmentation_trainer():
    # Disable the dataset augmentation pipeline itself, including optional Albumentations.
    # Merely setting mosaic/flips to zero leaves optional photometric transforms active.
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    class NoAugmentationTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            dataset = super().build_dataset(img_path, mode, batch)
            if mode == "train":
                dataset.augment = False
                dataset.transforms = dataset.build_transforms(hyp=self.args)
            return dataset
    return NoAugmentationTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augmentation", choices=["on", "off"], default="on")
    ap.add_argument("--data", default="data/pcb_yolo/pcb.yaml")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu")
    ap.add_argument("--name", default="yolo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0, help="train on this fraction of the data (quick tests)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="weights/yolo_best.pt")
    args = ap.parse_args()

    from ultralytics import YOLO
    if args.epochs < 1 or args.batch < 1 or not 0 < args.fraction <= 1:
        ap.error("epochs/batch must be positive and fraction in (0, 1]")
    model = YOLO(args.model)
    started = time.perf_counter()
    model.train(
        trainer=no_augmentation_trainer() if args.augmentation == "off" else None,
        data=str(Path(args.data).resolve()), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        patience=args.patience, seed=args.seed, device=args.device, fraction=args.fraction, workers=args.workers,
        project=str(Path("runs").resolve()), name=args.name, exist_ok=True,
        **augmentation_options(args.augmentation == "on"),
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, args.out)
    config = vars(args) | {"resolved_train_args": dict(model.trainer.args.__dict__),
                           "wall_time_seconds": time.perf_counter() - started,
                           "augmentation_pipeline": "standard" if args.augmentation == "on" else "resize_and_format_only"}
    history = Path(model.trainer.save_dir) / "results.csv"
    if history.exists():
        shutil.copyfile(history, Path(args.out).with_suffix(".history.csv"))
    Path(args.out).with_suffix(".config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(f"best weights copied to {args.out}")


if __name__ == "__main__":
    main()
````

## `streamlit_app/app.py`

````python
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
````

## `streamlit_app/requirements.txt`

````text
--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision
ultralytics==8.4.181
streamlit>=1.37
opencv-python-headless
numpy
pandas
Pillow
matplotlib
pyyaml
````

## `tests/test_app.py`

````python
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
````

## `tests/test_download.py`

````python
from src.download_data import find_dataset_root


def test_find_dataset_root_nested_and_case_insensitive(tmp_path):
    root = tmp_path / "cache" / "v1" / "PCB_DATASET"
    (root / "Annotations" / "x").mkdir(parents=True)
    (root / "images" / "x").mkdir(parents=True)
    (root / "rotation").mkdir()
    assert find_dataset_root(tmp_path) == str(root)
    assert find_dataset_root(tmp_path / "missing") is None
    lower = tmp_path / "hf"
    (lower / "annotations").mkdir(parents=True); (lower / "images").mkdir()
    assert find_dataset_root(lower) == str(lower)
````

## `tests/test_improvements.py`

````python
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
````

## `tests/test_metrics.py`

````python
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
````

## `tests/test_pipeline.py`

````python
"""End-to-end check of tiling inference + evaluation with a fake predictor (no torch needed)."""
import json
import sys

import numpy as np
from PIL import Image

from src import prepare_data
from src.evaluate import analyse
from src.infer import nms, tiled_predict
from tests.test_prepare_data import _make_raw


class OraclePredictor:
    """Returns the ground truth read from the label file of the tile it is shown (via lookup by pixels)."""
    name = "oracle"
    n_params = 0

    def __init__(self, data_root):
        import cv2
        from pathlib import Path
        from src.common import read_yolo_labels
        self.lookup = []
        for sp in ("val", "test"):
            for p in sorted((Path(data_root) / "images" / sp).glob("*.jpg")):
                img = cv2.imread(str(p))
                b, c = read_yolo_labels(Path(data_root) / "labels" / sp / f"{p.stem}.txt", 640, 640)
                self.lookup.append((img.sum(), b, c))

    def predict(self, img, conf=None):
        key = img.sum()
        for s, b, c in self.lookup:
            if s == key:
                return b.copy(), np.full(len(b), 0.9, np.float32), c.copy()
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)


def test_analyse_with_oracle_is_perfect(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "out"
    _make_raw(raw)
    monkeypatch.setattr(sys, "argv", ["p", "--raw", str(raw), "--out", str(out)])
    prepare_data.main()
    m = analyse(OraclePredictor(out), out, tmp_path / "res", conf=0.5, name="oracle", do_speed=False)
    assert np.isclose(m["map50"], 1.0) and np.isclose(m["map50_95"], 1.0)
    assert m["precision"] == 1.0 and m["recall"] == 1.0
    assert (tmp_path / "res" / "confusion_matrix.png").exists()
    assert json.loads((tmp_path / "res" / "metrics.json").read_text())["n_gt"] == m["n_gt"]


def test_tiled_predict_merges_duplicates():
    class Box40:  # reports a 40x40 box at (100,100) in whichever tile covers it
        def predict(self, crop, conf=None):
            return np.array([[100, 100, 140, 140]], np.float32), np.array([0.9], np.float32), np.array([2])
    img = np.zeros((900, 1400, 3), np.uint8)
    b, s, c = tiled_predict(Box40(), img, tile=640, stride=480)
    # every tile reports a box; after offsets + NMS, distinct locations remain but none overlap > 0.5
    assert len(b) >= 1
    keep = nms(b, s, c, 0.5)
    assert len(keep) == len(b)


def test_max_tiles_and_single_model_compare(tmp_path, monkeypatch):
    from src.evaluate import load_split
    from src import compare
    raw, out = tmp_path / "raw", tmp_path / "out"
    _make_raw(raw)
    monkeypatch.setattr(sys, "argv", ["p", "--raw", str(raw), "--out", str(out)])
    prepare_data.main()
    paths, gts = load_split(out, "test", max_tiles=10)
    assert len(paths) == 10 and any(len(g["cls"]) for g in gts) and any(len(g["cls"]) == 0 for g in gts)
    analyse(OraclePredictor(out), out, tmp_path / "results" / "oracle", conf=0.5, name="oracle", do_speed=False, max_tiles=40)
    monkeypatch.setattr(sys, "argv", ["compare", "--names", "oracle", "--results", str(tmp_path / "results")])
    compare.main()
    assert (tmp_path / "results" / "comparison.md").exists() and (tmp_path / "results" / "comparison.png").exists()
````

## `tests/test_prepare_data.py`

````python
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src import prepare_data
from src.common import CLASSES, positions


def _make_raw(root: Path, n_per_class=6, W=1400, H=900, seed=0):
    rng = np.random.default_rng(seed)
    (root / "Annotations").mkdir(parents=True)
    (root / "images").mkdir(parents=True)
    for ci, cname in enumerate(CLASSES):
        (root / "Annotations" / cname).mkdir()
        (root / "images" / cname).mkdir()
        for k in range(n_per_class):
            stem = f"{ci:02d}_{cname}_{k:02d}"
            Image.fromarray(rng.integers(0, 255, (H, W, 3), dtype=np.uint8)).save(root / "images" / cname / f"{stem}.jpg")
            objs = ""
            for _ in range(3):
                x0, y0 = int(rng.integers(0, W - 60)), int(rng.integers(0, H - 60))
                objs += (f"<object><name>{cname}</name><bndbox><xmin>{x0}</xmin><ymin>{y0}</ymin>"
                         f"<xmax>{x0 + 40}</xmax><ymax>{y0 + 40}</ymax></bndbox></object>")
            (root / "Annotations" / cname / f"{stem}.xml").write_text(f"<annotation>{objs}</annotation>")


def test_positions_cover_edges():
    assert positions(1400, 640, 480) == [0, 480, 760]
    assert positions(500, 640, 480) == [0]
    assert positions(640, 640, 480) == [0]


def test_pipeline_no_leakage_and_valid_labels(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "out"
    _make_raw(raw)
    monkeypatch.setattr(sys, "argv", ["prepare_data", "--raw", str(raw), "--out", str(out)])
    prepare_data.main()

    rows = list(csv.DictReader(open(out / "splits.csv")))
    split_of = {r["image"]: r["split"] for r in rows}
    assert len(split_of) == 36
    assert all(sum(1 for s in split_of.values() if s == sp) >= len(CLASSES) for sp in ("train", "val", "test"))

    # no source image may appear in two splits (tiles are named <stem>_x.._y..)
    seen = {}
    for sp in ("train", "val", "test"):
        for p in (out / "images" / sp).glob("*.jpg"):
            stem = p.stem.rsplit("_x", 1)[0]
            assert split_of[stem] == sp
            seen.setdefault(stem, set()).add(sp)
            assert Image.open(p).size == (640, 640)
            lab = out / "labels" / sp / f"{p.stem}.txt"
            assert lab.exists()
            for line in lab.read_text().splitlines():
                c, cx, cy, w, h = line.split()
                assert 0 <= int(c) < len(CLASSES)
                assert all(0 <= float(v) <= 1 for v in (cx, cy, w, h))
                assert float(cx) - float(w) / 2 >= -1e-5 and float(cx) + float(w) / 2 <= 1 + 1e-5
    assert all(len(v) == 1 for v in seen.values())
    assert (out / "pcb.yaml").exists() and (out / "stats.json").exists()
````
