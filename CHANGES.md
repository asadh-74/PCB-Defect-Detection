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


## Trained-artifact follow-up

The user supplied `pcb_trained_artifacts.zip` after completing real Colab training and evaluation.
The original pending/synthetic-only statements above describe the earlier source-only update; they
are superseded by the recorded real run in `results/yolo_final/`.

- `models/yolo_best.pt`: actual six-class checkpoint; checkpoint SHA256 verified against provenance.
- `models/metrics_yolo.json` and `models/confusion_matrix_yolo.png`: matching app exports.
- `models/README.md`: explain bundled checkpoint and CPU demo startup.
- `README.md`: real test table, class scores, split membership and evidence links; demo no longer requires training.
- `LOCAL_SETUP.md`, `DEPLOY.md`: add immediate demo instructions and preserve-model note before optional retraining.
- `results/yolo_final/README.md`: supported analysis, protocol, size support, environment distinction and limitations.
- All submitted `results/yolo_final/` files: retain original predictions, ground truth, metrics, manifests,
  split tables, training history/settings, plots, error observations and error example images unchanged.
- `FULL_FILE_CONTENTS.md`: regenerate current pasteable text; binary artifacts remain separate files.

Validation: recomputed mAP50, mAP50:95, precision and recall from submitted predictions/ground truth;
verified disjoint prefix groups, matching app metrics, 60 training-history rows, checkpoint hash,
six-class loading and actual CPU inference on a blank image (execution check only). The recorded
report numbers remain the original Tesla T4 test run; no CPU performance claim is substituted.

Exact artifact paths added from the submitted ZIP:

- `results/yolo_final/README.md`
- `results/yolo_final/conf_sweep.png`
- `results/yolo_final/confidence_sweep.json`
- `results/yolo_final/confusion_matrix.png`
- `results/yolo_final/error_context.json`
- `results/yolo_final/error_examples/FN_05_short_06_1.jpg`
- `results/yolo_final/error_examples/FN_05_spurious_copper_07_0.jpg`
- `results/yolo_final/error_examples/FN_05_spurious_copper_07_2.jpg`
- `results/yolo_final/error_examples/FN_05_spurious_copper_08_1.jpg`
- `results/yolo_final/error_examples/FN_05_spurious_copper_09_0.jpg`
- `results/yolo_final/error_examples/FN_05_spurious_copper_09_1.jpg`
- `results/yolo_final/error_examples/FN_05_spurious_copper_10_0.jpg`
- `results/yolo_final/error_examples/FN_05_spurious_copper_10_3.jpg`
- `results/yolo_final/error_examples/FN_11_missing_hole_08_1.jpg`
- `results/yolo_final/error_examples/FN_11_short_06_4.jpg`
- `results/yolo_final/error_examples/FN_11_short_08_2.jpg`
- `results/yolo_final/error_examples/FN_11_short_10_4.jpg`
- `results/yolo_final/error_examples/FP_05_short_06_0.jpg`
- `results/yolo_final/error_examples/FP_05_short_06_2.jpg`
- `results/yolo_final/error_examples/FP_05_short_10_2.jpg`
- `results/yolo_final/error_examples/FP_05_spur_10_4.jpg`
- `results/yolo_final/error_examples/FP_11_missing_hole_03_2.jpg`
- `results/yolo_final/error_examples/FP_11_missing_hole_05_4.jpg`
- `results/yolo_final/error_examples/FP_11_missing_hole_08_2.jpg`
- `results/yolo_final/error_examples/FP_11_open_circuit_02_1.jpg`
- `results/yolo_final/error_examples/FP_11_short_06_2.jpg`
- `results/yolo_final/error_examples/FP_11_short_07_0.jpg`
- `results/yolo_final/error_examples/FP_11_short_08_0.jpg`
- `results/yolo_final/error_examples/FP_11_spurious_copper_02_4.jpg`
- `results/yolo_final/errors.csv`
- `results/yolo_final/errors.json`
- `results/yolo_final/ground_truth.json`
- `results/yolo_final/manifest.json`
- `results/yolo_final/metrics.json`
- `results/yolo_final/per_class.csv`
- `results/yolo_final/predictions.json`
- `results/yolo_final/provenance.json`
- `results/yolo_final/size_by_class.csv`
- `results/yolo_final/splits.csv`
- `results/yolo_final/stats.json`
- `results/yolo_final/training_args.yaml`
- `results/yolo_final/training_history.csv`


## Streamlit deployment preparation

- `streamlit_app/requirements.txt`: pin verified CPU PyTorch/torchvision builds and Streamlit,
  retain the tested Ultralytics version, and bound other app dependency major versions. The root
  training dependency list is unchanged. This prevents a generic Torch resolution choosing CUDA
  packages on the CPU-only hosting service.
- `DEPLOY.md`: add exact repository, branch, entrypoint, Python version, empty-secrets setup,
  build checks and live test instructions. Authentication/deployment status must be verified separately.
- `FULL_FILE_CONTENTS.md`: current complete text for the changed requirements and deployment guide.

The model and evaluation results are unchanged. These edits prepare deployment; they do not by
themselves demonstrate that a Streamlit account has signed in or that a hosted app is running.
