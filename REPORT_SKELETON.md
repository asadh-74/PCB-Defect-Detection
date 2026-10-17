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
