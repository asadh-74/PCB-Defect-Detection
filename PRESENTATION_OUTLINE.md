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
