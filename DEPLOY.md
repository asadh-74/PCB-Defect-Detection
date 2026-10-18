> A trained checkpoint is now bundled in `models/`. For the demo, install dependencies and run
> `python -m streamlit run streamlit_app/app.py`; dataset download and training are optional.
> The recorded test evidence is in `results/yolo_final/`. Back up the bundled `models/` before
> running any training pipeline, because its export step replaces the demo model.

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
