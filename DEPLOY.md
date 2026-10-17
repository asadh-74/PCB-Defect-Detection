# Deploy the app on Streamlit Community Cloud

## 1. Export the trained model (in Colab, after training and evaluation)

```python
!python -m src.export_for_app --model yolo --weights weights/yolo_best.pt
!zip -r models.zip models
from google.colab import files; files.download('models.zip')
```

Unzip `models.zip` into the `models/` folder of your local copy of this project. It should contain
`yolo_best.pt` (about 22 MB for yolov8s) plus `metrics_yolo.json` and `confusion_matrix_yolo.png`.

## 2. Test locally

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

Upload a board image from the dataset. If you have no `models/` folder yet, the sidebar lets you upload a `.pt` file.

## 3. Push to GitHub

```bash
git init
git add .
git commit -m "PCB defect detection app"
git branch -M main
git remote add origin https://github.com/<your-username>/pcb-defect-detection.git
git push -u origin main
```

Check that `models/yolo_best.pt` is in the commit (`git ls-files models`). The `.gitignore` already excludes
datasets, training runs, and the `weights/` folder, but not `models/`.

## 4. Deploy

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Create app, then pick the repository, branch `main`, and main file path `streamlit_app/app.py`.
3. Open Advanced settings and choose Python 3.11 (safest for the PyTorch CPU wheels).
4. Deploy. The first build takes several minutes because it installs PyTorch.

`streamlit_app/requirements.txt` installs the CPU build of PyTorch (no GPU on the free tier) and
`packages.txt` installs the system libraries OpenCV needs.

## If the build runs out of memory or times out

Streamlit Community Cloud has limited RAM. Hugging Face Spaces gives more room and takes the same code:

1. Create a new Space with the Streamlit SDK.
2. Push this repo to it. Put this at the top of the Space README.md:

```
---
title: PCB Defect Detector
sdk: streamlit
app_file: streamlit_app/app.py
python_version: "3.11"
---
```

3. Copy `streamlit_app/requirements.txt` to the repo root as `requirements.txt` and `packages.txt` stays at the root.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `libGL.so.1` import error | Make sure `packages.txt` is at the repo root and redeploy |
| "No trained model is available yet" | `models/yolo_best.pt` was not committed; check `git ls-files models` |
| Slow on big boards | Free tier is CPU only; a 3000x1600 board is about 20 tiles. Lower the rescale slider |
| File over 100 MB rejected by GitHub | Use a smaller model (yolov8s or yolov8n) or Git LFS |
