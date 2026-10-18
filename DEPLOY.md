# Deploy this trained PCB app on Streamlit Community Cloud

The trained checkpoint and its verified evaluation files are already in this repository.
Deployment does not require running Colab, downloading the dataset, or keeping VS Code open.

## Exact deployment settings

| Setting | Value |
|---|---|
| Repository | `asadh-74/PCB-Defect-Detection` |
| Branch | `main` |
| Main file path | `streamlit_app/app.py` |
| Python version (Advanced settings) | `3.12` |
| Secrets | Leave empty; this app requires no API keys |

1. Sign in at https://share.streamlit.io using the account linked to your GitHub workspace.
2. Choose **Create app**, then **Yup, I have an app** if prompted.
3. Enter the repository, branch and file path above.
4. Open **Advanced settings**, select Python **3.12**, and save.
5. Optionally choose an available descriptive app URL, then click **Deploy**.
6. Wait for the build and confirm the app displays YOLO, the validation-selected threshold 0.60,
   and the recorded test results on its About tab.
7. Upload an original bare-PCB image at 100% rescale. Confirm predictions and PNG/CSV downloads.

Streamlit searches the entrypoint directory before the repository root for dependencies. Therefore
`streamlit_app/requirements.txt` is used for this deployment; root `requirements.txt` remains the local
training/development environment. The app dependency list pins the CPU PyTorch/torchvision builds,
Ultralytics and Streamlit versions exercised during verification. Root `packages.txt` supplies the
OpenCV runtime libraries. No model download is required at inference time.

Official guidance:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

## Share and close local tools

Only after the hosted app has loaded and a test image has completed inference, copy its actual
`streamlit.app` URL for your submission. Do not substitute the local `localhost:8501` URL.
The hosted app runs independently of your computer; VS Code and the local server can then be closed.
Free hosting availability and resource limits are controlled by the provider.

## Local use and optional retraining

Run from the project root after local setup:

```powershell
& ".\.venv\Scripts\python.exe" -m streamlit run .\streamlit_app\app.py
```

The existing model was trained and tested already. Do not run training inside the cloud app.
If you retrain later, export only the selected checkpoint and its matching evaluation, then commit
those changes intentionally. The normal pipeline export replaces the local `models/` files.

## Deployment troubleshooting

- **No trained model:** confirm `models/yolo_best.pt` is present on the selected GitHub branch.
- **Wrong dependency file:** the entrypoint must be `streamlit_app/app.py`, whose adjacent requirements
  file contains the cloud CPU versions. Do not select a different Python entrypoint.
- **Missing wheel or dependency:** verify Python 3.12 in Advanced settings and inspect build logs.
- **Slow first image:** allow the model to initialize. CPU cloud inference is different from the
  Tesla T4 timing recorded in the report.
- **Resource limit:** test with one normal-resolution board at 100% rescale before uploading batches.
- **Sign-in or GitHub authorization:** complete the account's authentication/consent in Streamlit;
  never paste passwords, verification codes or API keys into chat or this repository.

A repository update alone is not a completed deployment. Verify an actual hosted URL and inference
before claiming the app is live. No public hosted URL is recorded until that verification succeeds.

## Render CPU web demo

Render can host the standalone `webapp.main:app` dashboard using Python 3.12. Build: `bash scripts/build-render.sh`. Start: `uvicorn webapp.main:app --host 0.0.0.0 --port $PORT --workers 1`.

The build exports the **bundled** trained YOLO checkpoint to ONNX in an isolated export environment. The serving process uses ONNX Runtime without loading PyTorch, reducing memory usage. The Streamlit app and training pipeline are unchanged. Runtime accepts only board images, not executable model uploads. One scan runs at a time, with 8 MB / 12 MP / 5000 px input limits. Health: `/health`; model metadata: `/model-info`.

Recorded benchmark metrics remain those of the submitted PyTorch test run; a conversion parity check does not replace a fresh ONNX benchmark. No training is performed during deployment.
