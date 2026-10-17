# Run the project locally in VS Code

## 1. One-time setup

Open the `pcb-defect-detection` folder in VS Code (File > Open Folder), then open a terminal (Ctrl+`).
Use Python 3.10, 3.11, or 3.12.

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Install PyTorch first.** Check whether you have an NVIDIA GPU with `nvidia-smi`.
- NVIDIA GPU: pick the command for your CUDA version at https://pytorch.org/get-started/locally
  (example: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`).
- No NVIDIA GPU: `pip install torch torchvision`.

Then:
```bash
pip install -r requirements.txt
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```
VS Code may ask you to select the interpreter: choose the one inside `.venv`.

## 2. Check that everything works (minutes, not hours)

```bash
python run_pipeline.py --profile quick
```
This downloads the dataset, prepares it, trains for 2 epochs on a small slice, evaluates, and exports.
The model will be weak. The point is to confirm every step runs on your machine.

## 3. Full run

```bash
python run_pipeline.py --profile gpu     # NVIDIA GPU: YOLOv8s 100 epochs + Faster R-CNN baseline
python run_pipeline.py --profile cpu     # no GPU: YOLOv8n 30 epochs, no baseline
```
You can also pick these from the Run and Debug panel (F5) in VS Code.

- Training time depends on your GPU. With no NVIDIA GPU a full run takes many hours; for the real run,
  a free GPU on Kaggle or Colab is much faster (see the tip below).
- If you stop it, run the same command again: finished steps are skipped and only the missing ones run.
  `--force` redoes everything. Changing the profile retrains automatically.
- Out of GPU memory: lower the batch size in `PROFILES` inside `run_pipeline.py`.

## 4. Results and app

- Tables and charts: `results/comparison.md`, `results/comparison.png`, `results/yolo/` (confusion matrix,
  error examples), and `data/pcb_yolo/stats.json` for dataset statistics.
- Web app: `streamlit run streamlit_app/app.py` (or "4. Streamlit app" in the Run panel). The pipeline already
  copied the trained weights into `models/`. Deployment steps are in `DEPLOY.md`.

## Using your own dataset folder

If the automatic download fails, download PKU-Market-PCB from
https://robotics.pkusz.edu.cn/resources/datasetENG, unzip it, and run:
```bash
python run_pipeline.py --profile quick --raw "C:\path\to\PCB_DATASET"
```
The folder must contain `Annotations/` and `images/`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `GPU available: False` but you have an NVIDIA GPU | You installed the CPU build. Reinstall PyTorch with the CUDA command from pytorch.org |
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project root folder, not from inside `src/` |
| Download fails | Use `--raw` with a manually downloaded dataset (see above) |
| `CUDA out of memory` | Lower `batch` for your profile in `run_pipeline.py` (for example 8 or 4) |
| Very slow data loading on Windows | Close other apps; the script already limits workers on Windows |

**Tip, free GPU:** a Kaggle notebook (Settings > Accelerator > GPU) can attach the `akhatova/pcb-defects` dataset
directly with no download. Upload this project as a dataset or `git clone` it, install with `pip install -r requirements.txt`,
and run `python run_pipeline.py --profile gpu --raw /kaggle/input/pcb-defects/PCB_DATASET`.
