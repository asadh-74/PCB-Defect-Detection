#!/usr/bin/env bash
set -euo pipefail
pip install -r requirements-render.txt
python -m venv /tmp/pcb-export
/tmp/pcb-export/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu torchvision==0.29.1+cpu numpy==2.2.6 pillow==12.3.0 opencv-python-headless==4.11.0.86 matplotlib pyyaml tqdm requests scipy psutil onnx==1.23.2 ultralytics-thop
/tmp/pcb-export/bin/pip install --no-deps ultralytics==8.4.181
YOLO_AUTOINSTALL=false /tmp/pcb-export/bin/python scripts/export_onnx.py
