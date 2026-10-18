"""Build-time export only. No model retraining or external model downloads."""
from pathlib import Path
from ultralytics import YOLO
root = Path(__file__).resolve().parent.parent
YOLO(root / 'models/yolo_best.pt').export(format='onnx', imgsz=640, opset=17, simplify=False, dynamic=False, nms=False)
