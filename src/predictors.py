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
