"""CPU ONNX adapter for the bundled YOLOv8 detector; same tile and NMS protocol."""
import ast
import cv2
import numpy as np
import onnxruntime as ort
from src.common import CLASSES, norm_name
from src.infer import nms


class OnnxPredictor:
    def __init__(self, path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        names = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map['names'])
        if [norm_name(names[i]) for i in range(len(names))] != CLASSES:
            raise ValueError('Export must contain the six PCB classes in the expected order')
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, bgr, conf=0.001):
        h, w = bgr.shape[:2]
        ratio = min(640 / h, 640 / w)
        nw, nh = round(w * ratio), round(h * ratio)
        image = cv2.resize(bgr, (nw, nh)) if (nw, nh) != (w, h) else bgr
        dw, dh = (640 - nw) / 2, (640 - nh) / 2
        image = cv2.copyMakeBorder(image, round(dh - .1), round(dh + .1), round(dw - .1), round(dw + .1), cv2.BORDER_CONSTANT, value=(114,114,114))
        x = np.ascontiguousarray(image[:, :, ::-1].transpose(2,0,1)[None], dtype=np.float32) / 255
        pred = self.session.run(None, {self.input_name: x})[0][0].T
        classes = np.argmax(pred[:, 4:], axis=1)
        scores = pred[np.arange(len(pred)), classes + 4]
        keep = scores >= conf
        pred, scores, classes = pred[keep], scores[keep], classes[keep]
        boxes = pred[:, :4].copy()
        boxes[:, :2] -= pred[:, 2:4] / 2
        boxes[:, 2:4] += boxes[:, :2]
        boxes[:, [0,2]] = (boxes[:, [0,2]] - round(dw - .1)) / ratio
        boxes[:, [1,3]] = (boxes[:, [1,3]] - round(dh - .1)) / ratio
        boxes[:, [0,2]] = boxes[:, [0,2]].clip(0,w)
        boxes[:, [1,3]] = boxes[:, [1,3]].clip(0,h)
        idx = nms(boxes, scores, classes, .6)
        idx = idx[np.argsort(-scores[idx])][:100]
        return boxes[idx].astype(np.float32), scores[idx].astype(np.float32), classes[idx].astype(np.int64)
