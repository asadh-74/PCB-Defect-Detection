"""Small public CPU inference demo. The local Streamlit training/demo remains available."""
import base64
import io
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from src.common import CLASSES
from src.app_logic import annotate, build_table, counts, detect, encode_png, filter_conf
from .predictor import OnnxPredictor

ROOT = Path(__file__).resolve().parent.parent
predictor = None
lock = threading.Lock()
Image.MAX_IMAGE_PIXELS = 12_000_000

@asynccontextmanager
async def lifespan(app):
    global predictor
    predictor = OnnxPredictor(ROOT / 'models/yolo_best.onnx')
    yield

app = FastAPI(title='PCB Defect Detection', lifespan=lifespan)

@app.get('/', include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / 'static/index.html')

@app.get('/health')
def health():
    return {'status':'ok' if predictor else 'starting', 'model_loaded': predictor is not None, 'runtime':'ONNX CPU'}

@app.get('/model-info')
def model_info():
    m = json.loads((ROOT / 'models/metrics_yolo.json').read_text())
    return {'classes': CLASSES, 'conf': m['conf'], 'map50':m['map50'], 'map50_95':m['map50_95'], 'evaluation':'Recorded PyTorch held-out board-group test; ONNX parity checked separately.'}

@app.post('/detect')
def scan(file: UploadFile = File(...), confidence: float = Query(.6, ge=.01, le=.95)):
    if not lock.acquire(blocking=False):
        raise HTTPException(429, 'Another board is being scanned. Please retry shortly.')
    try:
        data = file.file.read(8 * 1024 * 1024 + 1)
        if len(data) > 8 * 1024 * 1024:
            raise HTTPException(413, 'Image exceeds 8 MB.')
        try:
            image = Image.open(io.BytesIO(data))
            w, h = image.size
            if w*h > 12_000_000 or max(w,h) > 5000:
                raise HTTPException(413, 'Use an image no larger than 12 megapixels and 5000 pixels per side.')
            image = ImageOps.exif_transpose(image).convert('RGB')
            bgr = np.ascontiguousarray(np.asarray(image)[:,:,::-1])
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
            raise HTTPException(400, 'Upload a valid PNG or JPEG board image.')
        b,s,c = detect(predictor,bgr)
        b,s,c = filter_conf(b,s,c,confidence)
        return {'counts':counts(c),'detections':build_table(b,s,c).to_dict(orient='records'), 'annotated_png':base64.b64encode(encode_png(annotate(bgr,b,s,c))).decode(), 'width':bgr.shape[1], 'height':bgr.shape[0]}
    finally:
        lock.release()
