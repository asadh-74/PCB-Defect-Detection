import io
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from webapp.main import app

@pytest.fixture(scope='module')
def client():
    if not (Path(__file__).parent.parent/'models/yolo_best.onnx').exists():
        pytest.skip('Build-time ONNX export required')
    with TestClient(app) as c:
        yield c

def test_model_ready(client):
    assert client.get('/health').json()['model_loaded']
    assert client.get('/').status_code==200
    assert len(client.get('/model-info').json()['classes'])==6

def test_blank_board_and_exports(client):
    image=io.BytesIO();Image.fromarray(np.zeros((640,640,3),dtype=np.uint8)).save(image,format='PNG')
    r=client.post('/detect',files={'file':('blank.png',image.getvalue(),'image/png')})
    assert r.status_code==200
    d=r.json();assert d['width']==640 and d['height']==640
    assert d['detections']==[]
    import base64
    assert base64.b64decode(d['annotated_png']).startswith(b'\x89PNG')

def test_bad_image_and_threshold(client):
    assert client.post('/detect',files={'file':('bad.png',b'bad','image/png')}).status_code==400
    assert client.post('/detect?confidence=2',files={'file':('bad.png',b'bad','image/png')}).status_code==422
