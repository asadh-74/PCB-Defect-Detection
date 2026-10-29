import io
import pytest
from PIL import Image
from src import app_logic

def image_bytes():
    out = io.BytesIO()
    Image.new('RGB', (20, 20)).save(out, format='PNG')
    return out.getvalue()

def test_reject_large_decoded_image_before_conversion(monkeypatch):
    monkeypatch.setattr(app_logic, 'MAX_PIXELS', 100)
    with pytest.raises(ValueError, match='megapixel'):
        app_logic.decode_image(image_bytes())

def test_upload_byte_and_scale_limits(monkeypatch):
    for scale in (float('nan'), float('inf'), 0, -1, 3):
        with pytest.raises(ValueError, match='Rescale'):
            app_logic.decode_image(image_bytes(), scale)
    monkeypatch.setattr(app_logic, 'MAX_UPLOAD_BYTES', 1)
    with pytest.raises(ValueError, match='upload limit'):
        app_logic.decode_image(image_bytes())
