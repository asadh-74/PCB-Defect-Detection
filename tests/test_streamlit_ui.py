"""Exercise the real UI with fixture detections; no model download is needed."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "streamlit_app" / "app.py"


def test_workspace_renders_without_loading_predictor():
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Inspection workspace", "Model performance", "Defect reference"]
    assert "Find the fault." in " ".join(item.value for item in app.markdown)


def test_result_filters_and_comparison_views():
    script = f'''import runpy
import numpy as np
ui = runpy.run_path({str(APP)!r})
ui["show_result"](99, "fixture.png", np.zeros((240, 320, 3), dtype=np.uint8),
    np.array([[10, 10, 50, 50], [100, 100, 160, 160]], dtype=np.float32),
    np.array([0.99, 0.98]), np.array([0, 3]))
'''
    app = AppTest.from_string(script).run(timeout=30)
    assert not app.exception
    def visible_count():
        return next(m.value for m in app.metric if m.label == "Visible detections")
    assert visible_count() == "2"
    app.multiselect[0].set_value(["short"]).run()
    assert not app.exception
    assert visible_count() == "1"
    app.radio[0].set_value("Side by side").run()
    assert not app.exception
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert visible_count() == "0"
