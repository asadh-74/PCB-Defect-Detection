# Validation — 29 October 2026

- Python 3.12, CPU-only PyTorch 2.14.1, torchvision 0.29.1, Ultralytics 8.4.181 and Streamlit 1.66.0.
- Bundled `models/yolo_best.pt` loaded and predicted on a synthetic white 640x640 image: zero detections. This checks execution, not accuracy.
- Exported the bundled checkpoint to ONNX locally with `scripts/export_onnx.py`.
- Full suite: **36 passed**, including ONNX model readiness, blank-board/export and invalid-input checks. No skipped tests in the final run.
- Seven dependency deprecation warnings remain; no test failure.
- Added limits for upload bytes, decoded pixel count and rescale factors. Untrusted checkpoint uploads are disabled unless explicitly enabled for trusted local work.
- Full-dataset accuracy and physical inspection performance were not remeasured. Historical evaluation artifacts retain their original interpretation.

Reproduce the complete suite after installing the application and test dependencies, exporting ONNX, and running `python -m pytest -q tests`. The ONNX binary is generated locally rather than committed as an additional model copy.
