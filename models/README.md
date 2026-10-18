# Trained PCB demo model

`yolo_best.pt` is the submitted six-class PCB checkpoint, not the COCO starting model.
The exported `metrics_yolo.json` and `confusion_matrix_yolo.png` correspond to the full-board
held-out test run in `results/yolo_final/`.

Checkpoint SHA256: `eccc897d54cee32aa27ac2b7b59cd35e0db1764ae0cff772980088895b1e3cc8`.

Run `python -m streamlit run streamlit_app/app.py` after installation. CPU inference is supported;
no training or dataset download is required to start the demo. No hosted app is deployed by this update.
