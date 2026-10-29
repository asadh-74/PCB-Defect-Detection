# PCB Inspection Studio demo — 90 seconds

1. Install `streamlit_app/requirements.txt` and run `python -m streamlit run streamlit_app/app.py`.
2. Open Model performance, explain the held-out split and distinguish mAP from accuracy.
3. Upload an unannotated bare-board image you have permission to share. Do not use previously annotated evaluation images as fresh test inputs.
4. Show detected regions, class counts, confidence threshold and class filtering.
5. Export the annotated PNG and table. Explain a false positive or missed defect honestly.
6. Close with limitations: this is a six-class research/portfolio detector, not certified manufacturing inspection.

The existing `results/yolo_final/error_examples/` files are historical annotated evaluation evidence. Label them as such. The current verification also includes a synthetic blank-image smoke test; it is not an accuracy evaluation.
