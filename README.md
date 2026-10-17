# PCB Defect Detection

Detect and classify six bare-PCB defect types (missing hole, mouse bite, open circuit, short, spur,
spurious copper) with a fine-tuned YOLO, compared against a Faster R-CNN baseline.
AI Bootcamp 18 (Atomcamp) final project, Computer Vision track.

## What is in here

```
run_pipeline.py       one command for the whole pipeline (profiles: quick, cpu, gpu)
src/download_data.py  automatic dataset download (Kaggle, then Hugging Face)
src/prepare_data.py   VOC XML -> tiled YOLO dataset, split by source image
src/train_yolo.py     fine-tune COCO-pretrained YOLO (Ultralytics)
src/train_frcnn.py    fine-tune COCO-pretrained Faster R-CNN (torchvision), small-object anchors
src/metrics.py        mAP / precision / recall / confusion matrix in numpy (same code scores both models)
src/evaluate.py       test-split evaluation, size-bucket recall, confusion matrix, error examples, FPS
src/compare.py        side-by-side table and chart
src/infer.py          sliding-window inference for full-size boards
src/app_logic.py      UI-independent app logic (decode, detect, table, crops)
src/export_for_app.py copy trained weights + metrics into models/ for the app
streamlit_app/app.py  Streamlit web app (upload a PCB image, get defects)
models/               exported weights used by the deployed app
DEPLOY.md             step-by-step Streamlit Community Cloud deployment
notebooks/colab_run.ipynb   runs the whole pipeline on a free Colab GPU
tests/                15 tests (metrics, data prep, tiling, evaluation, app logic, app boot)
```

## Get the dataset

PKU-Market-PCB (Peking University). Download it from the official page:
https://robotics.pkusz.edu.cn/resources/datasetENG ("Printed circuit board (PCB) defect dataset").
Unzip it so you have a folder containing `Annotations/` and `images/`. The dataset has 1,386 images in total;
the annotated defect images are the ones with an XML file. Only those are used. The script ignores any
`rotation` and template folders because it reads only `Annotations/` and `images/`.
If your folder names differ, pass `--ann_dir` and `--img_dir`.

## Run it

Easiest: follow `LOCAL_SETUP.md` (VS Code, one command: `python run_pipeline.py --profile quick`).
The individual steps are below.

```bash
pip install -r requirements.txt
python -m pytest -q tests                                   # sanity check, no GPU needed

python -m src.prepare_data --raw /path/to/PCB_DATASET --out data/pcb_yolo
python -m src.train_yolo   --data data/pcb_yolo/pcb.yaml --epochs 100
python -m src.train_frcnn  --data data/pcb_yolo --epochs 20
python -m src.evaluate --model yolo  --weights weights/yolo_best.pt  --name yolo
python -m src.evaluate --model frcnn --weights weights/frcnn_best.pt --name frcnn
python -m src.compare --names yolo frcnn
python -m src.export_for_app --model yolo --weights weights/yolo_best.pt
streamlit run streamlit_app/app.py                          # web app, see DEPLOY.md to host it
```

On Colab, open `notebooks/colab_run.ipynb` and run the cells top to bottom (GPU runtime).

## Design decisions (and why)

- **Split before tiling.** Images are split 70/15/15 by source image, stratified by defect class, then tiled.
  Tiling first would put overlapping pixels of the same board in train and test and inflate every metric.
- **640 px tiles, 480 px stride.** PKU boards are about 3000x1600 px and defects are often 15-40 px.
  Downscaling a whole board to 640 px would erase them. Tiling keeps native resolution.
- **Boxes cut by a tile edge.** A box is kept if at least 50% of it is inside the tile. A tile that cuts a
  defect without a kept box is dropped, so half-visible defects are never scored as background.
- **Honest negatives.** Train keeps 10% of clean tiles. Val and test keep all clean tiles, so false
  positives are measured against the real background rate.
- **Operating threshold from validation only.** Confidence is chosen for best macro-F1 on val and applied
  unchanged to test. mAP uses all detections down to conf 0.001.
- **One scorer for both models.** `src/metrics.py` computes COCO-style AP (101-point, IoU 0.5:0.95) for
  YOLO and Faster R-CNN alike, so the comparison is like for like.
- **Faster R-CNN anchors** start at 16 px instead of 32 px, to match defect size.

## Results

Not yet run. Fill this in after training; `results/comparison.md` is generated for you.

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | FPS |
|---|---|---|---|---|---|
| YOLO | | | | | |
| Faster R-CNN | | | | | |

## Final report outline

1. Problem and motivation (reuse the proposal).
2. Data: source, class counts, box-size distribution (`data/pcb_yolo/stats.json`), split method.
3. Method: tiling, augmentation, YOLO fine-tune settings, Faster R-CNN baseline.
4. Results: `results/comparison.md`, `comparison.png`, per-class table, confusion matrices.
5. Error analysis: look at `results/*/error_examples/` and `size_recall`. Which defect types are missed,
   at what size, and which classes get confused (usually short vs. spur, mouse bite vs. open circuit).
6. Limitations and next steps: single dataset, synthetic defects, tile-level evaluation, no real-line test.

## Limitations

- Metrics are tile-level. Full-board behaviour (tile merging) is shown in the demo but is not scored.
- PKU-Market-PCB defects are synthetically produced on clean boards, so real-line performance is untested.
- The dataset is small (hundreds of boards). Expect variance between runs; report the seed.
