"""Fine-tune a COCO-pretrained YOLO on the tiled PCB dataset.

    python -m src.train_yolo --data data/pcb_yolo/pcb.yaml --model yolov8s.pt --epochs 100
"""
import time
import json
import argparse
import shutil
from pathlib import Path


def augmentation_options(enabled):
    if enabled:
        return dict(flipud=0.5, fliplr=0.5, degrees=0.0, mosaic=1.0, close_mosaic=10,
                    scale=0.25, translate=0.1, hsv_h=0.01, hsv_s=0.3, hsv_v=0.3)
    return dict(flipud=0.0, fliplr=0.0, degrees=0.0, mosaic=0.0, close_mosaic=0,
                scale=0.0, translate=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                mixup=0.0, copy_paste=0.0, shear=0.0, perspective=0.0,
                auto_augment=None, erasing=0.0)


def no_augmentation_trainer():
    # Disable the dataset augmentation pipeline itself, including optional Albumentations.
    # Merely setting mosaic/flips to zero leaves optional photometric transforms active.
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    class NoAugmentationTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            dataset = super().build_dataset(img_path, mode, batch)
            if mode == "train":
                dataset.augment = False
                dataset.transforms = dataset.build_transforms(hyp=self.args)
            return dataset
    return NoAugmentationTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augmentation", choices=["on", "off"], default="on")
    ap.add_argument("--data", default="data/pcb_yolo/pcb.yaml")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu")
    ap.add_argument("--name", default="yolo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0, help="train on this fraction of the data (quick tests)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="weights/yolo_best.pt")
    args = ap.parse_args()

    from ultralytics import YOLO
    if args.epochs < 1 or args.batch < 1 or not 0 < args.fraction <= 1:
        ap.error("epochs/batch must be positive and fraction in (0, 1]")
    model = YOLO(args.model)
    started = time.perf_counter()
    model.train(
        trainer=no_augmentation_trainer() if args.augmentation == "off" else None,
        data=str(Path(args.data).resolve()), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        patience=args.patience, seed=args.seed, device=args.device, fraction=args.fraction, workers=args.workers,
        project=str(Path("runs").resolve()), name=args.name, exist_ok=True,
        **augmentation_options(args.augmentation == "on"),
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, args.out)
    config = vars(args) | {"resolved_train_args": dict(model.trainer.args.__dict__),
                           "wall_time_seconds": time.perf_counter() - started,
                           "augmentation_pipeline": "standard" if args.augmentation == "on" else "resize_and_format_only"}
    history = Path(model.trainer.save_dir) / "results.csv"
    if history.exists():
        shutil.copyfile(history, Path(args.out).with_suffix(".history.csv"))
    Path(args.out).with_suffix(".config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(f"best weights copied to {args.out}")


if __name__ == "__main__":
    main()
