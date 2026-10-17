"""Fine-tune the Faster R-CNN baseline on the same tiled dataset.

    python -m src.train_frcnn --data data/pcb_yolo --epochs 20
"""
import csv
import json
import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .common import CLASSES, read_yolo_labels
from .frcnn import build_model
from .metrics import compute_map


class TileDataset(Dataset):
    def __init__(self, root, split, train):
        self.root, self.split, self.train = Path(root), split, train
        self.paths = sorted((self.root / "images" / split).glob("*.jpg"))

    def __len__(self):
        return len(self.paths)

    def load(self, i):
        p = self.paths[i]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        boxes, cls = read_yolo_labels(self.root / "labels" / self.split / f"{p.stem}.txt", w, h)
        return img, boxes, cls

    def __getitem__(self, i):
        img, boxes, cls = self.load(i)
        h, w = img.shape[:2]
        if self.train:
            if random.random() < 0.5:
                img = img[:, ::-1]
                boxes = boxes.copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            if random.random() < 0.5:
                img = img[::-1]
                boxes = boxes.copy()
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
            img = np.ascontiguousarray(img)
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255)
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.as_tensor(cls + 1, dtype=torch.int64)}  # 0 is background
        return x, target


def collate(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def validate(model, ds, indices, device):
    model.eval()
    gts, preds = [], []
    for i in indices:
        img, boxes, cls = ds.load(i)
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(device)
        out = model([x])[0]
        gts.append({"boxes": boxes, "cls": cls})
        preds.append({"boxes": out["boxes"].cpu().numpy(), "scores": out["scores"].cpu().numpy(),
                      "cls": out["labels"].cpu().numpy() - 1})
    ap = compute_map(gts, preds, len(CLASSES), iou_thrs=[0.5])
    return float(np.nanmean(ap[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cpu, cuda, cuda:0, or 0")
    ap.add_argument("--data", default="data/pcb_yolo")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val_every", type=int, default=2)
    ap.add_argument("--val_max", type=int, default=600, help="tiles used for per-epoch validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_train", type=int, default=None, help="use only N training tiles (quick tests)")
    ap.add_argument("--out", default="weights/frcnn_best.pt")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.epochs < 1 or args.batch < 1 or args.val_every < 1 or args.val_max < 1:
        ap.error("epochs, batch, val_every and val_max must be positive")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).isdigit():
        device = f"cuda:{device}"
    train_ds = TileDataset(args.data, "train", True)
    val_ds = TileDataset(args.data, "val", False)
    if args.max_train and args.max_train < len(train_ds):
        train_ds.paths = sorted(random.Random(args.seed).sample(train_ds.paths, args.max_train))
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        collate_fn=collate, drop_last=False)
    if not len(train_ds) or not len(val_ds):
        raise ValueError("Training and validation datasets must be nonempty")
    val_idx = random.Random(args.seed).sample(range(len(val_ds)), min(args.val_max, len(val_ds)))

    model = build_model(pretrained=True).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    total = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / 200) * 0.5 * (1 + np.cos(np.pi * it / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.perf_counter()
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss = time.time(), 0.0
        for imgs, targets in loader:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type="cuda", enabled=str(device).startswith("cuda")):
                loss = sum(model(imgs, targets).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
        msg = f"epoch {epoch}/{args.epochs} loss {run_loss / len(loader):.4f} ({time.time() - t0:.0f}s)"
        if epoch % args.val_every == 0 or epoch == args.epochs:
            m50 = validate(model, val_ds, val_idx, device)
            msg += f" | val mAP@0.5 {m50:.4f}"
            if m50 > best:
                best = m50
                torch.save(model.state_dict(), args.out)
                msg += "  (saved best)"
        history.append({"epoch": epoch, "loss": run_loss / len(loader), "best_val_map50": best})
        print(msg, flush=True)
    Path(args.out).with_suffix(".config.json").write_text(json.dumps(vars(args) | {
        "resolved_device": device, "wall_time_seconds": time.perf_counter() - started}, indent=2), encoding="utf-8")
    with Path(args.out).with_suffix(".history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "best_val_map50"])
        writer.writeheader(); writer.writerows(history)
    print(f"best val mAP@0.5 = {best:.4f}; weights at {args.out}")


if __name__ == "__main__":
    main()
