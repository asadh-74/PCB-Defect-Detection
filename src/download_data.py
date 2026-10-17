"""Download PKU-Market-PCB automatically (Kaggle first, Hugging Face as fallback).

    python -m src.download_data
Prints the dataset root (the folder containing Annotations/ and images/).
"""
import os
from pathlib import Path


def find_dataset_root(root, depth=5):
    root = str(root)
    if not os.path.isdir(root):
        return None
    base = root.rstrip("/\\").count(os.sep)
    for d, dirs, _ in os.walk(root):
        if d.count(os.sep) - base >= depth:
            dirs[:] = []
        names = {x.lower() for x in dirs}
        if "annotations" in names and "images" in names:
            return d
    return None


def _kaggle():
    import kagglehub
    return kagglehub.dataset_download("akhatova/pcb-defects")


def _hf():
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id="KeenForgeAI/PKU-Market-PCB-corrected", repo_type="dataset")


SOURCES = (("Kaggle akhatova/pcb-defects (original, about 2 GB)", _kaggle),
           ("Hugging Face KeenForgeAI/PKU-Market-PCB-corrected", _hf))


def get_dataset(local_dir="data/raw"):
    root = find_dataset_root(local_dir)
    if root:
        return root
    errors = []
    for name, fn in SOURCES:
        try:
            print(f"Downloading from {name} ...", flush=True)
            root = find_dataset_root(fn())
            if root:
                print("Dataset ready at", root)
                return root
            errors.append(f"{name}: downloaded, but no Annotations/images folders found")
        except Exception as e:
            errors.append(f"{name}: {repr(e)[:200]}")
            print("  failed:", errors[-1], flush=True)
    raise SystemExit(
        "Automatic download failed:\n  " + "\n  ".join(errors) +
        "\nDownload it manually from https://robotics.pkusz.edu.cn/resources/datasetENG, unzip it, and pass "
        "the folder with --raw <folder containing Annotations and images>.")


if __name__ == "__main__":
    print(get_dataset())
