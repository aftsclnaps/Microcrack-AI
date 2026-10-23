"""Torch-free dataset constants and file discovery (usable without PyTorch installed)."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

CLASS_NAMES = ["No Crack", "Crack"]
FOLDER_TO_LABEL = {"negative": 0, "positive": 1}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def discover_images(root) -> pd.DataFrame:
    """Find Positive/Negative folders under `root` (case-insensitive, recursive). Returns rel_path,label."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root '{root}' does not exist. Set paths.dataset_root in the config "
                                "(download from https://data.mendeley.com/datasets/5y9wdsg2zt/2).")
    class_dirs = {p.name.lower(): p for p in root.rglob("*") if p.is_dir() and p.name.lower() in FOLDER_TO_LABEL}
    if set(class_dirs) != set(FOLDER_TO_LABEL):
        raise FileNotFoundError(f"Expected 'Positive' and 'Negative' folders under '{root}', found: {sorted(class_dirs)}")
    rows = [{"rel_path": str(f.relative_to(root)), "label": FOLDER_TO_LABEL[name]}
            for name, d in class_dirs.items() for f in sorted(d.iterdir()) if f.suffix.lower() in IMG_EXT]
    return pd.DataFrame(rows)
