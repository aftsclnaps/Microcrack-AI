"""Dataset utilities: manifest-driven PyTorch dataset and loaders."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.augment import build_transforms
from src.data.common import CLASS_NAMES, FOLDER_TO_LABEL, IMG_EXT, discover_images  # noqa: F401
from src.utils import worker_init_fn

class CrackDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, root, transform):
        self.df = manifest.reset_index(drop=True)
        self.root, self.transform = Path(root), transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row["rel_path"]).convert("RGB")
        return self.transform(img), int(row["label"]), i


def load_manifest(splits_dir, split: str) -> pd.DataFrame:
    path = Path(splits_dir) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.data.prepare_data")
    return pd.read_csv(path)


def make_loader(cfg: dict, split: str, aug: str = "none", shuffle: bool | None = None,
                batch_size: int | None = None, img_size: int | None = None) -> DataLoader:
    train = split == "train"
    img_size = img_size or cfg["data"]["img_size"]
    ds = CrackDataset(load_manifest(cfg["paths"]["splits_dir"], split), cfg["paths"]["dataset_root"],
                      build_transforms(aug, img_size, train))
    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    nw = cfg["data"]["num_workers"]
    return DataLoader(ds, batch_size=batch_size or cfg["train"]["batch_size"],
                      shuffle=train if shuffle is None else shuffle, num_workers=nw,
                      worker_init_fn=worker_init_fn, generator=g, pin_memory=torch.cuda.is_available(),
                      persistent_workers=nw > 0)
