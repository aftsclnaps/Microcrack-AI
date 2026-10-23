"""Shared helpers: config loading, data-root resolution, seeding, environment info."""
from __future__ import annotations

import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply ``a.b.c=value`` overrides (value parsed as YAML: 1, 0.01, true, null, text)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
        key, val = item.split("=", 1)
        keys = key.strip().split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = yaml.safe_load(val)
    return cfg


def resolve_data_root(cli_value: str | None, cfg: dict) -> Path:
    """Priority: CLI flag > env MICROCRACK_DATA_ROOT > config paths.dataset_root. Never guessed."""
    value = cli_value or os.environ.get("MICROCRACK_DATA_ROOT") or cfg.get("paths", {}).get("dataset_root")
    if not value:
        raise SystemExit(
            "Dataset root is not set. Pass --data-root /path/to/dataset, set the environment "
            "variable MICROCRACK_DATA_ROOT, or set paths.dataset_root in configs/config.yaml.\n"
            "See data/README.md for download instructions."
        )
    root = Path(value).expanduser()
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist or is not a directory: {root}")
    return root


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python/numpy/torch. Full determinism is NOT guaranteed (see README)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def env_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        import torchvision

        info.update(
            torch=torch.__version__,
            torchvision=torchvision.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            torch_threads=torch.get_num_threads(),
        )
    except ImportError:
        info["torch"] = None
    return info
