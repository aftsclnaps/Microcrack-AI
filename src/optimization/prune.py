"""Structured (channel) pruning of pointwise convs via zero-masking, optional fine-tuning.

  python -m src.optimization.prune --ckpt artifacts/tiny_targeted/best.pt --amount 0.3 --finetune-epochs 3

HONEST CAVEAT: torch.nn.utils.prune zeroes channels but does not physically remove them, so file size and CPU latency
typically do NOT improve; only the count of non-zero parameters drops. Report that finding as-is. Real speed-ups need
channel removal (e.g. torch-pruning) which is outside this baseline.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.nn.utils import prune

from src.data.dataset import make_loader
from src.models.distillation import load_checkpoint_model
from src.training.losses import DistillationLoss
from src.training.train import fit
from src.utils import get_device, load_config, log, save_json, set_seed, setup_logging


def prune_pointwise(model: nn.Module, amount: float):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1) and m.out_channels > 8:
            prune.ln_structured(m, "weight", amount=amount, n=1, dim=0)
            prune.remove(m, "weight")
    return model


def nonzero_params(model):
    return int(sum((p != 0).sum().item() for p in model.parameters())), int(sum(p.numel() for p in model.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml"); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ckpt", required=True); ap.add_argument("--amount", type=float, default=0.3)
    ap.add_argument("--finetune-epochs", type=int, default=0)
    a = ap.parse_args(); setup_logging(); cfg = load_config(a.config, a.set); set_seed(cfg["seed"])
    device = get_device(); model, ck = load_checkpoint_model(a.ckpt, device); size = ck["img_size"]
    before = nonzero_params(model); prune_pointwise(model, a.amount)
    if a.finetune_epochs:
        tl = make_loader(cfg, "train", ck.get("aug", "none"), img_size=size); vl = make_loader(cfg, "val", "none", img_size=size)
        _, best, _ = fit(model, tl, vl, cfg, device, DistillationLoss(1.0, 0.0), epochs=a.finetune_epochs, lr=cfg["train"]["lr"] * 0.1)
        model.load_state_dict(best)
    after = nonzero_params(model)
    out = Path(a.ckpt).parent / f"pruned_{int(a.amount * 100)}.pt"
    torch.save({**{k: v for k, v in ck.items() if k != "state_dict"}, "state_dict": model.state_dict(), "pruned_amount": a.amount}, out)
    save_json({"amount": a.amount, "nonzero_before": before[0], "nonzero_after": after[0], "total": after[1],
               "finetune_epochs": a.finetune_epochs}, out.with_suffix(".json"))
    log.info("Pruned: non-zero params %d -> %d of %d. Saved %s", before[0], after[0], after[1], out)


if __name__ == "__main__":
    main()
