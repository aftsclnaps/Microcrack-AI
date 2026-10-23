"""Evaluate a checkpoint (.pt) or compressed TorchScript model (.ts) on val/test.

  python -m src.evaluation.evaluate --ckpt artifacts/tiny_targeted/best.pt --split test
  python -m src.evaluation.evaluate --ckpt artifacts/tiny_targeted/best.pt --split test --calibrate
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.data.dataset import load_manifest, make_loader
from src.evaluation.metrics import (category_metrics, choose_review_confidence, compute_metrics,
                                    expected_calibration_error, pick_threshold_for_recall)
from src.models.distillation import load_checkpoint_model
from src.utils import get_device, load_config, log, save_json, setup_logging


def load_for_eval(path, device="cpu"):
    """Return (model, img_size, meta). Supports state-dict checkpoints and TorchScript (.ts) compressed models."""
    path = Path(path)
    if path.suffix == ".ts":
        meta = {}
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            import json; meta = json.loads(meta_path.read_text())
        return torch.jit.load(str(path), map_location="cpu").eval(), meta.get("img_size", 227), meta
    model, ck = load_checkpoint_model(path, device)
    return model, ck["img_size"], ck


def run_split(model, cfg, split, img_size, device):
    from src.training.train import predict_probs
    loader = make_loader(cfg, split, "none", shuffle=False, img_size=img_size, batch_size=64)
    y, p, idx = predict_probs(model, loader, device)
    df = load_manifest(cfg["paths"]["splits_dir"], split).iloc[idx].reset_index(drop=True)
    df["prob_crack"] = p
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml"); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ckpt", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--annotations", default="data/annotations/test_annotations.csv")
    ap.add_argument("--threshold", type=float); ap.add_argument("--calibrate", action="store_true",
                    help="choose threshold + review cut-off on VALIDATION data, then apply to --split")
    ap.add_argument("--tag"); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(); setup_logging(); cfg = load_config(a.config, a.set)
    ckpt = Path(a.ckpt); out = ckpt.parent
    device = torch.device("cpu") if (a.cpu or ckpt.suffix == ".ts") else get_device()
    model, img_size, _ = load_for_eval(ckpt, device)
    thr = a.threshold if a.threshold is not None else cfg["eval"]["threshold"]
    calib = {}
    if a.calibrate:
        v = run_split(model, cfg, "val", img_size, device)
        thr = pick_threshold_for_recall(v.label, v.prob_crack, cfg["eval"]["recall_target"])
        calib = {"threshold": thr, "review": choose_review_confidence(v.label, v.prob_crack, thr, cfg["eval"]["review_accuracy_target"]),
                 "val_ece": expected_calibration_error(v.label, v.prob_crack), "chosen_on": "validation"}
        save_json(calib, out / f"calibration_{ckpt.stem}.json")
        log.info("Calibrated on val: %s", calib)
    df = run_split(model, cfg, a.split, img_size, device)
    ann = Path(a.annotations)
    if a.split == "test" and ann.exists():
        adf = pd.read_csv(ann)
        df = df.merge(adf[["image_path", "visibility_category"]].rename(columns={"image_path": "rel_path"}), on="rel_path", how="left")
    else:
        log.warning("No annotation CSV at %s -> difficulty-stratified metrics will be skipped (see annotate_difficulty.py)", ann)
    res = category_metrics(df, thr); res["ece"] = expected_calibration_error(df.label, df.prob_crack)
    res.update({"checkpoint": str(ckpt), "split": a.split, "calibration": calib})
    df["pred"] = (df.prob_crack >= thr).astype(int)
    suffix = f"{ckpt.stem}_{a.split}"
    df.to_csv(out / f"predictions_{suffix}.csv", index=False)
    save_json(res, out / f"eval_{suffix}.json")
    m = res["all"]
    log.info("%s | acc %.4f prec %.4f recall %.4f f1 %.4f spec %.4f FNR %.4f AUC %.4f", a.split, m["accuracy"], m["precision"],
             m["recall"], m["f1"], m["specificity"], m["fnr"], m["auc"])


if __name__ == "__main__":
    main()
