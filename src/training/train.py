"""Train a model (optionally with knowledge distillation).

Examples
  python -m src.training.train --model resnet18 --aug none --tag teacher_resnet18
  python -m src.training.train --model tiny --aug targeted --tag tiny_targeted
  python -m src.training.train --model tiny --aug targeted --teacher artifacts/teacher_resnet18/best.pt --tag tiny_kd
"""
from __future__ import annotations
import argparse, copy, time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from src.data.dataset import make_loader
from src.evaluation.metrics import compute_metrics
from src.models.distillation import load_teacher
from src.models.model_factory import build_model, count_params
from src.training.losses import DistillationLoss
from src.utils import get_device, load_config, log, save_json, set_seed, setup_logging


@torch.no_grad()
def predict_probs(model, loader, device):
    """Return (labels, P(crack), dataset_indices) as numpy arrays."""
    model.eval()
    ys, ps, ids = [], [], []
    for x, y, i in loader:
        logits = model(x.to(device))
        ps.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy()); ys.append(y.numpy()); ids.append(i.numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(ids)


def fit(model, train_loader, val_loader, cfg, device, loss_fn, teacher=None, epochs=None, lr=None):
    tc = cfg["train"]
    epochs, lr = epochs or tc["epochs"], lr or tc["lr"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if tc["scheduler"] == "cosine" else None
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    metric, best, best_state, bad, history = tc["selection_metric"], -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        model.train(); t0, run, n = time.time(), 0.0, 0
        for x, y, _ in tqdm(train_loader, desc=f"epoch {ep}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                t_logits = teacher(x) if teacher is not None else None
                loss = loss_fn(logits, y, t_logits)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item() * len(y); n += len(y)
        if sched:
            sched.step()
        yv, pv, _ = predict_probs(model, val_loader, device)
        m = compute_metrics(yv, pv)
        history.append({"epoch": ep, "train_loss": run / n, **{f"val_{k}": m[k] for k in
                        ["accuracy", "precision", "recall", "f1", "auc", "specificity"]}, "seconds": time.time() - t0})
        score = m[metric] if m[metric] == m[metric] else -1.0
        log.info("ep %d | loss %.4f | val acc %.4f recall %.4f f1 %.4f", ep, run / n, m["accuracy"], m["recall"], m["f1"])
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tc["patience"]:
                log.info("Early stopping (no val %s gain for %d epochs)", metric, bad); break
    return history, best_state, best


def train_run(cfg, model_name, aug, tag, teacher_path=None, temperature=None, alpha=None, img_size=None):
    set_seed(cfg["seed"]); device = get_device()
    img_size = img_size or cfg["data"]["img_size"]
    out = Path(cfg["paths"]["artifacts_dir"]) / tag; out.mkdir(parents=True, exist_ok=True)
    model, init = build_model(model_name, pretrained=cfg["train"]["pretrained"]); model.to(device)
    teacher = None
    kd = {}
    if teacher_path:
        teacher, tck = load_teacher(teacher_path, device)
        kd = {"teacher": str(teacher_path), "teacher_model": tck["model_name"],
              "temperature": temperature or cfg["distill"]["temperature"], "alpha": alpha if alpha is not None else cfg["distill"]["alpha"]}
        loss_fn = DistillationLoss(kd["temperature"], kd["alpha"], cfg["train"]["label_smoothing"])
    else:
        loss_fn = DistillationLoss(1.0, 0.0, cfg["train"]["label_smoothing"])  # plain cross-entropy
    tl = make_loader(cfg, "train", aug, img_size=img_size); vl = make_loader(cfg, "val", "none", img_size=img_size)
    log.info("Training %s | aug=%s | img=%d | init=%s | params=%d | kd=%s | device=%s",
             model_name, aug, img_size, init, count_params(model), bool(teacher), device)
    history, best_state, best = fit(model, tl, vl, cfg, device, loss_fn, teacher)
    ck = {"model_name": model_name, "state_dict": best_state, "img_size": img_size, "init": init, "aug": aug,
          "kd": kd, "best_val_" + cfg["train"]["selection_metric"]: best, "seed": cfg["seed"], "tag": tag}
    torch.save(ck, out / "best.pt")
    save_json(history, out / "history.json"); save_json({"config": cfg, **{k: v for k, v in ck.items() if k != "state_dict"}}, out / "run.json")
    log.info("Saved %s (best val %s = %.4f)", out / "best.pt", cfg["train"]["selection_metric"], best)
    return out / "best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml"); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--model", required=True); ap.add_argument("--aug", default="none", choices=["none", "ordinary", "targeted"])
    ap.add_argument("--tag"); ap.add_argument("--teacher"); ap.add_argument("--temperature", type=float); ap.add_argument("--alpha", type=float)
    ap.add_argument("--img-size", type=int)
    a = ap.parse_args(); setup_logging(); cfg = load_config(a.config, a.set)
    size = a.img_size or cfg["data"]["img_size"]
    tag = a.tag or f"{a.model}_{a.aug}{'_kd' if a.teacher else ''}_{size}"
    train_run(cfg, a.model, a.aug, tag, a.teacher, a.temperature, a.alpha, a.img_size)


if __name__ == "__main__":
    main()
