"""CPU benchmark: params, serialized size, preprocessing and model-forward latency (batch 1), throughput.

  python -m src.evaluation.benchmark --ckpt artifacts/tiny_targeted/best.pt
Notes: CPU and GPU latency are different quantities; this script measures CPU only. Run every model on the SAME machine.
"""
from __future__ import annotations
import argparse, io, platform, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from src.data.augment import build_transforms
from src.evaluation.evaluate import load_for_eval
from src.utils import load_config, log, save_json, setup_logging


def model_size_mb(model, path: Path | None = None) -> float:
    if path is not None and path.suffix == ".ts":
        return path.stat().st_size / 1e6
    buf = io.BytesIO(); torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def _pct(a, q):
    return float(np.percentile(a, q))


@torch.inference_mode()
def benchmark(model, img_size, bench_cfg, sample_img: Image.Image | None = None):
    torch.set_num_threads(bench_cfg["threads"]); model.eval()
    tf = build_transforms("none", img_size, train=False)
    img = sample_img or Image.fromarray(np.random.RandomState(0).randint(0, 255, (227, 227, 3), dtype=np.uint8))
    x = tf(img).unsqueeze(0)
    for _ in range(bench_cfg["warmup"]):
        model(x)
    pre, fwd = [], []
    for _ in range(bench_cfg["runs"]):
        t0 = time.perf_counter(); xi = tf(img).unsqueeze(0); t1 = time.perf_counter(); model(xi); t2 = time.perf_counter()
        pre.append((t1 - t0) * 1e3); fwd.append((t2 - t1) * 1e3)
    xb = x.repeat(bench_cfg["throughput_batch"], 1, 1, 1)
    model(xb); t0 = time.perf_counter(); reps = 10
    for _ in range(reps):
        model(xb)
    thr = reps * xb.shape[0] / (time.perf_counter() - t0)
    return {"forward_ms_median": _pct(fwd, 50), "forward_ms_p95": _pct(fwd, 95), "preprocess_ms_median": _pct(pre, 50),
            "preprocess_ms_p95": _pct(pre, 95), "throughput_img_per_s": thr, "runs": len(fwd), "warmup": bench_cfg["warmup"]}


def environment(threads):
    return {"machine": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
            "torch": torch.__version__, "threads": threads, "device": "cpu", "precision_note": "fp32 unless model is INT8-quantized"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml"); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ckpt", required=True); ap.add_argument("--precision", default="fp32")
    a = ap.parse_args(); setup_logging(); cfg = load_config(a.config, a.set)
    ckpt = Path(a.ckpt); model, img_size, _ = load_for_eval(ckpt)
    res = benchmark(model, img_size, cfg["benchmark"])
    res.update({"params": int(sum(p.numel() for p in model.parameters())), "size_mb": model_size_mb(model, ckpt),
                "img_size": img_size, "precision": a.precision, "environment": environment(cfg["benchmark"]["threads"])})
    save_json(res, ckpt.parent / f"benchmark_{ckpt.stem}.json")
    log.info("%s | %.0f params | %.2f MB | fwd median %.2f ms (p95 %.2f) | pre %.2f ms | %.0f img/s", ckpt.name, res["params"],
             res["size_mb"], res["forward_ms_median"], res["forward_ms_p95"], res["preprocess_ms_median"], res["throughput_img_per_s"])


if __name__ == "__main__":
    main()
