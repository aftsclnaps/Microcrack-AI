"""INT8 quantization, evaluated individually. Outputs TorchScript (.ts) + sidecar .json (img_size).

  python -m src.optimization.quantize --ckpt artifacts/tiny_targeted/best.pt --mode dynamic
  python -m src.optimization.quantize --ckpt artifacts/tiny_targeted/best.pt --mode static

dynamic: quantizes nn.Linear only -> little effect on conv-heavy CNNs (expect ~no gain; we report it anyway).
static : post-training INT8 (FX graph mode, fbgemm/x86) calibrated on validation images. Uses torch.ao.quantization,
         which PyTorch is migrating to torchao; pin torch<3 as in requirements.txt. Accuracy MUST be re-measured.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

from src.data.dataset import make_loader
from src.models.distillation import load_checkpoint_model
from src.utils import load_config, log, save_json, setup_logging


def quantize_dynamic(model):
    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def quantize_static(model, calib_loader, n_batches=20):
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    torch.backends.quantized.engine = "fbgemm"
    ex = next(iter(calib_loader))[0][:1]
    prepared = prepare_fx(model.eval(), get_default_qconfig_mapping("fbgemm"), (ex,))
    with torch.no_grad():
        for i, (x, _, _) in enumerate(calib_loader):
            prepared(x)
            if i + 1 >= n_batches:
                break
    return convert_fx(prepared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml"); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--ckpt", required=True); ap.add_argument("--mode", choices=["dynamic", "static"], required=True)
    a = ap.parse_args(); setup_logging(); cfg = load_config(a.config, a.set)
    model, ck = load_checkpoint_model(a.ckpt, "cpu"); size = ck["img_size"]
    loader = make_loader(cfg, "val", "none", shuffle=False, img_size=size, batch_size=32)
    q = quantize_dynamic(model) if a.mode == "dynamic" else quantize_static(model, loader)
    ex = torch.randn(1, 3, size, size)
    ts = torch.jit.trace(q, ex)
    out = Path(a.ckpt).parent / f"int8_{a.mode}.ts"
    ts.save(str(out)); save_json({"img_size": size, "source": str(a.ckpt), "mode": a.mode}, out.with_suffix(".json"))
    log.info("Saved %s. Now run evaluate + benchmark on it.", out)


if __name__ == "__main__":
    main()
