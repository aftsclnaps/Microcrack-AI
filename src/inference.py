"""Single-image inference used by the Streamlit app (CPU)."""
from __future__ import annotations
import json, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from src.data.augment import build_transforms
from src.evaluation.benchmark import model_size_mb
from src.evaluation.evaluate import load_for_eval
from src.explainability.gradcam import GradCAM, target_layer


@dataclass
class Prediction:
    prob_crack: float
    label: str
    confidence: float
    needs_review: bool
    preprocess_ms: float
    forward_ms: float
    heatmap: np.ndarray | None = None


class ModelBundle:
    def __init__(self, ckpt_path: str | Path):
        self.path = Path(ckpt_path)
        self.model, self.img_size, self.meta = load_for_eval(self.path, "cpu")
        self.is_ts = self.path.suffix == ".ts"
        self.name = self.meta.get("model_name", self.path.parent.name) if not self.is_ts else f"{self.path.parent.name} ({self.path.stem})"
        self.tf = build_transforms("none", self.img_size, train=False)
        calib = sorted(self.path.parent.glob(f"calibration_{self.path.stem}.json"))
        c = json.loads(calib[0].read_text()) if calib else {}
        self.threshold = c.get("threshold", 0.5)
        self.review_tau = c.get("review", {}).get("tau")
        self.params = int(sum(p.numel() for p in self.model.parameters()))
        self.size_mb = model_size_mb(self.model, self.path)
        bench = self.path.parent / f"benchmark_{self.path.stem}.json"
        self.benchmark = json.loads(bench.read_text()) if bench.exists() else None
        ev = self.path.parent / f"eval_{self.path.stem}_test.json"
        self.test_eval = json.loads(ev.read_text()) if ev.exists() else None

    def predict(self, img: Image.Image, heatmap: bool = False) -> Prediction:
        img = img.convert("RGB")
        t0 = time.perf_counter(); x = self.tf(img).unsqueeze(0); t1 = time.perf_counter()
        with torch.inference_mode():
            p = torch.softmax(self.model(x), 1)[0, 1].item()
        t2 = time.perf_counter()
        label = "Crack" if p >= self.threshold else "No Crack"
        conf = p if label == "Crack" else 1 - p
        review = conf < (self.review_tau if self.review_tau is not None else 0.65)  # 0.65 = uncalibrated default
        hm = None
        if heatmap and not self.is_ts:
            cam = GradCAM(self.model, target_layer(self.model, self.meta["model_name"]))
            try:
                hm = cam(x, 1)
            finally:
                cam.remove()
        return Prediction(p, label, conf, bool(review), (t1 - t0) * 1e3, (t2 - t1) * 1e3, hm)


def find_models(artifacts_dir="artifacts"):
    paths = sorted(Path(artifacts_dir).glob("*/best.pt")) + sorted(Path(artifacts_dir).glob("*/int8_*.ts"))
    return {f"{p.parent.name}/{p.name}": p for p in paths}
