"""Minimal Grad-CAM. A heatmap is an interpretability aid, NOT proof the prediction is correct."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def target_layer(model, name: str):
    return {"tiny": lambda m: m.features[-1], "resnet18": lambda m: m.layer4,
            "mobilenetv3_small": lambda m: m.features[-1], "shufflenetv2": lambda m: m.conv5}[name](model)


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts = model.eval(), None
        self._h = layer.register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def remove(self):
        self._h.remove()

    def __call__(self, x: torch.Tensor, class_idx: int = 1) -> np.ndarray:
        """x: (1,3,H,W). Returns HxW heatmap in [0,1] for `class_idx` (default: Crack)."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            logits[0, class_idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        return cam / cam.max() if cam.max() > 0 else cam
