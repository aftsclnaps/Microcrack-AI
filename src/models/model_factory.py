"""Build models by name. Returns (model, init_description)."""
from __future__ import annotations
from torch import nn
from torchvision import models as tvm

from src.models.tiny_cnn import MicroCrackTiny
from src.utils import log

MODEL_NAMES = ["tiny", "mobilenetv3_small", "shufflenetv2", "resnet18"]


def build_model(name: str, num_classes: int = 2, pretrained: bool = True) -> tuple[nn.Module, str]:
    init = "random"
    if name == "tiny":
        return MicroCrackTiny(num_classes), "random (trained from scratch)"
    builders = {
        "mobilenetv3_small": (tvm.mobilenet_v3_small, "MobileNet_V3_Small_Weights"),
        "shufflenetv2": (tvm.shufflenet_v2_x0_5, "ShuffleNet_V2_X0_5_Weights"),  # x0.5 = the lightweight variant
        "resnet18": (tvm.resnet18, "ResNet18_Weights"),
    }
    if name not in builders:
        raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")
    fn, wname = builders[name]
    weights = getattr(tvm, wname).DEFAULT if pretrained else None
    try:
        model = fn(weights=weights)
        init = "ImageNet-1k pretrained" if pretrained else "random"
    except Exception as e:  # offline / download failure: say so loudly, do not hide it
        log.warning("Could not load pretrained weights for %s (%s). Falling back to RANDOM init.", name, e)
        model, init = fn(weights=None), "random (pretrained download failed)"
    if name == "mobilenetv3_small":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model, init


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
