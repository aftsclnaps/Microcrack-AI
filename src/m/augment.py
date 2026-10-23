"""Training-time augmentation. Applied to the TRAIN split only.

Modes
  none      resize + normalize
  ordinary  flips + small rotation/translation
  targeted  ordinary + brightness/contrast, mild gamma, soft shadow, mild noise, mild blur
Parameters are deliberately mild: contrast is never reduced below 0.75x and noise stays small so that
hairline cracks are not erased (which would corrupt the label).
"""
from __future__ import annotations
import math
import random
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as TF

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class RandomGamma:
    def __init__(self, low=0.8, high=1.25, p=0.4):
        self.low, self.high, self.p = low, high, p

    def __call__(self, img):
        return TF.adjust_gamma(img, random.uniform(self.low, self.high)) if random.random() < self.p else img


class RandomShadow:
    """Darken one side of the image with a soft-edged ramp (simulates a cast shadow)."""

    def __init__(self, strength=(0.15, 0.45), p=0.3):
        self.strength, self.p = strength, p

    def __call__(self, x: torch.Tensor):
        if random.random() >= self.p:
            return x
        _, h, w = x.shape
        theta = random.uniform(0, 2 * math.pi)
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        proj = xs * math.cos(theta) + ys * math.sin(theta)
        mask = torch.sigmoid((proj - random.uniform(-0.4, 0.4)) * random.uniform(4, 10))
        factor = 1.0 - random.uniform(*self.strength) * mask
        return (x * factor.unsqueeze(0)).clamp(0, 1)


class RandomNoise:
    def __init__(self, sigma=(0.005, 0.03), p=0.3):
        self.sigma, self.p = sigma, p

    def __call__(self, x):
        if random.random() >= self.p:
            return x
        return (x + torch.randn_like(x) * random.uniform(*self.sigma)).clamp(0, 1)


def build_transforms(mode: str, img_size: int, train: bool):
    resize = [T.Resize((img_size, img_size))]
    norm = [T.Normalize(MEAN, STD)]
    if not train or mode == "none":
        return T.Compose(resize + [T.ToTensor()] + norm)
    pil_ops = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
               T.RandomAffine(degrees=10, translate=(0.05, 0.05), fill=0)]
    tensor_ops = []
    if mode == "targeted":
        pil_ops += [T.RandomApply([T.ColorJitter(brightness=0.3, contrast=(0.75, 1.3))], p=0.6),
                    RandomGamma(),
                    T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 0.8))], p=0.2)]
        tensor_ops = [RandomShadow(), RandomNoise()]
    elif mode != "ordinary":
        raise ValueError(f"Unknown augmentation mode '{mode}' (use none | ordinary | targeted)")
    return T.Compose(resize + pil_ops + [T.ToTensor()] + tensor_ops + norm)
