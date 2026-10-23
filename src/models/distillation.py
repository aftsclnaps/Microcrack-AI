"""Teacher loading for knowledge distillation."""
from __future__ import annotations
import torch
from src.models.model_factory import build_model


def load_checkpoint_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model, _ = build_model(ck["model_name"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


def load_teacher(path, device):
    teacher, ck = load_checkpoint_model(path, device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, ck
