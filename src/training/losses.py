"""Loss functions, including temperature-scaled knowledge distillation."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn


class DistillationLoss(nn.Module):
    """(1 - alpha) * CE(student, y) + alpha * T^2 * KL(softmax(teacher/T) || softmax(student/T))."""

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5, label_smoothing: float = 0.0):
        super().__init__()
        self.T, self.alpha = temperature, alpha
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, student_logits, labels, teacher_logits=None):
        hard = self.ce(student_logits, labels)
        if teacher_logits is None or self.alpha == 0:
            return hard
        soft = F.kl_div(F.log_softmax(student_logits / self.T, dim=1), F.softmax(teacher_logits / self.T, dim=1),
                        reduction="batchmean") * (self.T ** 2)
        return (1 - self.alpha) * hard + self.alpha * soft
