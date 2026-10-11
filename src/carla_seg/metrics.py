"""Explicit multiclass Dice and dataset-level confusion-matrix metrics."""

import torch
from torch.nn import functional as F


def multiclass_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match SMP's multiclass Dice defaults with all output classes enabled.

    Pool over batch and pixels. An absent target class contributes zero loss;
    the final mean still divides by the number of output classes, as in SMP.
    """
    classes = logits.shape[1]
    probability = logits.float().log_softmax(dim=1).exp().flatten(2)
    truth = F.one_hot(target.long(), classes).permute(0, 3, 1, 2).flatten(2).float()
    intersection = (probability * truth).sum(dim=(0, 2))
    denominator = probability.sum(dim=(0, 2)) + truth.sum(dim=(0, 2))
    dice = 2 * intersection / denominator.clamp_min(1e-7)
    present = truth.sum(dim=(0, 2)) > 0
    return ((1 - dice) * present).mean()


def hybrid_loss(logits, target, ce_weight=0.5):
    return ce_weight * F.cross_entropy(logits, target) + (1 - ce_weight) * multiclass_dice_loss(logits, target)


class SegmentationMetrics:
    """Accumulate on CPU, also supporting devices without int64 bincount."""
    def __init__(self, num_classes=13):
        self.num_classes = num_classes
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, predictions: torch.Tensor, target: torch.Tensor):
        if predictions.ndim == target.ndim + 1:
            predictions = predictions.argmax(dim=1)
        if predictions.shape != target.shape:
            raise ValueError("Prediction and target shapes must match")
        predictions = predictions.detach().to(device="cpu", dtype=torch.int64).flatten()
        target = target.detach().to(device="cpu", dtype=torch.int64).flatten()
        if ((predictions < 0) | (predictions >= self.num_classes) |
                (target < 0) | (target >= self.num_classes)).any():
            raise ValueError("Metric received an out-of-range class ID")
        bins = torch.bincount(target * self.num_classes + predictions,
                              minlength=self.num_classes ** 2)
        self.confusion += bins.reshape(self.num_classes, self.num_classes)

    def compute(self):
        matrix = self.confusion.double()
        intersection = matrix.diag()
        union = matrix.sum(0) + matrix.sum(1) - intersection
        present = union > 0
        if not present.any():
            raise ValueError("Cannot compute metrics before observing pixels")
        iou = intersection / union.clamp_min(1)
        mean = iou[present].mean().item()
        return {
            "miou": mean,
            "miou_percent": 100 * mean,
            "pixel_accuracy": (intersection.sum() / matrix.sum()).item(),
            "per_class_iou": [value.item() if exists else None for value, exists in zip(iou, present)],
            "classes_in_macro_average": present.nonzero().flatten().tolist(),
            "class_support_pixels": self.confusion.sum(1).tolist(),
            "confusion_matrix": self.confusion.tolist(),
            "metric_policy": "Global confusion matrix; macro IoU over classes with nonzero union, including class 0; 255 remapped to 0.",
        }
