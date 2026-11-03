"""
evaluate.py — batched prediction and laundering-class metrics.
"""

import torch
from torchmetrics.functional.classification import (
    binary_average_precision,
    binary_f1_score,
    binary_precision,
    binary_recall,
)

from .model import MultiPNA
from .sampler import GraphView


@torch.no_grad()
def predict(model: MultiPNA, view: GraphView, targets: torch.Tensor,
            batch_size: int, max_batches: int = None) -> torch.Tensor:
    """Laundering probability for each target edge, in order."""
    model.eval()
    probs = []
    for i, chunk in enumerate(targets.split(batch_size)):
        if max_batches and i >= max_batches:
            break
        probs.append(model(*view.batch(chunk)).softmax(-1)[:, 1])
    return torch.cat(probs)


def compute_metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict:
    """Precision, recall and F1 at 0.5, plus AUPRC, for the laundering class."""
    labels = labels[:len(probs)]
    return {
        "precision": binary_precision(probs, labels).item(),
        "recall":    binary_recall(probs, labels).item(),
        "f1":        binary_f1_score(probs, labels).item(),
        "auprc":     binary_average_precision(probs, labels).item(),
    }


def evaluate_split(model: MultiPNA, view: GraphView, targets: torch.Tensor,
                   labels: torch.Tensor, batch_size: int, max_batches: int = None) -> dict:
    """Metrics for one split: predict every target edge, compare with its label."""
    return compute_metrics(predict(model, view, targets, batch_size, max_batches), labels[targets])
