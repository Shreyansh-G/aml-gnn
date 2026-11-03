"""
artifacts.py — save and load everything training and scoring produce.

  multipna_model.pt      best model weights + constructor arguments
  feature_spec.json      training-time feature encoding (vocabularies, normalisation)
  training_metrics.json  run metadata and metrics
  edge_scores.parquet    laundering probability for every transaction
  predictions.parquet    scored val/test probabilities, for the report

Paths default to the active workspace (config.use_workspace), resolved at call time.
"""

import json

import numpy as np
import pandas as pd
import torch

from . import config
from .model import MultiPNA


def _target(path, default):
    path = path or default
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(state_dict: dict, model_args: dict, path=None) -> None:
    torch.save({"state_dict": state_dict, "model_args": model_args}, _target(path, config.MODEL_PATH))


def load_model(path=None, device: str = "cpu") -> MultiPNA:
    """Rebuild a trained model from its checkpoint, ready for inference."""
    ckpt = torch.load(path or config.MODEL_PATH, map_location=device, weights_only=True)
    model = MultiPNA(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def save_feature_spec(spec: dict, path=None) -> None:
    _target(path, config.FEATURE_SPEC_PATH).write_text(json.dumps(spec, indent=2))


def load_feature_spec(path=None) -> dict:
    return json.loads((path or config.FEATURE_SPEC_PATH).read_text())


def save_metrics(summary: dict, path=None) -> None:
    _target(path, config.METRICS_PATH).write_text(json.dumps(summary, indent=2))


def save_scores(scores: np.ndarray, path=None) -> None:
    pd.DataFrame({"score": scores.astype(np.float32)}).to_parquet(_target(path, config.SCORES_PATH), index=False)


def save_predictions(edge_ids: np.ndarray, split: np.ndarray, probs: np.ndarray, path=None) -> None:
    """Probabilities of the evaluated edges (val/test), each scored in its own split's graph."""
    pd.DataFrame({"edge_id": edge_ids.astype(np.int64), "split": split.astype(np.int8),
                  "prob": probs.astype(np.float32)}).to_parquet(_target(path, config.PREDICTIONS_PATH), index=False)
