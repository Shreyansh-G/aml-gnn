"""
score.py — run the trained model on a new transaction dataset (inference only).

  python -m aml_gnn score --input bank_transactions.csv --name mybank --columns mapping.json

Builds the workspace runs/<name>/:

  data/processed/graph/   the dataset as a transaction graph
  artifacts/              per-transaction scores, run metadata, results report
  web/                    a self-contained copy of the web app for this dataset

The model is not retrained. Features use the training-time encoding
(feature_spec.json) and alerts use the training-time threshold, so when the
new data has labels the report measures true out-of-dataset generalisation.
"""

import json
import re
import time
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import artifacts, config
from .data import load_graph
from .export_web import export_web
from .preprocess import build_graph
from .report import build_report
from .sampler import GraphView
from .train import get_device, score_all_edges
from .utils import get_logger

log = get_logger(__name__)

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def score_dataset(input_csv: Path, name: str, columns: Optional[dict] = None, patterns: Optional[Path] = None,
                  time_format: Optional[str] = None, start: Optional[date] = None, end: Optional[date] = None,
                  model_dir: Optional[Path] = None, batch_size: int = config.BATCH_SIZE,
                  neighbors: list = config.NUM_NEIGHBORS, max_batches: Optional[int] = None) -> dict:
    """Score every transaction of *input_csv* with the trained model and build its report and web app."""
    if not NAME_PATTERN.match(name):
        raise ValueError("--name may contain letters, digits, '.', '_' and '-' only")
    model_dir = Path(model_dir or config.MODEL_DIR)
    spec = artifacts.load_feature_spec(model_dir / "feature_spec.json")
    reference = json.loads((model_dir / "training_metrics.json").read_text())
    reference.pop("history", None)
    results_path = model_dir / "results.json"
    threshold = (json.loads(results_path.read_text())["test"]["at_tuned_threshold"]["threshold"]
                 if results_path.exists() else config.HIGH_RISK)

    config.use_workspace(config.RUNS_DIR / name, dataset_name=name)
    log.info("=== Scoring '%s' with the model trained on %s (no retraining) ===",
             name, reference.get("trained_on", "HI-Small"))
    meta = build_graph(Path(input_csv), patterns, columns, time_format, start, end, mode="score")

    device = get_device()
    graph, _ = load_graph(device, spec)
    model = artifacts.load_model(model_dir / "multipna_model.pt", device)
    view = GraphView(graph, torch.ones(graph.num_edges, dtype=torch.bool, device=device), neighbors)

    started = time.time()
    scores = score_all_edges(model, view, graph, batch_size, max_batches)
    seconds = time.time() - started
    scored = np.flatnonzero(~np.isnan(scores))
    artifacts.save_scores(scores)
    artifacts.save_predictions(scored, np.full(len(scored), 2), scores[scored])
    log.info("Scored %d transactions in %.1f s (%.0f per second)", len(scored), seconds, len(scored) / max(seconds, 1e-9))

    artifacts.save_metrics({
        **reference,                           # the model's own training facts and benchmark metrics
        "dataset": name,
        "source": str(input_csv),
        "has_labels": meta["has_labels"],
        "total_nodes": graph.num_nodes,
        "total_edges": graph.num_edges,
        "fraud_rate_pct": round(100 * graph.y.float().mean().item(), 4) if meta["has_labels"] else None,
        "reference_threshold": threshold,
        "scored_transactions": int(len(scored)),
        "scoring_seconds": round(seconds, 1),
        "scoring_transactions_per_second": round(len(scored) / max(seconds, 1e-9)),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    })

    results = build_report()
    export_web()
    log.info("Done. Report: %s | web app: %s (serve with: python -m http.server -d %s)",
             config.RESULTS_MD, config.WEB_DATA_DIR.parent, config.WEB_DATA_DIR.parent)
    return results
