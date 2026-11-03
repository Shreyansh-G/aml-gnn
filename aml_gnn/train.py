"""
train.py — train Multi-PNA, test the best epoch, write the artifacts.

Protocol (same as the published HI-Small benchmarks): temporal 64/19/17 split,
labels from `Is Laundering`, 2-hop neighbour sampling around each target edge,
weighted cross-entropy, model selection on validation F1 of the laundering class.

Training cost is bounded by `train_batches` (batches per epoch) and
`time_budget` (minutes); validation, test and scoring always cover every edge.

Also records what is needed to quantify the run: per-epoch history, GPU name,
peak GPU memory, training and scoring throughput.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from . import artifacts, config
from .data import TransactionGraph, load_graph
from .evaluate import compute_metrics, evaluate_split, predict
from .model import MultiPNA, model_args_for
from .sampler import GraphView
from .utils import get_logger

log = get_logger(__name__)

SPLIT_NAMES = ["Train", "Val", "Test"]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        # TF32 matmuls: large speed-up on A100-class GPUs, same model
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one_epoch(model: MultiPNA, view: GraphView, train_edges: torch.Tensor,
                    labels: torch.Tensor, optimizer, class_weights: torch.Tensor,
                    batch_size: int, max_batches: int = None):
    """One pass over (up to *max_batches* batches of) shuffled training edges → (mean loss, edges seen)."""
    model.train()
    total_loss, seen = 0.0, 0
    order = train_edges[torch.randperm(len(train_edges), device=train_edges.device)]
    for i, chunk in enumerate(order.split(batch_size)):
        if max_batches and i >= max_batches:
            break
        loss = F.cross_entropy(model(*view.batch(chunk)), labels[chunk], weight=class_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(chunk)
        seen += len(chunk)
    return total_loss / max(seen, 1), seen


@dataclass
class FitResult:
    history: list = field(default_factory=list)
    best_epoch: int = 0
    best_f1: float = -1.0
    best_state: dict = None
    stopped_by: str = "max_epochs"


def fit(model, views, targets, labels, epochs, patience, batch_size,
        train_batches=None, max_batches=None, time_budget=None) -> FitResult:
    """Train with early stopping on validation F1, within an optional time budget (minutes)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    class_weights = torch.tensor(config.CLASS_WEIGHTS, device=labels.device)
    train_cap = min([b for b in (train_batches, max_batches) if b], default=None)
    result, started = FitResult(), time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss, seen = train_one_epoch(model, views[0], targets[0], labels, optimizer,
                                     class_weights, batch_size, train_cap)
        t1 = time.time()
        val = evaluate_split(model, views[1], targets[1], labels, batch_size, max_batches)
        t2 = time.time()
        result.history.append({"epoch": epoch, "loss": loss, "train_edges": seen,
                               **{f"val_{k}": v for k, v in val.items()},
                               "train_seconds": round(t1 - t0, 1), "val_seconds": round(t2 - t1, 1)})
        log.info("Epoch %3d | loss %.4f | val P %.4f R %.4f F1 %.4f AUPRC %.4f | train %.0fs val %.0fs",
                 epoch, loss, val["precision"], val["recall"], val["f1"], val["auprc"], t1 - t0, t2 - t1)

        if val["f1"] > result.best_f1:
            result.best_f1, result.best_epoch = val["f1"], epoch
            result.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log.info("  ↑ new best (val F1 %.4f)", result.best_f1)
        elif epoch - result.best_epoch >= patience:
            log.info("Early stopping: no val F1 gain for %d epochs.", patience)
            result.stopped_by = "early_stopping"
            break

        elapsed = time.time() - started
        if time_budget and elapsed + elapsed / epoch > time_budget * 60:   # next epoch would overrun
            log.info("Time budget of %s min reached after %d epochs (%.1f min).",
                     time_budget, epoch, elapsed / 60)
            result.stopped_by = "time_budget"
            break
    return result


def train(epochs: int = config.EPOCHS, patience: int = config.PATIENCE,
          batch_size: int = config.BATCH_SIZE, neighbors: list = config.NUM_NEIGHBORS,
          train_batches: int = config.TRAIN_BATCHES, time_budget: float = config.TIME_BUDGET,
          seed: int = config.SEED, max_batches: int = None, score_all: bool = True) -> dict:
    """Full run: train, test the best epoch, save model, predictions, scores and metrics."""
    set_seed(seed)
    device = get_device()
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    log.info("=== Multi-PNA training === device=%s (%s) seed=%d | %s batches/epoch, budget %s min",
             device, gpu, seed, train_batches or "all", time_budget or "none")
    started = time.time()

    graph, spec = load_graph(device)
    views = [GraphView(graph, graph.split <= s, neighbors) for s in range(3)]
    targets = [graph.edges_in(s) for s in range(3)]
    for s, name in enumerate(SPLIT_NAMES):
        log.info("  %-5s: %8d edges, %5d laundering", name, len(targets[s]),
                 graph.y[targets[s]].sum().item())

    model_args = model_args_for(graph, targets[0])
    model = MultiPNA(**model_args).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %d", num_params)

    fitted = fit(model, views, targets, graph.y, epochs, patience, batch_size,
                 train_batches, max_batches, time_budget)
    history = fitted.history

    # --- Best epoch: predict val and test (each in its own split's graph) ---
    model.load_state_dict(fitted.best_state)
    probs = {s: predict(model, views[s], targets[s], batch_size, max_batches) for s in (1, 2)}
    test = compute_metrics(probs[2], graph.y[targets[2]])
    log.info("=" * 55)
    log.info("TEST (best epoch %d): P %.4f  R %.4f  F1 %.4f  AUPRC %.4f",
             fitted.best_epoch, test["precision"], test["recall"], test["f1"], test["auprc"])
    log.info("=" * 55)

    artifacts.save_checkpoint(fitted.best_state, model_args)
    artifacts.save_feature_spec(spec)        # lets the model score new datasets identically
    scored = {s: targets[s][:len(probs[s])].cpu().numpy() for s in (1, 2)}
    artifacts.save_predictions(
        np.concatenate([scored[1], scored[2]]),
        np.concatenate([np.full(len(scored[1]), 1), np.full(len(scored[2]), 2)]),
        torch.cat([probs[1], probs[2]]).cpu().numpy(),
    )

    train_seconds = sum(h["train_seconds"] for h in history)
    summary = {
        "model": "Multi-PNA (reverse MP + ports + ego IDs + edge updates)",
        "trained_on": config.DATASET_NAME,
        "dataset": config.DATASET_NAME,
        "parameters": num_params,
        "total_nodes": graph.num_nodes,
        "total_edges": graph.num_edges,
        "fraud_rate_pct": round(100 * graph.y.float().mean().item(), 4),
        "split_edges": [len(t) for t in targets],
        "seed": seed,
        "train_batches_per_epoch": train_batches or "all",
        "time_budget_minutes": time_budget or "none",
        "stopped_by": fitted.stopped_by,
        "epochs_trained": len(history),
        "best_epoch": fitted.best_epoch,
        "best_val_f1": fitted.best_f1,
        **{f"test_{k}": v for k, v in test.items()},
        "device": str(device),
        "gpu": gpu,
        "mean_epoch_train_seconds": round(train_seconds / len(history), 1),
        "mean_epoch_val_seconds": round(sum(h["val_seconds"] for h in history) / len(history), 1),
        "train_transactions_per_second": round(sum(h["train_edges"] for h in history) / max(train_seconds, 1e-9)),
    }

    if score_all:
        t0 = time.time()
        artifacts.save_scores(score_all_edges(model, views[2], graph, batch_size, max_batches))
        scored_count = min(graph.num_edges, (max_batches or 10**9) * batch_size)
        summary["scoring_transactions_per_second"] = round(scored_count / max(time.time() - t0, 1e-9))
        log.info("Saved %s", config.SCORES_PATH)

    if device.type == "cuda":
        summary["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    summary["train_minutes"] = round((time.time() - started) / 60, 1)
    summary["history"] = history
    artifacts.save_metrics(summary)
    log.info("Saved %s, %s and %s", config.MODEL_PATH, config.PREDICTIONS_PATH, config.METRICS_PATH)
    return summary


def score_all_edges(model: MultiPNA, view: GraphView, graph: TransactionGraph,
                    batch_size: int, max_batches: int = None) -> np.ndarray:
    """Laundering probability for every transaction (full graph); NaN where not scored."""
    probs = predict(model, view, torch.arange(graph.num_edges, device=graph.src.device),
                    batch_size, max_batches).cpu().numpy()
    scores = np.full(graph.num_edges, np.nan, dtype=np.float32)
    scores[:len(probs)] = probs
    return scores
