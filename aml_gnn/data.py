"""
data.py — load the processed graph as tensors with model-ready edge features.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from . import config
from .utils import get_logger

log = get_logger(__name__)


@dataclass
class TransactionGraph:
    """All transactions as tensors on one device; edge i = row i of edges_labeled.parquet."""
    num_nodes: int
    src: torch.Tensor    # sender account id                (E,)
    dst: torch.Tensor    # receiver account id              (E,)
    attr: torch.Tensor   # static edge features             (E, F)
    time: torch.Tensor   # seconds since first transaction  (E,)
    y: torch.Tensor      # 1 = laundering                   (E,)
    split: torch.Tensor  # 0 train, 1 val, 2 test           (E,)

    @property
    def num_edges(self) -> int:
        return self.src.numel()

    def edges_in(self, split: int) -> torch.Tensor:
        """Ids of the transactions in one split (0 train, 1 val, 2 test)."""
        return (self.split == split).nonzero().squeeze(1)


def fit_feature_spec(edges: pd.DataFrame) -> dict:
    """Feature encoding learned from the training split: category vocabularies and normalisation."""
    train = (edges["split"] == 0).to_numpy()
    cont = np.log1p(edges.loc[train, config.CONTINUOUS_FEATURES].to_numpy(np.float64))
    return {
        "continuous": config.CONTINUOUS_FEATURES,
        "mean": cont.mean(0).tolist(),
        "std": cont.std(0).tolist(),
        "categories": {col: [str(v) for v in edges[col].cat.categories]
                       for col in config.CATEGORICAL_FEATURES},
    }


def encode_edges(edges: pd.DataFrame, spec: dict) -> torch.Tensor:
    """
    Static edge features with a fixed encoding, so any dataset matches what the model saw:
    log-scaled z-scored continuous features, one-hot categories (unseen category → all zeros).
    """
    cont = np.log1p(edges[spec["continuous"]].to_numpy(np.float64))
    cont = (cont - np.array(spec["mean"])) / (np.array(spec["std"]) + 1e-8)
    feats = [cont.astype(np.float32)]
    for col, vocab in spec["categories"].items():
        index = {value: i for i, value in enumerate(vocab)}
        codes = edges[col].astype(str).map(index).fillna(-1).astype(int).to_numpy()
        one_hot = np.zeros((len(edges), len(vocab)), dtype=np.float32)
        known = np.flatnonzero(codes >= 0)
        one_hot[known, codes[known]] = 1.0
        unseen = len(edges) - len(known)
        if unseen:
            log.info("  %s: %d transactions with a category unseen in training (encoded as all zeros)", col, unseen)
        feats.append(one_hot)
    return torch.from_numpy(np.concatenate(feats, axis=1))


def load_graph(device: torch.device, spec: dict = None):
    """
    Load the active workspace's graph as tensors on *device* → (graph, spec).

    Training passes no spec (it is fitted on the training split); scoring a new
    dataset passes the trained model's spec so features are encoded identically.
    """
    log.info("Loading %s ...", config.EDGES_PATH)
    edges = pd.read_parquet(config.EDGES_PATH)
    num_nodes = len(pd.read_parquet(config.NODES_PATH, columns=["node_id"]))
    spec = spec or fit_feature_spec(edges)
    seconds = (edges["Timestamp"] - edges["Timestamp"].min()).dt.total_seconds()

    def tensor(values, dtype):
        return torch.tensor(values, dtype=dtype, device=device)

    graph = TransactionGraph(
        num_nodes=num_nodes,
        src=tensor(edges["src"].to_numpy(), torch.long),
        dst=tensor(edges["dst"].to_numpy(), torch.long),
        attr=encode_edges(edges, spec).to(device),
        time=tensor(seconds.to_numpy(), torch.float32),
        y=tensor(edges["is_laundering"].to_numpy(), torch.long),
        split=tensor(edges["split"].to_numpy(), torch.long),
    )
    log.info("  → %d accounts, %d transactions, %d edge features (+1 time).",
             graph.num_nodes, graph.num_edges, graph.attr.size(1))
    return graph, spec
