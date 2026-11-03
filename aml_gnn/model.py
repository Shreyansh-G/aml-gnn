"""
model.py — Multi-PNA edge classifier.

PNA message passing with the "Multi-GNN" adaptations of Egressy et al.,
AAAI 2024 (reference code: github.com/IBM/Multi-GNN, Apache-2.0):

  - reverse message passing : separate convolutions for incoming and outgoing
                              transactions, averaged
  - port numbers            : in/out port of each edge as edge features
  - ego IDs                 : a flag on the two accounts of each scored edge
  - edge updates            : an MLP refreshes every edge embedding per layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv
from torch_geometric.utils import degree

from . import config
from .data import TransactionGraph

NODE_DIM = 2   # constant 1 + ego-ID flag


class MultiPNA(nn.Module):
    """PNA with reverse message passing, edge updates and an edge-level MLP head."""

    def __init__(self, edge_dim: int, deg_in: torch.Tensor, deg_out: torch.Tensor,
                 node_dim: int = NODE_DIM, hidden: int = config.HIDDEN,
                 num_layers: int = config.NUM_LAYERS, dropout: float = config.DROPOUT,
                 final_dropout: float = config.FINAL_DROPOUT):
        super().__init__()
        self.dropout = dropout
        self.node_emb = nn.Linear(node_dim, hidden)
        self.edge_emb = nn.Linear(edge_dim, hidden)

        def pna(deg):
            return PNAConv(hidden, hidden, aggregators=["mean", "min", "max", "std"],
                           scalers=["identity", "amplification", "attenuation"],
                           deg=deg, edge_dim=hidden, towers=5,
                           pre_layers=1, post_layers=1, divide_input=False)

        self.conv_in  = nn.ModuleList(pna(deg_in) for _ in range(num_layers))   # sender → receiver
        self.conv_out = nn.ModuleList(pna(deg_out) for _ in range(num_layers))  # receiver → sender
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(num_layers))
        self.edge_mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            for _ in range(num_layers)
        )
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, 50), nn.ReLU(), nn.Dropout(final_dropout),
            nn.Linear(50, 25), nn.ReLU(), nn.Dropout(final_dropout),
            nn.Linear(25, 2),
        )

    def forward(self, x, edge_index, edge_attr, target):
        src, dst = edge_index
        h = self.node_emb(x)
        e = self.edge_emb(edge_attr)
        for conv_in, conv_out, norm, edge_mlp in zip(self.conv_in, self.conv_out,
                                                     self.norms, self.edge_mlps):
            msg = (conv_in(h, edge_index, e) + conv_out(h, edge_index.flip(0), e)) / 2
            h = (h + F.relu(norm(msg))) / 2
            h = F.dropout(h, self.dropout, self.training)
            e = e + edge_mlp(torch.cat([h[src], h[dst], e], dim=-1)) / 2

        s, d = src[target], dst[target]
        return self.head(torch.cat([h[s], h[d], e[target]], dim=-1))


def model_args_for(graph: TransactionGraph, train_edges: torch.Tensor) -> dict:
    """Constructor arguments: edge feature size and PNA degree histograms of the training graph."""
    deg_in = torch.bincount(degree(graph.dst[train_edges], graph.num_nodes).long()).cpu()
    deg_out = torch.bincount(degree(graph.src[train_edges], graph.num_nodes).long()).cpu()
    return {"edge_dim": graph.attr.size(1) + 1, "deg_in": deg_in, "deg_out": deg_out}
