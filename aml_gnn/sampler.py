"""
sampler.py — 2-hop neighbour sampling around target transactions.

Plain PyTorch (runs on the GPU), so no pyg-lib / torch-sparse builds are needed.
"""

import torch

from .data import TransactionGraph


class GraphView:
    """
    The transactions visible at one stage of the temporal split.

      train view : train edges            → scores train edges
      val view   : train + val edges      → scores val edges
      test view  : all edges              → scores test edges
    """

    def __init__(self, graph: TransactionGraph, visible: torch.Tensor, num_neighbors: list):
        self.graph = graph
        self.num_neighbors = num_neighbors
        eids = visible.nonzero().squeeze(1)

        # Time feature, z-scored over this view's edges (as in Multi-GNN)
        t = graph.time[eids]
        self.time = (graph.time - t.mean()) / (t.std() + 1e-8)

        # CSR index: for every account, its incident edges (sent and received)
        ends = torch.cat([graph.src[eids], graph.dst[eids]])
        order = torch.argsort(ends)
        self.incident = torch.cat([eids, eids])[order]
        self.ptr = torch.zeros(graph.num_nodes + 1, dtype=torch.long, device=eids.device)
        self.ptr[1:] = torch.bincount(ends, minlength=graph.num_nodes).cumsum(0)

    def _sample(self, nodes: torch.Tensor, k: int) -> torch.Tensor:
        """Up to k incident edges per node: all of them if degree <= k, else k at random."""
        start = self.ptr[nodes]
        deg = self.ptr[nodes + 1] - start
        take = deg.clamp(max=k)
        owner = torch.repeat_interleave(torch.arange(len(nodes), device=nodes.device), take)
        offset = torch.arange(len(owner), device=nodes.device) - (take.cumsum(0) - take)[owner]
        deg = deg[owner]
        random = (torch.rand(len(owner), device=nodes.device) * deg).long()
        return self.incident[start[owner] + torch.where(deg <= k, offset, random)]

    def batch(self, target: torch.Tensor):
        """Sampled subgraph around *target* edges → (x, edge_index, edge_attr, target_pos)."""
        g = self.graph
        seeds = torch.cat([g.src[target], g.dst[target]])
        eids, frontier = target, torch.unique(seeds)
        seen = frontier
        for k in self.num_neighbors:
            sampled = self._sample(frontier, k)
            eids = torch.cat([eids, sampled])
            reached = torch.unique(torch.cat([g.src[sampled], g.dst[sampled]]))
            frontier = reached[~torch.isin(reached, seen)]
            seen = torch.cat([seen, frontier])

        eids = torch.unique(eids)                       # sorted
        nodes = torch.unique(torch.cat([g.src[eids], g.dst[eids]]))
        edge_index = torch.stack([torch.searchsorted(nodes, g.src[eids]),
                                  torch.searchsorted(nodes, g.dst[eids])])

        ego = torch.isin(nodes, seeds).float()          # ego IDs
        x = torch.stack([torch.ones_like(ego), ego], dim=1)
        edge_attr = torch.cat([g.attr[eids], self.time[eids].unsqueeze(1)], dim=1)
        return x, edge_index, edge_attr, torch.searchsorted(eids, target)
