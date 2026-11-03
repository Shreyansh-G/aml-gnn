"""
store.py — read-only query layer over scored transactions.

The single home of the investigation logic (transaction lookup, account risk,
neighbourhoods, alert queue), shared by the REST API (api/) and the web export
(export_web.py). Pure pandas — no PyTorch — so the serving image stays small.
"""

import json
from datetime import date
from functools import cached_property
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from . import config

def missing_files() -> list:
    """Files the store needs, in the active workspace, that do not exist yet."""
    required = [config.NODES_PATH, config.EDGES_PATH, config.SCORES_PATH, config.METRICS_PATH]
    return [p for p in required if not p.exists()]


def risk_level(scores) -> np.ndarray:
    """HIGH / MEDIUM / LOW from the model's laundering probability."""
    scores = np.asarray(scores)
    return np.where(scores >= config.HIGH_RISK, "HIGH",
                    np.where(scores >= config.MEDIUM_RISK, "MEDIUM", "LOW"))


class ScoreStore:
    """
    Every transaction with its model score, plus the account table.

    edges : one row per transaction, index = edge_id, with a `model_score` column
    nodes : one row per account, index = node_id, with a `label` column

    Two indexes are built once at startup so queries never scan all edges:
      - CSR adjacency (account → its transactions): account queries cost O(degree)
      - edges sorted by time: a one-day alert query is a binary search, O(log E)
    """

    def __init__(self, edges: pd.DataFrame, nodes: pd.DataFrame, metrics: dict):
        self.edges = edges
        self.nodes = nodes
        self.metrics = metrics
        self._src = edges["src"].to_numpy()
        self._dst = edges["dst"].to_numpy()

        # CSR adjacency: incident edge ids of node v are _incident[_ptr[v]:_ptr[v + 1]]
        ends = np.concatenate([self._src, self._dst])
        edge_ids = np.arange(len(edges))
        self._incident = np.concatenate([edge_ids, edge_ids])[np.argsort(ends, kind="stable")]
        self._ptr = np.zeros(len(nodes) + 1, dtype=np.int64)
        np.cumsum(np.bincount(ends, minlength=len(nodes)), out=self._ptr[1:])

        # Time index for the alert queue
        times = edges["Timestamp"].to_numpy()
        self._by_time = np.argsort(times, kind="stable")
        self._sorted_time = pd.DatetimeIndex(times[self._by_time])

    @classmethod
    def from_artifacts(cls) -> "ScoreStore":
        missing = missing_files()
        if missing:
            raise FileNotFoundError("Missing: " + ", ".join(str(p) for p in missing)
                                    + ". Run `python -m aml_gnn prepare` and `python -m aml_gnn train`.")
        edges = pd.read_parquet(config.EDGES_PATH)
        edges["model_score"] = pd.read_parquet(config.SCORES_PATH)["score"].fillna(0.0).to_numpy()
        nodes = pd.read_parquet(config.NODES_PATH)
        nodes["label"] = nodes["bank"] + " | " + nodes["account"]
        return cls(edges, nodes, json.loads(config.METRICS_PATH.read_text()))

    @property
    def num_transactions(self) -> int:
        return len(self.edges)

    # --- Lookups -----------------------------------------------------------

    def transaction(self, edge_id: int) -> Optional[dict]:
        if not 0 <= edge_id < len(self.edges):
            return None
        return self.to_records(self.edges.iloc[[edge_id]])[0]

    def find_accounts(self, account: str) -> pd.DataFrame:
        """All (bank, account) nodes with this account number."""
        return self.nodes[self.nodes["account"] == account.strip()]

    def has_account(self, node_id: int) -> bool:
        return 0 <= node_id < len(self.nodes)

    def incident_edges(self, nodes) -> np.ndarray:
        """Sorted ids of every transaction sent or received by any of *nodes* — O(total degree)."""
        nodes = np.fromiter(nodes, dtype=np.int64)
        starts, stops = self._ptr[nodes], self._ptr[nodes + 1]
        lengths = stops - starts
        # Concatenate the CSR slices of all nodes without a Python loop
        offsets = np.repeat(starts - (np.cumsum(lengths) - lengths), lengths)
        return np.unique(self._incident[offsets + np.arange(lengths.sum())])

    def account_risk(self, node_id: int) -> Optional[dict]:
        """Risk summary of one account from the scores of all its transactions."""
        if not self.has_account(node_id):
            return None
        ids = self.incident_edges([node_id])
        scores = self.edges["model_score"].to_numpy()[ids]
        labels = self.edges["is_laundering"].to_numpy()[ids]
        max_score = float(scores.max()) if len(ids) else 0.0
        node = self.nodes.iloc[node_id]
        return {
            "node_id": node_id,
            "bank": node["bank"],
            "account": node["account"],
            "transactions_sent": int((self._src[ids] == node_id).sum()),
            "transactions_received": int((self._dst[ids] == node_id).sum()),
            "max_score": max_score,
            "mean_score": float(scores.mean()) if len(scores) else 0.0,
            "high_risk_transactions": int((scores >= config.HIGH_RISK).sum()),
            "known_laundering": int(labels.sum()),
            "risk": str(risk_level([max_score])[0]),
        }

    # --- Investigation -----------------------------------------------------

    def neighborhood(self, node_id: int, hops: int = 1, max_edges: int = 2000,
                     max_nodes: int = 5000) -> Tuple[pd.DataFrame, set]:
        """k-hop neighbourhood of an account → (transactions among those accounts, account ids)."""
        node_set, frontier = {node_id}, {node_id}
        for _ in range(hops):
            if not frontier:
                break
            ids = self.incident_edges(frontier)
            reached = set(np.unique(np.concatenate([self._src[ids], self._dst[ids]])).tolist())
            frontier = reached - node_set
            node_set |= reached
            if len(node_set) > max_nodes:
                break

        # Transactions with both accounts inside the neighbourhood
        members = np.fromiter(node_set, dtype=np.int64)
        candidates = self.incident_edges(node_set)
        inside = np.isin(self._src[candidates], members) & np.isin(self._dst[candidates], members)
        sub = self.edges.iloc[candidates[inside]]
        if len(sub) > max_edges:
            # Keep the riskiest transactions rather than a random sample
            sub = sub.nlargest(max_edges, "model_score")
            node_set = set(sub["src"].tolist()) | set(sub["dst"].tolist()) | {node_id}
        return sub, node_set

    @cached_property
    def test_period(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """First and last timestamp of the test split, where scores are out-of-sample."""
        times = self.edges.loc[self.edges["split"] == 2, "Timestamp"]
        return times.min(), times.max()

    def scored_between(self, start=None, end=None, min_score: float = 0.0) -> pd.DataFrame:
        """Transactions in [start, end) scoring ≥ min_score, found by binary search on time."""
        edges = self.edges
        if start is not None or end is not None:
            lo = 0 if start is None else self._sorted_time.searchsorted(pd.Timestamp(start))
            hi = len(edges) if end is None else self._sorted_time.searchsorted(pd.Timestamp(end))
            edges = edges.iloc[np.sort(self._by_time[lo:hi])]
        return edges[edges["model_score"] >= min_score]

    def alerts(self, limit: int = 100, day: Optional[date] = None,
               min_score: float = config.HIGH_RISK) -> pd.DataFrame:
        """Alert queue: the highest-scoring transactions, optionally for one day."""
        if day is None:
            flagged = self.scored_between(min_score=min_score)
        else:
            start = pd.Timestamp(day)
            flagged = self.scored_between(start, start + pd.Timedelta(days=1), min_score)
        return flagged.nlargest(limit, "model_score")

    # --- Output ------------------------------------------------------------

    def to_records(self, edges: pd.DataFrame) -> list:
        """Transactions in the public (API / table) format."""
        label = self.nodes["label"]
        out = pd.DataFrame({
            "edge_id":            edges.index.astype(int),
            "timestamp":          edges["Timestamp"].to_numpy(),
            "sender_id":          edges["src"].to_numpy(),
            "sender":             edges["src"].map(label).to_numpy(),
            "receiver_id":        edges["dst"].to_numpy(),
            "receiver":           edges["dst"].map(label).to_numpy(),
            "amount_paid":        edges["Amount Paid"].to_numpy(),
            "payment_currency":   edges["Payment Currency"].astype(str).to_numpy(),
            "amount_received":    edges["Amount Received"].to_numpy(),
            "receiving_currency": edges["Receiving Currency"].astype(str).to_numpy(),
            "payment_format":     edges["Payment Format"].astype(str).to_numpy(),
            "score":              edges["model_score"].astype(float).to_numpy(),
            "risk":               risk_level(edges["model_score"]),
            "is_laundering":      edges["is_laundering"].astype(int).to_numpy(),
            "pattern_type":       edges["pattern_type"].astype(str).to_numpy(),
        })
        return out.to_dict("records")
