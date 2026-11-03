"""
cases.py — group high-risk transactions into investigation cases.

A case is a set of accounts connected by high-risk transactions (score ≥
threshold) inside a date range, typically one laundering ring. Investigators
review cases instead of single alerts; each case carries its detected typology
(typology.py) and can be turned into a draft STR (str_report.py).
"""

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple

import networkx as nx
import pandas as pd

from . import config
from .store import ScoreStore
from .typology import detect_typology


@dataclass
class Case:
    case_id: str
    edge_ids: List[int]
    accounts: List[int]
    max_score: float
    mean_score: float
    first_seen: pd.Timestamp
    last_seen: pd.Timestamp
    typology: str
    evidence: str
    known_laundering: int      # dataset labels — for evaluation, not used by the rules

    @property
    def hours(self) -> float:
        return (self.last_seen - self.first_seen).total_seconds() / 3600


def case_period(store: ScoreStore, start: Optional[date] = None,
                end: Optional[date] = None) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) covering whole days; defaults to the out-of-sample test period."""
    test_start, test_end = store.test_period
    start = pd.Timestamp(start) if start else test_start.normalize()
    end = pd.Timestamp(end) if end else test_end.normalize()
    return start, end + pd.Timedelta(days=1)


def find_cases(store: ScoreStore, start=None, end=None,
               min_score: float = config.HIGH_RISK) -> List[Case]:
    """Connected groups of high-risk transactions in [start, end), riskiest first."""
    flagged = store.scored_between(start, end, min_score)
    if flagged.empty:
        return []

    graph = nx.Graph()
    graph.add_edges_from(zip(flagged["src"].tolist(), flagged["dst"].tolist()))
    component = {acc: i for i, accs in enumerate(nx.connected_components(graph)) for acc in accs}

    labels = store.nodes["label"]
    cases = []
    for _, txs in flagged.groupby(flagged["src"].map(component)):
        typology, evidence = detect_typology(txs, labels)
        cases.append(Case(
            case_id=f"C{int(txs.index.min())}",
            edge_ids=[int(i) for i in txs.index],
            accounts=sorted({int(a) for a in pd.concat([txs["src"], txs["dst"]])}),
            max_score=float(txs["model_score"].max()),
            mean_score=float(txs["model_score"].mean()),
            first_seen=txs["Timestamp"].min(),
            last_seen=txs["Timestamp"].max(),
            typology=typology,
            evidence=evidence,
            known_laundering=int(txs["is_laundering"].sum()),
        ))
    return sorted(cases, key=lambda c: (-c.max_score, c.case_id))


def get_case(store: ScoreStore, case_id: str, start=None, end=None,
             min_score: float = config.HIGH_RISK) -> Optional[Case]:
    return next((c for c in find_cases(store, start, end, min_score) if c.case_id == case_id), None)


def summary(case: Case) -> dict:
    """One case in the public (API / table) format, without its transactions."""
    return {
        "case_id": case.case_id,
        "typology": case.typology,
        "evidence": case.evidence,
        "num_accounts": len(case.accounts),
        "num_transactions": len(case.edge_ids),
        "max_score": case.max_score,
        "mean_score": case.mean_score,
        "first_seen": case.first_seen,
        "last_seen": case.last_seen,
        "known_laundering": case.known_laundering,
    }
