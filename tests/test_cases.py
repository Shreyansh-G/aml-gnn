import pandas as pd
import pytest

from aml_gnn.cases import find_cases, get_case
from aml_gnn.store import ScoreStore
from aml_gnn.str_report import draft_str
from aml_gnn.typology import detect_typology


def edges(*pairs):
    return pd.DataFrame(pairs, columns=["src", "dst"])


@pytest.mark.parametrize("pairs, expected", [
    ([(1, 2), (2, 3), (3, 1)], "CYCLE"),
    ([(1, 2), (1, 3), (1, 4)], "FAN-OUT"),
    ([(2, 1), (3, 1), (4, 1)], "FAN-IN"),
    ([(1, 3), (1, 4), (2, 3), (2, 4)], "BIPARTITE"),
    ([(1, 2), (1, 3), (1, 4), (2, 5), (3, 5), (4, 5)], "SCATTER-GATHER"),
    ([(1, 3), (2, 3), (3, 4), (3, 5)], "GATHER-SCATTER"),
    ([(1, 3), (1, 4), (2, 3), (2, 4), (3, 5), (3, 6), (4, 5), (4, 6)], "STACK"),
    ([(1, 2), (2, 3), (3, 4)], "RANDOM"),
    ([(1, 2)], "PAIR"),
    ([(1, 1)], "SELF-TRANSFER"),
])
def test_detect_typology(pairs, expected):
    typology, evidence = detect_typology(edges(*pairs))
    assert typology == expected and evidence


def case_store() -> ScoreStore:
    """Two rings: a fan-out (0 → 1, 2, 3) and a cycle (4 → 5 → 6 → 4), plus low-risk noise."""
    pairs = [(0, 1), (0, 2), (0, 3), (4, 5), (5, 6), (6, 4), (7, 8)]
    scores = [0.9, 0.8, 0.7, 0.95, 0.6, 0.55, 0.05]
    nodes = pd.DataFrame({"node_id": range(9), "bank": ["001"] * 9, "account": [f"A{i}" for i in range(9)]})
    nodes["label"] = nodes["bank"] + " | " + nodes["account"]
    edges_df = pd.DataFrame({
        "src": [p[0] for p in pairs], "dst": [p[1] for p in pairs],
        "Timestamp": pd.date_range("2022-09-10 08:00", periods=7, freq="h"),
        "Amount Received": [100.0] * 7, "Amount Paid": [100.0] * 7,
        "Receiving Currency": pd.Categorical(["Euro"] * 7), "Payment Currency": pd.Categorical(["Euro"] * 7),
        "Payment Format": pd.Categorical(["ACH"] * 7),
        "split": [2] * 7,
        "is_laundering": [1, 1, 1, 1, 1, 1, 0],
        "pattern_type": pd.Categorical(["FAN-OUT"] * 3 + ["CYCLE"] * 3 + ["NONE"]),
        "model_score": scores,
    })
    return ScoreStore(edges_df, nodes, {"model": "test", "test_f1": 0.5, "test_precision": 0.5, "test_recall": 0.5})


def test_find_cases_groups_rings_and_ranks_by_risk():
    store = case_store()
    cases = find_cases(store, min_score=0.5)
    assert [(c.typology, len(c.edge_ids), c.known_laundering) for c in cases] == [
        ("CYCLE", 3, 3), ("FAN-OUT", 3, 3)]                 # riskiest first; noise below threshold
    assert cases[0].case_id == "C3" and cases[0].accounts == [4, 5, 6]
    assert get_case(store, "C0", min_score=0.5).typology == "FAN-OUT"
    assert get_case(store, "C99", min_score=0.5) is None


def test_account_network_has_hops_and_respects_caps(monkeypatch):
    import aml_gnn.export_web as export_web
    store = case_store()
    window = (pd.Timestamp("2022-09-10"), pd.Timestamp("2022-09-11"))
    pairs = lambda g, hop: {(e[0], e[1]) for e in g["edges"] if e[8] == hop}   # noqa: E731

    g = export_web.account_graph_builder(store, *window)(5)                     # cycle 4 → 5 → 6 → 4
    assert g["seed"] == 5 and set(g["nodes"]) == {4, 5, 6}
    assert pairs(g, 1) == {(4, 5), (5, 6)} and pairs(g, 2) == {(6, 4)}           # 2nd hop never touches the seed

    monkeypatch.setattr(export_web, "GRAPH_HOP1_MAX", 1)                        # keep only the riskiest direct edge
    g = export_web.account_graph_builder(store, *window)(5)
    assert pairs(g, 1) == {(4, 5)} and pairs(g, 2) == {(6, 4)}


def test_draft_str_has_every_section():
    store = case_store()
    text = draft_str(get_case(store, "C0", min_score=0.5), store)
    for heading in ["Summary", "Subject accounts", "Grounds for suspicion", "Transactions",
                    "Analyst checklist", "Model information"]:
        assert heading in text
    assert "FAN-OUT" in text and "001 | A0" in text and "300.00 Euro" in text
