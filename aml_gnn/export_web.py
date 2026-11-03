"""
export_web.py — precompute the static data behind the web app.

Netlify serves static files only, so everything the browser needs is exported
once from the scored transactions into JSON (scored period only — for the
training workspace that is the out-of-sample test period):

  summary.json            results, case metrics, model and dataset facts
  alerts.json             per day: the top-N transactions by model score
  cases.json              every case (summary rows)
  cases/<case_id>.json    one case: accounts, transactions, draft STR (loaded on demand)
  accounts.json           risk summary of every account in an alert or a case
  graphs/<node_id>.json   that account's 1-hop / 2-hop transaction network (loaded on demand)

The training workspace writes into web/data/ (the deployed app). A scored
dataset gets a self-contained copy of the app in runs/<name>/web/.
"""

import json
import math
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

from . import config
from .cases import case_period, find_cases, summary
from .preprocess import graph_meta
from .store import ScoreStore
from .str_report import draft_str
from .utils import get_logger

log = get_logger(__name__)

ALERTS_PER_DAY = 500
GRAPH_HOP1_MAX = 60      # riskiest direct transactions in an account's network graph
GRAPH_HOP2_MAX = 120     # riskiest second-hop transactions
APP_FILES = ["index.html", "_headers", "css", "js"]


def account_graph_builder(store: ScoreStore, start, end):
    """
    Returns build(node_id) → the account's 1-hop and 2-hop transaction network in [start, end),
    riskiest first and capped, in a compact form for the floating graph on the account page:
      {"seed": id, "nodes": {id: label}, "edges": [[src, dst, score, label, amount, currency, format, time, hop]]}
    """
    e = store.edges
    src, dst = e["src"].to_numpy(), e["dst"].to_numpy()
    score, label = e["model_score"].to_numpy(), e["is_laundering"].to_numpy()
    times = e["Timestamp"].to_numpy()
    in_window = (times >= pd.Timestamp(start).to_datetime64()) & (times < pd.Timestamp(end).to_datetime64())
    amount = e["Amount Paid"].to_numpy()
    currency = e["Payment Currency"].astype(str).to_numpy()
    fmt = e["Payment Format"].astype(str).to_numpy()
    names = store.nodes["label"]

    def riskiest(ids: np.ndarray, k: int) -> np.ndarray:
        ids = ids[in_window[ids]]
        if len(ids) > k:
            ids = ids[np.argpartition(-score[ids], k - 1)[:k]]
        return ids[np.argsort(-score[ids], kind="stable")]

    def build(node: int) -> dict:
        hop1 = riskiest(store.incident_edges([node]), GRAPH_HOP1_MAX)
        neighbours = np.setdiff1d(np.union1d(src[hop1], dst[hop1]), [node])
        second = store.incident_edges(neighbours) if len(neighbours) else np.empty(0, dtype=np.int64)
        hop2 = riskiest(second[(src[second] != node) & (dst[second] != node)], GRAPH_HOP2_MAX)
        ids = np.concatenate([hop1, hop2])
        hops = [1] * len(hop1) + [2] * len(hop2)
        stamps = pd.to_datetime(times[ids]).strftime("%m-%d %H:%M")
        return {
            "seed": int(node),
            "nodes": {int(n): names[n] for n in np.union1d(src[ids], dst[ids])},
            "edges": [[int(src[i]), int(dst[i]), round(float(score[i]), 3), int(label[i]),
                       round(float(amount[i]), 2), currency[i], fmt[i], stamps[k], h]
                      for k, (i, h) in enumerate(zip(ids, hops))],
        }
    return build


def _clean(obj):
    """NaN/inf → null: browsers reject the NaN that Python's json writes by default."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _write(path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean(obj), default=str, separators=(",", ":"), allow_nan=False)
    path.write_text(text, encoding="utf-8")
    return len(text)


def copy_app(target) -> None:
    """Copy the web app's code (not its data) next to a workspace's exported data."""
    for name in APP_FILES:
        source, destination = config.WEB_DIR / name, target / name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)


def export_web() -> None:
    store = ScoreStore.from_artifacts()
    meta = graph_meta()
    start, end = case_period(store)
    WEB_DATA = config.WEB_DATA_DIR
    if WEB_DATA.exists():
        shutil.rmtree(WEB_DATA)            # generated folder: always rebuilt from scratch
    if WEB_DATA.parent != config.WEB_DIR:
        WEB_DATA.parent.mkdir(parents=True, exist_ok=True)
        copy_app(WEB_DATA.parent)

    # --- Cases on the test period ---
    cases = find_cases(store, start, end)
    case_of_edge = {e: c.case_id for c in cases for e in c.edge_ids}
    labels = store.nodes["label"]
    size = 0
    for c in cases:
        size += _write(WEB_DATA / "cases" / f"{c.case_id}.json", {
            **summary(c),
            "accounts": [{"node_id": a, "label": labels[a]} for a in c.accounts],
            "transactions": store.to_records(store.edges.loc[c.edge_ids].sort_values("Timestamp")),
            "str": draft_str(c, store),
        })
    size += _write(WEB_DATA / "cases.json", [summary(c) for c in cases])

    # --- Daily alert queues ---
    alerts = {}
    for day in pd.date_range(start, end - pd.Timedelta(days=1), freq="D"):
        records = store.to_records(store.alerts(limit=ALERTS_PER_DAY, day=day.date(), min_score=0.0))
        for r in records:
            r["case_id"] = case_of_edge.get(r["edge_id"])
        if records:
            alerts[str(day.date())] = records
    size += _write(WEB_DATA / "alerts.json", alerts)

    # --- Accounts that appear in an alert or a case ---
    cases_of_account = {}
    for c in cases:
        for a in c.accounts:
            cases_of_account.setdefault(a, []).append(c.case_id)
    in_alerts = {r[k] for day in alerts.values() for r in day for k in ("sender_id", "receiver_id")}
    accounts = {}
    network_of = account_graph_builder(store, start, end)
    for a in sorted(in_alerts | set(cases_of_account)):
        accounts[a] = {**store.account_risk(a), "label": labels[a], "cases": cases_of_account.get(a, [])}
        size += _write(WEB_DATA / "graphs" / f"{a}.json", network_of(a))
    size += _write(WEB_DATA / "accounts.json", accounts)

    # --- Results and facts for the overview page ---
    metrics = {k: v for k, v in store.metrics.items() if k != "history"}
    results = json.loads(config.RESULTS_JSON.read_text()) if config.RESULTS_JSON.exists() else None
    size += _write(WEB_DATA / "summary.json", {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "dataset": config.DATASET_NAME,
        "trained_on": store.metrics.get("trained_on", "HI-Small"),
        "has_labels": meta["has_labels"],
        "facts": {
            "transactions": len(store.edges),
            "accounts": len(store.nodes),
            "laundering": int(store.edges["is_laundering"].sum()),
        },
        "test_period": [start.date(), (end - pd.Timedelta(days=1)).date()],
        "alerts_per_day": ALERTS_PER_DAY,
        "alert_threshold": config.HIGH_RISK,
        "published_f1": config.PUBLISHED_F1,
        "metrics": metrics,
        "results": results,
    })
    log.info("Exported %d cases, %d alert days, %d accounts to %s (%.1f MB)",
             len(cases), len(alerts), len(accounts), WEB_DATA, size / 1e6)
