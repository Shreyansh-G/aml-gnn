"""
preprocess.py — transaction CSV → transaction graph
====================================================

Builds the two Parquet files used for training, scoring and the apps, following
the standard benchmark setup (Altman et al. 2023; Egressy et al. 2024):

  - Node  = (bank, account) pair → integer node_id.
  - Edge  = one transaction; label from `Is Laundering` when the data has one.
  - Split = mode "train": temporal 60/20/20 train/val/test, cut at day boundaries;
            mode "score": every transaction is scored (split 2) by a trained model.
  - Ports = Multi-GNN port numbers (see graph_features.port_numbers).
  - pattern_type = laundering typology from an AMLworld Patterns.txt, if given.

Input is either the IBM AMLworld CSV format or any CSV plus a column mapping
(see examples/column_mapping.json), so new datasets can be scored.
"""

import json
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from . import config
from .graph_features import load_pattern_types, port_numbers, temporal_split
from .utils import get_logger

log = get_logger(__name__)

# The AMLworld header repeats "Account"; name the columns explicitly.
CSV_COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
]
AMLWORLD_TIME_FORMAT = "%Y/%m/%d %H:%M"

# Column-mapping keys (your CSV's column names go on the right-hand side of the mapping file)
FIELDS = {
    "timestamp": "Timestamp",
    "sender_bank": "From Bank",
    "sender_account": "Account",
    "receiver_bank": "To Bank",
    "receiver_account": "Account.1",
    "amount_received": "Amount Received",
    "receiving_currency": "Receiving Currency",
    "amount_paid": "Amount Paid",
    "payment_currency": "Payment Currency",
    "payment_format": "Payment Format",
    "is_laundering": "Is Laundering",
}
REQUIRED_FIELDS = ["timestamp", "sender_account", "receiver_account"]
CHUNK_ROWS = 2_000_000


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _standardise(chunk: pd.DataFrame, columns: Optional[dict]) -> pd.DataFrame:
    """Rename a custom CSV to the internal columns and fill optional ones with safe defaults."""
    if columns is None:
        return chunk
    missing = [f for f in REQUIRED_FIELDS if f not in columns]
    if missing or not ({"amount_paid", "amount_received"} & set(columns)):
        raise ValueError(f"Column mapping needs {REQUIRED_FIELDS} and amount_paid or amount_received; "
                         f"missing {missing or ['an amount']}")
    out = pd.DataFrame({FIELDS[f]: chunk[col] for f, col in columns.items() if f in FIELDS})
    for bank in ("From Bank", "To Bank"):
        out[bank] = out.get(bank, pd.Series("0", index=out.index))          # single institution
    out["Amount Paid"] = out.get("Amount Paid", out.get("Amount Received"))
    out["Amount Received"] = out.get("Amount Received", out["Amount Paid"])
    paid_ccy = out.get("Payment Currency", out.get("Receiving Currency", pd.Series("Unknown", index=out.index)))
    out["Payment Currency"] = paid_ccy
    out["Receiving Currency"] = out.get("Receiving Currency", paid_ccy)
    out["Payment Format"] = out.get("Payment Format", pd.Series("Unknown", index=out.index))
    if "Is Laundering" in out:      # accept 1/0, true/false, yes/no
        truthy = out["Is Laundering"].str.strip().str.lower().isin(["1", "1.0", "true", "t", "yes", "y"])
        out["Is Laundering"] = np.where(truthy, "1", "0")
    return out[[c for c in CSV_COLUMNS if c in out]]


def read_transactions(path=None, columns: Optional[dict] = None, time_format: Optional[str] = None,
                      start: Optional[date] = None, end: Optional[date] = None) -> pd.DataFrame:
    """
    Read the CSV in chunks as text (exact strings are needed for pattern matching),
    parse timestamps into `ts`, and keep only rows in [start, end] when given.
    """
    path = path or config.TRANSACTIONS_CSV
    log.info("Reading %s ...", path)
    if columns is None:       # AMLworld format
        reader = pd.read_csv(path, header=0, names=CSV_COLUMNS, dtype=str, chunksize=CHUNK_ROWS)
        time_format = time_format or AMLWORLD_TIME_FORMAT
    else:
        reader = pd.read_csv(path, dtype=str, chunksize=CHUNK_ROWS)

    parts = []
    for chunk in reader:
        chunk = _standardise(chunk, columns)
        try:
            chunk["ts"] = pd.to_datetime(chunk["Timestamp"], format=time_format)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Could not parse timestamps ({error}); pass --time-format") from error
        if start is not None:
            chunk = chunk[chunk["ts"] >= pd.Timestamp(start)]
        if end is not None:
            chunk = chunk[chunk["ts"] < pd.Timestamp(end) + pd.Timedelta(days=1)]
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    if df.empty:
        raise ValueError("No transactions left to process (check the file and the date window).")
    log.info("  → %d transactions%s.", len(df), " in the date window" if start or end else "")
    return df


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_nodes(df: pd.DataFrame):
    """One node per (bank, account) → (sorted account keys, nodes table)."""
    keys = pd.concat([df["From Bank"] + "|" + df["Account"],
                      df["To Bank"] + "|" + df["Account.1"]])
    accounts = pd.Index(keys.unique()).sort_values()
    nodes = pd.DataFrame({"node_id": np.arange(len(accounts), dtype=np.int32)})
    nodes[["bank", "account"]] = accounts.to_series().str.split("|", n=1, expand=True).to_numpy()
    return accounts, nodes


def build_edges(df: pd.DataFrame, accounts: pd.Index, mode: str = "train") -> pd.DataFrame:
    """One edge per transaction, in input order, with features, split and label."""
    seconds = (df["ts"] - df["ts"].min()).dt.total_seconds().to_numpy(np.int64)
    src = accounts.get_indexer(df["From Bank"] + "|" + df["Account"]).astype(np.int32)
    dst = accounts.get_indexer(df["To Bank"] + "|" + df["Account.1"]).astype(np.int32)
    currency = pd.CategoricalDtype(sorted(set(df["Receiving Currency"]) | set(df["Payment Currency"])))
    label = (df["Is Laundering"].astype(np.int8).to_numpy() if "Is Laundering" in df
             else np.zeros(len(df), dtype=np.int8))

    return pd.DataFrame({
        "src":                src,
        "dst":                dst,
        "Timestamp":          df["ts"],
        "Amount Received":    pd.to_numeric(df["Amount Received"]),
        "Amount Paid":        pd.to_numeric(df["Amount Paid"]),
        "Receiving Currency": df["Receiving Currency"].astype(currency),
        "Payment Currency":   df["Payment Currency"].astype(currency),
        "Payment Format":     df["Payment Format"].astype("category"),
        "in_port":            port_numbers(dst, src, seconds),
        "out_port":           port_numbers(src, dst, seconds),
        "split":              temporal_split(seconds) if mode == "train" else np.full(len(df), 2, np.int8),
        "is_laundering":      label,
    })


def pattern_types(df: pd.DataFrame, path=None) -> pd.Series:
    """Typology per transaction: NONE (normal), a Patterns.txt type, or UNLISTED."""
    types = pd.Series("NONE", index=df.index, dtype=object)
    if "Is Laundering" not in df:
        return types.astype("category")
    laundering = (df["Is Laundering"] == "1").to_numpy()
    matchable = path is not None and path.exists() and all(c in df for c in CSV_COLUMNS)
    pattern_of = load_pattern_types(path) if matchable else {}
    lines = df.loc[laundering, CSV_COLUMNS].apply(",".join, axis=1) if pattern_of else None
    types[laundering] = lines.map(pattern_of).fillna("UNLISTED") if pattern_of else "UNLISTED"
    return types.astype("category")


def build_graph(csv=None, patterns=None, columns: Optional[dict] = None, time_format: Optional[str] = None,
                start: Optional[date] = None, end: Optional[date] = None, mode: str = "train") -> dict:
    """Read a transaction CSV and write the graph Parquet files of the active workspace."""
    log.info("=== Building transaction graph (%s mode) ===", mode)
    config.GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    patterns = patterns if patterns is not None else (config.PATTERNS_TXT if csv is None else None)

    df = read_transactions(csv, columns, time_format, start, end)
    accounts, nodes = build_nodes(df)
    edges = build_edges(df, accounts, mode)
    edges["pattern_type"] = pattern_types(df, patterns)

    nodes.to_parquet(config.NODES_PATH, index=False)
    edges.to_parquet(config.EDGES_PATH, index=False)
    meta = {
        "dataset": config.DATASET_NAME,
        "source": str(csv or config.TRANSACTIONS_CSV),
        "mode": mode,
        "has_labels": "Is Laundering" in df,
        "has_patterns": bool(patterns and patterns.exists() and "Is Laundering" in df),
        "transactions": len(edges),
        "accounts": len(nodes),
        "first_transaction": str(edges["Timestamp"].min()),
        "last_transaction": str(edges["Timestamp"].max()),
    }
    config.GRAPH_META_PATH.write_text(json.dumps(meta, indent=2))
    log.info("  → saved %s and %s", config.NODES_PATH, config.EDGES_PATH)
    report(nodes, edges, meta)
    return meta


def graph_meta() -> dict:
    """Metadata of the active workspace's graph (defaults for graphs built before it existed)."""
    if config.GRAPH_META_PATH.exists():
        return json.loads(config.GRAPH_META_PATH.read_text())
    return {"dataset": config.DATASET_NAME, "mode": "train", "has_labels": True, "has_patterns": True}


def report(nodes: pd.DataFrame, edges: pd.DataFrame, meta: dict) -> None:
    """Log the data integrity report."""
    label = edges["is_laundering"]
    log.info("=" * 55)
    log.info("DATA INTEGRITY REPORT — %s", meta["dataset"])
    log.info("  Accounts (nodes)        : %d", len(nodes))
    log.info("  Transactions (edges)    : %d", len(edges))
    log.info("  Period                  : %s → %s", meta["first_transaction"], meta["last_transaction"])
    if meta["has_labels"]:
        log.info("  Laundering transactions : %d (%.4f%%)", label.sum(), 100 * label.mean())
    else:
        log.info("  Labels                  : none (scores only, no accuracy metrics)")
    for s, name in enumerate(["Train", "Val", "Test/scored"]):
        part = edges[edges["split"] == s]
        if len(part):
            log.info("  %-11s : %9d edges, %5d laundering", name, len(part), part["is_laundering"].sum())
    if meta["has_labels"]:
        log.info("  Laundering by typology  : %s",
                 edges.loc[label == 1, "pattern_type"].value_counts().to_dict())
    log.info("=" * 55)
