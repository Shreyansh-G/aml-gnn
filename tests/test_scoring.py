"""Scoring a new dataset with an already-trained model (generalisation pipeline)."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from aml_gnn import artifacts, config
from aml_gnn.data import encode_edges, fit_feature_spec, load_graph
from aml_gnn.model import MultiPNA, model_args_for
from aml_gnn.preprocess import build_graph
from aml_gnn.score import score_dataset

MAPPING = {"timestamp": "txn_time", "sender_account": "from_acct", "receiver_account": "to_acct",
           "amount_paid": "amount", "payment_currency": "ccy", "payment_format": "channel",
           "is_laundering": "is_suspicious"}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    yield tmp_path
    config.use_workspace(config.PROJECT_ROOT)            # restore the HI-Small workspace


def amlworld_csv(path, rows=400, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.Timestamp("2022-09-01") + pd.to_timedelta(rng.integers(0, 6 * 24 * 60, rows), unit="min")
    amount = rng.uniform(10, 5000, rows).round(2)
    currency = rng.choice(["US Dollar", "Euro"], rows)
    pd.DataFrame({
        "Timestamp": ts.strftime("%Y/%m/%d %H:%M"), "From Bank": rng.choice(["001", "002"], rows),
        "Account": [f"A{i}" for i in rng.integers(0, 30, rows)], "To Bank": rng.choice(["001", "002"], rows),
        "Account.1": [f"A{i}" for i in rng.integers(0, 30, rows)], "Amount Received": amount,
        "Receiving Currency": currency, "Amount Paid": amount, "Payment Currency": currency,
        "Payment Format": rng.choice(["ACH", "Wire", "Cash"], rows), "Is Laundering": (rng.random(rows) < 0.08).astype(int),
    }).to_csv(path, index=False, header=["Timestamp", "From Bank", "Account", "To Bank", "Account", "Amount Received",
                                          "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format",
                                          "Is Laundering"])


def custom_csv(path, rows=150, seed=1, labels=True):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "txn_time": (pd.Timestamp("2023-03-01") + pd.to_timedelta(rng.integers(0, 3 * 24 * 60, rows), unit="min"))
                    .strftime("%Y-%m-%d %H:%M:%S"),
        "from_acct": [f"X{i}" for i in rng.integers(0, 20, rows)],
        "to_acct": [f"X{i}" for i in rng.integers(0, 20, rows)],
        "amount": rng.uniform(5, 9000, rows).round(2),
        "ccy": rng.choice(["Euro", "Rupee"], rows),           # Rupee was never seen in training
        "channel": rng.choice(["Wire", "UPI"], rows),          # UPI was never seen in training
    })
    if labels:
        df["is_suspicious"] = np.where(rng.random(rows) < 0.1, "yes", "no")
    df.to_csv(path, index=False)


def trained_model_dir(root):
    """Stand-in for the HI-Small training run: a model + feature spec + metrics, as artifacts/ holds."""
    amlworld_csv(root / "train.csv")
    config.use_workspace(root / "train_ws", dataset_name="tiny")
    build_graph(root / "train.csv", mode="train")
    graph, spec = load_graph(torch.device("cpu"))
    args = model_args_for(graph, graph.edges_in(0))
    artifacts.save_checkpoint(MultiPNA(**args).state_dict(), args)
    artifacts.save_feature_spec(spec)
    artifacts.save_metrics({"model": "Multi-PNA", "parameters": 1, "trained_on": "tiny", "test_f1": 0.5,
                            "test_precision": 0.5, "test_recall": 0.5, "test_auprc": 0.5})
    config.RESULTS_JSON.write_text(json.dumps({"test": {"at_tuned_threshold": {"threshold": 0.42}}}))
    return config.ARTIFACTS_DIR


def test_encoding_is_fixed_by_the_spec_and_unseen_categories_are_zero():
    train = pd.DataFrame({"Amount Received": [10.0, 100.0], "Amount Paid": [10.0, 100.0],
                          "in_port": [0, 1], "out_port": [0, 1], "split": [0, 0],
                          "Receiving Currency": pd.Categorical(["Euro", "US Dollar"]),
                          "Payment Currency": pd.Categorical(["Euro", "US Dollar"]),
                          "Payment Format": pd.Categorical(["ACH", "Wire"])})
    spec = fit_feature_spec(train)
    new = train.assign(**{"Receiving Currency": pd.Categorical(["Rupee", "Euro"]),
                          "Payment Format": pd.Categorical(["UPI", "Wire"])})
    x = encode_edges(new, spec)
    assert x.shape == (2, 4 + 2 + 2 + 2)                      # width fixed by the spec, not the new data
    assert x[0, 4:6].sum() == 0 and x[1, 4] == 1               # unseen currency → zeros; Euro → its slot
    assert x[0, 8:10].sum() == 0 and x[1, 9] == 1              # unseen format → zeros; Wire → its slot
    assert torch.allclose(x[:, :4], encode_edges(train, spec)[:, :4])   # same normalisation as training


def test_custom_csv_without_labels(workspace):
    custom_csv(workspace / "bank.csv", labels=False)
    config.use_workspace(workspace / "ws", dataset_name="bank")
    meta = build_graph(workspace / "bank.csv", columns={k: v for k, v in MAPPING.items() if k != "is_laundering"},
                       mode="score")
    edges = pd.read_parquet(config.EDGES_PATH)
    assert meta["has_labels"] is False and (edges["is_laundering"] == 0).all() and (edges["split"] == 2).all()
    assert set(pd.read_parquet(config.NODES_PATH)["bank"]) == {"0"}       # no bank column → one institution


def test_score_new_dataset_end_to_end(workspace):
    model_dir = trained_model_dir(workspace)
    custom_csv(workspace / "bank.csv")
    results = score_dataset(workspace / "bank.csv", "mybank", columns=MAPPING, model_dir=model_dir,
                            batch_size=64, neighbors=[5, 5])

    run = workspace / "runs" / "mybank"
    scores = pd.read_parquet(run / "artifacts" / "edge_scores.parquet")["score"]
    assert len(scores) == 150 and scores.notna().all()
    assert results["has_labels"] and results["test"]["threshold_source"].endswith("not re-tuned")
    assert results["test"]["at_tuned_threshold"]["threshold"] == 0.42      # the model's own threshold
    summary = json.loads((run / "web" / "data" / "summary.json").read_text())
    assert summary["dataset"] == "mybank" and summary["trained_on"] == "tiny"
    assert (run / "web" / "index.html").exists() and (run / "web" / "js" / "main.js").exists()
