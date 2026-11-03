"""
report.py — quantified results for the active workspace (training run or scored dataset).

Reads the scored transactions, predictions.parquet and training_metrics.json, and
writes artifacts/results.json and artifacts/results.md:

  - score distribution (always — no labels needed)
  - with labels: precision / recall / F1 at 0.5 and at a tuned threshold; AUPRC and
    ROC-AUC. A training run tunes the threshold on its validation split; a new
    dataset reuses the model's training-time threshold (never re-tuned on the data
    being evaluated), so its numbers measure out-of-dataset generalisation
  - alert budget: precision and recall if investigators review the top-K per day
  - recall per laundering typology (FAN-OUT, CYCLE, …) when patterns are known
  - mean ± std of test F1 when artifacts/seeds/ holds several runs
  - cases: how alerts group into laundering rings, and typology accuracy
  - GPU, memory, timing and throughput of training and scoring

No PyTorch needed, so it runs anywhere the artifacts are.
"""

import json
from collections import Counter

import numpy as np
import pandas as pd

from . import config
from .cases import case_period, find_cases
from .preprocess import graph_meta
from .store import ScoreStore
from .utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def classification_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred = probs >= threshold
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1,
            "true_positives": tp, "false_positives": fp, "false_negatives": fn}


def best_f1_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximising F1; F1 at the top-k cut-off is 2·TP / (k + positives)."""
    order = np.argsort(-probs, kind="stable")
    tp = np.cumsum(labels[order])
    f1 = 2 * tp / (np.arange(1, len(probs) + 1) + labels.sum())
    return float(probs[order][int(np.argmax(f1))])


def average_precision(probs: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise, as in scikit-learn)."""
    order = np.argsort(-probs, kind="stable")
    y = labels[order]
    precision_at_k = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((precision_at_k * y).sum() / max(y.sum(), 1))


def roc_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the rank-sum (Mann–Whitney) formula."""
    ranks = pd.Series(probs).rank().to_numpy()
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def alert_budget(probs: np.ndarray, labels: np.ndarray, days: np.ndarray, budgets) -> list:
    """Precision / recall if investigators review the top-K scored transactions each day."""
    ranked = (pd.DataFrame({"prob": probs, "y": labels, "day": days})
              .sort_values("prob", ascending=False, kind="stable"))
    total = int(labels.sum())
    rows = []
    for k in budgets:
        reviewed = ranked.groupby("day").head(k)
        caught = int(reviewed["y"].sum())
        rows.append({"alerts_per_day": k, "alerts_reviewed": len(reviewed), "laundering_caught": caught,
                     "precision": caught / len(reviewed) if len(reviewed) else 0.0,
                     "recall": caught / total if total else 0.0})
    return rows


def recall_by_typology(flagged: np.ndarray, labels: np.ndarray, types: np.ndarray) -> list:
    """Share of each laundering typology that the model flags."""
    df = pd.DataFrame({"flagged": flagged, "type": types})[labels == 1]
    counts = df.groupby("type")["flagged"].agg(["size", "sum"]).sort_values("size", ascending=False)
    return [{"typology": t, "laundering": int(r["size"]), "caught": int(r["sum"]),
             "recall": float(r["sum"] / r["size"])} for t, r in counts.iterrows()]


def case_metrics(store: ScoreStore, min_score: float = config.HIGH_RISK) -> dict:
    """Alerts → cases on the scored period, and rule-based typology accuracy against the labels."""
    cases = find_cases(store, *case_period(store), min_score=min_score)
    if not cases:
        return None
    pattern = store.edges["pattern_type"].astype(str)
    laundering = store.edges["is_laundering"] == 1
    evaluated = correct = evaluated_multi = correct_multi = 0
    for case in cases:
        truth = pattern.loc[case.edge_ids][laundering.loc[case.edge_ids]]
        truth = truth[~truth.isin(["NONE", "UNLISTED"])]
        if len(truth):
            hit = case.typology == truth.mode().iloc[0]
            evaluated += 1
            correct += hit
            if len(case.edge_ids) >= 3:      # a lone transaction cannot show a pattern's shape
                evaluated_multi += 1
                correct_multi += hit

    alerts = sum(len(c.edge_ids) for c in cases)
    with_laundering = sum(1 for c in cases if c.known_laundering)
    return {
        "min_score": min_score,
        "alerts": alerts,
        "cases": len(cases),
        "alerts_per_case": alerts / len(cases),
        "cases_with_laundering": with_laundering,
        "share_with_laundering": with_laundering / len(cases),
        "largest_case": max(len(c.edge_ids) for c in cases),
        "typology_evaluated": evaluated,
        "typology_accuracy": correct / evaluated if evaluated else None,
        "typology_evaluated_3plus": evaluated_multi,
        "typology_accuracy_3plus": correct_multi / evaluated_multi if evaluated_multi else None,
        "typologies": dict(Counter(c.typology for c in cases).most_common()),
    }


def seed_summary() -> dict:
    """Mean ± std of test metrics over the runs saved in artifacts/seeds/ (needs ≥ 2)."""
    runs = [json.loads(p.read_text()) for p in sorted(config.SEEDS_DIR.glob("*.json"))]
    if len(runs) < 2:
        return None
    out = {"runs": len(runs), "seeds": [r["seed"] for r in runs]}
    for key in ("test_f1", "test_precision", "test_recall", "test_auprc"):
        values = np.array([r[key] for r in runs])
        out[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    return out


def score_summary(store: ScoreStore) -> dict:
    """How the scores are distributed over the dataset (needs no labels)."""
    scores = store.edges["model_score"].to_numpy()
    return {
        "transactions": int(len(scores)),
        "high_risk": int((scores >= config.HIGH_RISK).sum()),
        "medium_risk": int(((scores >= config.MEDIUM_RISK) & (scores < config.HIGH_RISK)).sum()),
        "high_risk_share": float((scores >= config.HIGH_RISK).mean()),
        "p99_score": float(np.quantile(scores, 0.99)),
    }


def label_metrics(store: ScoreStore, training: dict, has_patterns: bool) -> dict:
    """Accuracy on the scored transactions that have labels (see module docstring for the threshold)."""
    preds = pd.read_parquet(config.PREDICTIONS_PATH)
    label = store.edges["is_laundering"].to_numpy()
    val, test = preds[preds["split"] == 1], preds[preds["split"] == 2]
    p_test, test_ids = test["prob"].to_numpy(), test["edge_id"].to_numpy()
    y_test = label[test_ids]

    if len(val):
        threshold = best_f1_threshold(val["prob"].to_numpy(), label[val["edge_id"].to_numpy()])
        source = "tuned on validation"
    else:
        threshold = training.get("reference_threshold", config.HIGH_RISK)
        source = f"from {training.get('trained_on', 'training')} validation, not re-tuned"

    out = {
        "test": {
            "transactions": len(test),
            "laundering": int(y_test.sum()),
            "threshold_source": source,
            "at_0.5": classification_metrics(p_test, y_test, 0.5),
            "at_tuned_threshold": classification_metrics(p_test, y_test, threshold),
            "auprc": average_precision(p_test, y_test),
            "roc_auc": roc_auc(p_test, y_test),
        },
        "alert_budget": alert_budget(p_test, y_test,
                                     store.edges["Timestamp"].dt.floor("D").to_numpy()[test_ids],
                                     config.ALERT_BUDGETS),
    }
    if has_patterns:
        out["recall_by_typology"] = recall_by_typology(
            p_test >= threshold, y_test, store.edges["pattern_type"].astype(str).to_numpy()[test_ids])
    return out

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

TRAINING_KEYS = ["model", "trained_on", "parameters", "gpu", "peak_gpu_memory_gb", "seed",
                 "train_batches_per_epoch", "time_budget_minutes", "stopped_by", "epochs_trained",
                 "best_epoch", "best_val_f1", "mean_epoch_train_seconds", "mean_epoch_val_seconds",
                 "train_transactions_per_second", "reference_threshold", "scored_transactions",
                 "scoring_seconds", "scoring_transactions_per_second", "train_minutes"]


def build_report() -> dict:
    """Compute every result for the active workspace, write results.json and results.md."""
    meta = graph_meta()
    training = json.loads(config.METRICS_PATH.read_text())
    store = ScoreStore.from_artifacts()

    results = {
        "dataset": config.DATASET_NAME,
        "trained_on": training.get("trained_on", "HI-Small"),
        "has_labels": meta["has_labels"],
        "scores": score_summary(store),
    }
    if meta["has_labels"]:
        results.update(label_metrics(store, training, meta.get("has_patterns", False)))
    results.update({
        "seeds": seed_summary(),
        "training": {k: training[k] for k in TRAINING_KEYS if k in training},
        "cases": case_metrics(store),
    })

    config.RESULTS_JSON.write_text(json.dumps(results, indent=2))
    config.RESULTS_MD.write_text(to_markdown(results))
    log.info("Saved %s and %s", config.RESULTS_JSON, config.RESULTS_MD)
    return results


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def _fmt(v) -> str:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"


def to_markdown(r: dict) -> str:
    s = r["scores"]
    lines = [f"# AML-GNN results — {r['dataset']}", ""]
    if r["dataset"] != r["trained_on"]:
        lines += [f"Scored by the model trained on **{r['trained_on']}**, without retraining "
                  "(out-of-dataset generalisation).", ""]
    lines += [
        "## Scores",
        "",
        f"{s['transactions']:,} transactions scored · **{s['high_risk']:,}** high risk "
        f"(≥ {config.HIGH_RISK}, {_pct(s['high_risk_share'])}) · {s['medium_risk']:,} medium risk · "
        f"99th-percentile score {s['p99_score']:.3f}",
    ]

    if "test" in r:
        t, at05, tuned = r["test"], r["test"]["at_0.5"], r["test"]["at_tuned_threshold"]
        lines += [
            "",
            "## Model quality (laundering class)",
            "",
            f"{t['transactions']:,} evaluated transactions, {t['laundering']:,} laundering.",
            "",
            f"| Metric | Threshold 0.5 | Threshold {tuned['threshold']:.3f} ({t['threshold_source']}) |",
            "|---|---|---|",
            f"| Precision | {_pct(at05['precision'])} | {_pct(tuned['precision'])} |",
            f"| Recall | {_pct(at05['recall'])} | {_pct(tuned['recall'])} |",
            f"| **F1** | **{_pct(at05['f1'])}** | **{_pct(tuned['f1'])}** |",
            f"| Laundering caught / missed | {at05['true_positives']} / {at05['false_negatives']} "
            f"| {tuned['true_positives']} / {tuned['false_negatives']} |",
            f"| False alarms | {at05['false_positives']:,} | {tuned['false_positives']:,} |",
            "",
            f"AUPRC **{t['auprc']:.4f}** · ROC-AUC **{t['roc_auc']:.4f}**",
        ]
        if r["dataset"] == "HI-Small":
            lines += [
                "",
                "## Comparison with published results (F1 at the default threshold)",
                "",
                "| Model | HI-Small F1 (%) |",
                "|---|---|",
                f"| **This project (Multi-PNA)** | **{100 * at05['f1']:.2f}** |",
                *[f"| {name} | {f1} |" for name, f1 in config.PUBLISHED_F1],
            ]
        lines += [
            "",
            "## Alert budget (top-K transactions reviewed per day)",
            "",
            "| Alerts / day | Reviewed | Laundering caught | Precision | Recall |",
            "|---|---|---|---|---|",
            *[f"| {a['alerts_per_day']:,} | {a['alerts_reviewed']:,} | {a['laundering_caught']:,} "
              f"| {_pct(a['precision'])} | {_pct(a['recall'])} |" for a in r["alert_budget"]],
        ]

    if r.get("recall_by_typology"):
        lines += [
            "",
            "## Recall by laundering typology",
            "",
            "| Typology | Laundering transactions | Caught | Recall |",
            "|---|---|---|---|",
            *[f"| {x['typology']} | {x['laundering']:,} | {x['caught']:,} | {_pct(x['recall'])} |"
              for x in r["recall_by_typology"]],
        ]

    seeds = r.get("seeds")
    if seeds:
        lines += ["", f"## Reproducibility ({seeds['runs']} seeds: {seeds['seeds']})", "",
                  "| Metric | Mean ± std |", "|---|---|",
                  *[f"| {k.replace('test_', '').upper()} | {100 * v['mean']:.2f} ± {100 * v['std']:.2f} |"
                    for k, v in seeds.items() if k.startswith("test_")]]

    c = r.get("cases")
    if c:
        lines += ["", f"## Cases (high-risk transactions grouped into rings, score ≥ {c['min_score']})", "",
                  "| Measure | Value |", "|---|---|",
                  f"| Flagged transactions | {c['alerts']:,} |",
                  f"| Cases to review | **{c['cases']:,}** ({c['alerts_per_case']:.1f} transactions per case) |",
                  f"| Largest case | {c['largest_case']:,} transactions |"]
        if r["has_labels"]:
            lines.append(f"| Cases containing confirmed laundering | {c['cases_with_laundering']:,} "
                         f"({_pct(c['share_with_laundering'])}) |")
        if c["typology_accuracy"] is not None:
            lines.append(f"| Typology explanation accuracy, all cases | {_pct(c['typology_accuracy'])} "
                         f"(on {c['typology_evaluated']:,} cases with a known pattern) |")
        if c.get("typology_accuracy_3plus") is not None:
            lines.append(f"| Typology explanation accuracy, cases with ≥ 3 transactions | "
                         f"**{_pct(c['typology_accuracy_3plus'])}** (on {c['typology_evaluated_3plus']:,} cases) |")
        lines += ["", "Detected typologies: " + ", ".join(f"{k} {v}" for k, v in c["typologies"].items())]

    tr = r["training"]
    lines += ["", "## Model and run", "", "| Item | Value |", "|---|---|",
              *[f"| {k.replace('_', ' ')} | {_fmt(v)} |" for k, v in tr.items()]]
    return "\n".join(lines) + "\n"
