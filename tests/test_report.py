import numpy as np
import pytest

from aml_gnn.report import (
    alert_budget,
    average_precision,
    best_f1_threshold,
    classification_metrics,
    recall_by_typology,
    roc_auc,
)


def test_classification_metrics_counts():
    m = classification_metrics(np.array([0.9, 0.6, 0.4, 0.2]), np.array([1, 0, 1, 0]), 0.5)
    assert (m["true_positives"], m["false_positives"], m["false_negatives"]) == (1, 1, 1)
    assert m["precision"] == m["recall"] == m["f1"] == 0.5


def test_best_f1_threshold_picks_the_clean_cut():
    assert best_f1_threshold(np.array([0.9, 0.8, 0.3, 0.1]), np.array([1, 1, 0, 0])) == 0.8


def test_ranking_metrics_match_hand_computed_values():
    probs, labels = np.array([0.9, 0.8, 0.7]), np.array([1, 0, 1])
    assert average_precision(probs, labels) == pytest.approx((1 + 2 / 3) / 2)
    assert roc_auc(probs, labels) == pytest.approx(0.5)
    assert roc_auc(np.array([0.9, 0.1]), np.array([1, 0])) == 1.0


def test_alert_budget_takes_top_k_per_day():
    probs = np.array([0.9, 0.5, 0.1, 0.8, 0.7])
    labels = np.array([1, 0, 1, 0, 1])
    days = np.array(["d1", "d1", "d1", "d2", "d2"])
    [row] = alert_budget(probs, labels, days, [1])
    assert (row["alerts_reviewed"], row["laundering_caught"]) == (2, 1)
    assert row["precision"] == 0.5 and row["recall"] == pytest.approx(1 / 3)


def test_recall_by_typology():
    rows = recall_by_typology(np.array([True, False, True, False]), np.array([1, 1, 1, 0]),
                              np.array(["FAN-OUT", "FAN-OUT", "CYCLE", "NONE"]))
    assert {r["typology"]: r["recall"] for r in rows} == {"FAN-OUT": 0.5, "CYCLE": 1.0}
