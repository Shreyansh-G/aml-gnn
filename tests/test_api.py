import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aml_gnn.store import ScoreStore
from api.main import create_app

API_KEY = "test-key"


def tiny_store() -> ScoreStore:
    nodes = pd.DataFrame({"node_id": [0, 1, 2, 3], "bank": ["001", "001", "002", "003"],
                          "account": ["A1", "B2", "C3", "A1"]})
    nodes["label"] = nodes["bank"] + " | " + nodes["account"]
    usd = pd.Categorical(["US Dollar"] * 5)
    edges = pd.DataFrame({
        "src": [0, 0, 1, 2, 3],
        "dst": [1, 2, 2, 3, 0],
        "Timestamp": pd.to_datetime(["2022-09-01 10:00", "2022-09-01 11:00", "2022-09-02 09:00",
                                     "2022-09-02 10:00", "2022-09-03 08:00"]),
        "Amount Received": [100.0, 200.0, 300.0, 400.0, 500.0],
        "Amount Paid": [100.0, 200.0, 300.0, 400.0, 500.0],
        "Receiving Currency": usd,
        "Payment Currency": usd,
        "Payment Format": pd.Categorical(["ACH", "Wire", "ACH", "Cash", "ACH"]),
        "split": [2, 2, 2, 2, 2],
        "is_laundering": [0, 1, 0, 1, 0],
        "pattern_type": pd.Categorical(["NONE", "FAN-OUT", "NONE", "CYCLE", "NONE"]),
        "model_score": [0.01, 0.95, 0.20, 0.70, 0.05],
    })
    return ScoreStore(edges, nodes, {"model": "test", "test_f1": 0.5})


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AML_API_KEY", API_KEY)
    with TestClient(create_app(tiny_store())) as c:
        yield c


def get(client, url, **params):
    return client.get(url, params=params, headers={"X-API-Key": API_KEY})


def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["transactions"] == 5


def test_v1_requires_valid_api_key(client):
    assert client.get("/v1/model").status_code == 401
    assert client.get("/v1/model", headers={"X-API-Key": "wrong"}).status_code == 401
    assert get(client, "/v1/model").status_code == 200


def test_missing_server_key_fails_closed(monkeypatch):
    monkeypatch.delenv("AML_API_KEY", raising=False)
    with TestClient(create_app(tiny_store())) as c:
        assert c.get("/v1/model", headers={"X-API-Key": "anything"}).status_code == 503


def test_transaction_lookup(client):
    body = get(client, "/v1/transactions/1").json()
    assert body["risk"] == "HIGH" and body["sender"] == "001 | A1" and body["pattern_type"] == "FAN-OUT"
    assert get(client, "/v1/transactions/99").status_code == 404


def test_account_search_and_risk(client):
    assert [m["node_id"] for m in get(client, "/v1/accounts", account="A1").json()] == [0, 3]
    assert get(client, "/v1/accounts", account="A1;DROP").status_code == 422   # input validation

    risk = get(client, "/v1/accounts/2").json()
    assert (risk["transactions_sent"], risk["transactions_received"]) == (1, 2)
    assert risk["risk"] == "HIGH" and risk["high_risk_transactions"] == 2 and risk["known_laundering"] == 2
    assert get(client, "/v1/accounts/42").status_code == 404


def test_neighborhood(client):
    body = get(client, "/v1/accounts/0/neighborhood", hops=1).json()
    assert body["accounts"] == 4 and len(body["transactions"]) == 5
    assert get(client, "/v1/accounts/0/neighborhood", hops=5).status_code == 422


def test_cases_detail_and_str(client):
    [case] = get(client, "/v1/cases").json()                 # edges 1 (0→2) and 3 (2→3) share account 2
    assert case["case_id"] == "C1" and case["num_transactions"] == 2 and case["known_laundering"] == 2
    detail = get(client, "/v1/cases/C1").json()
    assert [t["edge_id"] for t in detail["transactions"]] == [1, 3] and detail["account_ids"] == [0, 2, 3]
    report = get(client, "/v1/cases/C1/str")
    assert report.headers["content-type"].startswith("text/markdown") and "case C1" in report.text
    assert get(client, "/v1/cases/C7").status_code == 404
    assert get(client, "/v1/cases/../etc").status_code in (404, 422)


def test_alert_queue_is_ranked_and_filtered(client):
    assert [a["edge_id"] for a in get(client, "/v1/alerts").json()] == [1, 3]
    day = get(client, "/v1/alerts", day="2022-09-02", min_score=0.0).json()
    assert [a["edge_id"] for a in day] == [3, 2]
