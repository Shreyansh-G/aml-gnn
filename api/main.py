"""
main.py — AML scoring REST API.

  GET /health                                  liveness + model summary (public)
  GET /v1/model                                training metrics
  GET /v1/transactions/{edge_id}               one scored transaction
  GET /v1/accounts?account=800737690           find accounts by account number
  GET /v1/accounts/{node_id}                   account risk summary
  GET /v1/accounts/{node_id}/neighborhood      scored transactions around an account
  GET /v1/alerts?day=2022-09-10&limit=100      alert queue: highest-risk transactions
  GET /v1/cases?start=&end=&min_score=0.5      laundering-ring cases with explained typology
  GET /v1/cases/{case_id}                      one case with its transactions
  GET /v1/cases/{case_id}/str                  draft suspicious transaction report (Markdown)

Every /v1 route requires the X-API-Key header. Interactive docs at /docs.

Run:  python -m aml_gnn api        (or: uvicorn api.main:app)
"""

from contextlib import asynccontextmanager
from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.responses import PlainTextResponse

from aml_gnn import config
from aml_gnn.cases import Case, case_period, find_cases, get_case, summary
from aml_gnn.store import ScoreStore
from aml_gnn.str_report import draft_str

from .schemas import AccountMatch, AccountRisk, CaseDetail, CaseSummary, Health, Neighborhood, Transaction
from .security import require_api_key


def get_store(request: Request) -> ScoreStore:
    return request.app.state.store


router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


@router.get("/model")
def model_metrics(store: ScoreStore = Depends(get_store)) -> dict:
    return store.metrics


@router.get("/transactions/{edge_id}", response_model=Transaction)
def transaction(edge_id: int = Path(ge=0), store: ScoreStore = Depends(get_store)):
    tx = store.transaction(edge_id)
    if tx is None:
        raise HTTPException(404, "Transaction not found")
    return tx


@router.get("/accounts", response_model=List[AccountMatch])
def find_accounts(account: str = Query(min_length=1, max_length=32, pattern=r"^[0-9A-Za-z]+$"),
                  store: ScoreStore = Depends(get_store)):
    return store.find_accounts(account)[["node_id", "bank", "account"]].to_dict("records")


@router.get("/accounts/{node_id}", response_model=AccountRisk)
def account_risk(node_id: int = Path(ge=0), store: ScoreStore = Depends(get_store)):
    risk = store.account_risk(node_id)
    if risk is None:
        raise HTTPException(404, "Account not found")
    return risk


@router.get("/accounts/{node_id}/neighborhood", response_model=Neighborhood)
def neighborhood(node_id: int = Path(ge=0),
                 hops: int = Query(1, ge=1, le=2),
                 max_edges: int = Query(500, ge=1, le=2000),
                 store: ScoreStore = Depends(get_store)):
    if not store.has_account(node_id):
        raise HTTPException(404, "Account not found")
    edges, accounts = store.neighborhood(node_id, hops=hops, max_edges=max_edges)
    return {"node_id": node_id, "hops": hops, "accounts": len(accounts),
            "transactions": store.to_records(edges)}


@router.get("/alerts", response_model=List[Transaction])
def alerts(limit: int = Query(100, ge=1, le=1000),
           day: Optional[date] = None,
           min_score: float = Query(config.HIGH_RISK, ge=0.0, le=1.0),
           store: ScoreStore = Depends(get_store)):
    return store.to_records(store.alerts(limit=limit, day=day, min_score=min_score))


class CaseQuery:
    """Shared case parameters: whole days [start, end] (default: test period) and alert threshold."""

    def __init__(self, start: Optional[date] = None, end: Optional[date] = None,
                 min_score: float = Query(config.HIGH_RISK, ge=0.05, le=1.0)):
        self.start, self.end, self.min_score = start, end, min_score


def _find_case(case_id: str, query: CaseQuery, store: ScoreStore) -> Case:
    case = get_case(store, case_id, *case_period(store, query.start, query.end), query.min_score)
    if case is None:
        raise HTTPException(404, "Case not found for these parameters")
    return case


@router.get("/cases", response_model=List[CaseSummary])
def cases(limit: int = Query(100, ge=1, le=1000), query: CaseQuery = Depends(),
          store: ScoreStore = Depends(get_store)):
    found = find_cases(store, *case_period(store, query.start, query.end), query.min_score)
    return [summary(c) for c in found[:limit]]


@router.get("/cases/{case_id}", response_model=CaseDetail)
def case_detail(case_id: str = Path(pattern=r"^C\d+$"), query: CaseQuery = Depends(),
                store: ScoreStore = Depends(get_store)):
    case = _find_case(case_id, query, store)
    transactions = store.edges.loc[case.edge_ids].sort_values("Timestamp")
    return {**summary(case), "account_ids": case.accounts, "transactions": store.to_records(transactions)}


@router.get("/cases/{case_id}/str", response_class=PlainTextResponse)
def case_str(case_id: str = Path(pattern=r"^C\d+$"), query: CaseQuery = Depends(),
             store: ScoreStore = Depends(get_store)):
    return PlainTextResponse(draft_str(_find_case(case_id, query, store), store), media_type="text/markdown")


def create_app(store: Optional[ScoreStore] = None) -> FastAPI:
    """App factory; loads the scored transactions once at startup unless *store* is given."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store if store is not None else ScoreStore.from_artifacts()
        yield

    app = FastAPI(title="AML Multi-PNA Scoring API", version="1.0.0", lifespan=lifespan)

    @app.get("/health", response_model=Health)
    def health(request: Request):
        s = request.app.state.store
        return {"status": "ok", "transactions": s.num_transactions,
                "model": s.metrics.get("model", ""), "test_f1": s.metrics.get("test_f1", 0.0)}

    app.include_router(router)
    return app


app = create_app()
