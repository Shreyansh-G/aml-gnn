"""
schemas.py — typed response models (validated and documented in /docs).
"""

from datetime import datetime
from typing import List, Literal

from pydantic import BaseModel, Field

Risk = Literal["HIGH", "MEDIUM", "LOW"]


class Transaction(BaseModel):
    edge_id: int
    timestamp: datetime
    sender_id: int
    sender: str = Field(description="bank | account")
    receiver_id: int
    receiver: str = Field(description="bank | account")
    amount_paid: float
    payment_currency: str
    amount_received: float
    receiving_currency: str
    payment_format: str
    score: float = Field(description="Model laundering probability, 0–1")
    risk: Risk
    is_laundering: int = Field(description="Ground-truth label of the synthetic dataset")
    pattern_type: str = Field(description="Laundering typology (NONE for normal transactions)")


class AccountMatch(BaseModel):
    node_id: int
    bank: str
    account: str


class AccountRisk(BaseModel):
    node_id: int
    bank: str
    account: str
    transactions_sent: int
    transactions_received: int
    max_score: float
    mean_score: float
    high_risk_transactions: int
    known_laundering: int
    risk: Risk


class Neighborhood(BaseModel):
    node_id: int
    hops: int
    accounts: int
    transactions: List[Transaction]


class CaseSummary(BaseModel):
    case_id: str
    typology: str
    evidence: str
    num_accounts: int
    num_transactions: int
    max_score: float
    mean_score: float
    first_seen: datetime
    last_seen: datetime
    known_laundering: int = Field(description="Laundering transactions in the case per the dataset labels")


class CaseDetail(CaseSummary):
    account_ids: List[int]
    transactions: List[Transaction]


class Health(BaseModel):
    status: str
    transactions: int
    model: str
    test_f1: float
