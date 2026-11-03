"""
str_report.py — draft a Suspicious Transaction Report (STR) for a case.

Turns a case into a structured Markdown draft an analyst can review, edit and
file: subject accounts, transactions, grounds for suspicion, and a review
checklist. It states facts from the data and model only; the decision to file
stays with the analyst.
"""

from datetime import datetime

import pandas as pd

from . import config
from .cases import Case
from .store import ScoreStore

MAX_LISTED_TRANSACTIONS = 50


def _money(amounts: pd.Series, currencies: pd.Series) -> str:
    """Totals per currency, e.g. '12,500.00 US Dollar; 3,100.00 Euro'."""
    totals = amounts.groupby(currencies.astype(str)).sum().sort_values(ascending=False)
    return "; ".join(f"{v:,.2f} {c}" for c, v in totals.items()) or "—"


def draft_str(case: Case, store: ScoreStore) -> str:
    txs = store.edges.loc[case.edge_ids].sort_values("Timestamp")
    label = store.nodes["label"]

    # --- Subject accounts and their roles ---
    rows = []
    for acc in case.accounts:
        sent, received = txs[txs["src"] == acc], txs[txs["dst"] == acc]
        role = ("Sender and receiver" if len(sent) and len(received)
                else "Sender" if len(sent) else "Receiver")
        rows.append(f"| {label[acc]} | {role} | {len(sent)} | "
                    f"{_money(sent['Amount Paid'], sent['Payment Currency'])} | {len(received)} | "
                    f"{_money(received['Amount Received'], received['Receiving Currency'])} |")

    # --- Grounds for suspicion ---
    pass_through = sum(1 for acc in case.accounts
                       if (txs["src"] == acc).any() and (txs["dst"] == acc).any())
    currencies = sorted(set(txs["Payment Currency"].astype(str)) | set(txs["Receiving Currency"].astype(str)))
    formats = txs["Payment Format"].astype(str).value_counts()
    grounds = [
        f"**Typology — {case.typology}:** {case.evidence}",
        f"{len(txs)} transactions among {len(case.accounts)} accounts within {case.hours:.1f} hours "
        f"({case.first_seen:%Y-%m-%d %H:%M} to {case.last_seen:%Y-%m-%d %H:%M}).",
        f"Model risk scores {txs['model_score'].min():.2f}–{case.max_score:.2f} "
        f"(alert threshold {config.HIGH_RISK}).",
        "Payment formats: " + ", ".join(f"{f} ({n})" for f, n in formats.items()) + ".",
    ]
    if pass_through:
        grounds.append(f"{pass_through} account(s) both receive and forward funds (pass-through behaviour).")
    if len(currencies) > 1:
        grounds.append(f"Funds move across {len(currencies)} currencies: {', '.join(currencies)}.")

    listed = txs.head(MAX_LISTED_TRANSACTIONS).rename(columns=lambda c: c.replace(" ", "_"))
    tx_rows = [f"| {t.Timestamp:%Y-%m-%d %H:%M} | {label[t.src]} | {label[t.dst]} | "
               f"{t.Amount_Paid:,.2f} {t.Payment_Currency} | {t.Payment_Format} | {t.model_score:.3f} |"
               for t in listed.itertuples()]

    metrics = store.metrics
    return "\n".join([
        f"# Suspicious Transaction Report — DRAFT (case {case.case_id})",
        "",
        f"*Generated {datetime.now():%Y-%m-%d %H:%M} by the AML-GNN investigation system. "
        "Draft for analyst review — not a filing.*",
        "",
        "## 1. Summary",
        "",
        f"The transaction-monitoring model flagged {len(txs)} transactions between {len(case.accounts)} "
        f"accounts from {case.first_seen:%Y-%m-%d %H:%M} to {case.last_seen:%Y-%m-%d %H:%M}, "
        f"consistent with a **{case.typology}** laundering pattern. {case.evidence} "
        f"Amounts paid: {_money(txs['Amount Paid'], txs['Payment Currency'])}.",
        "",
        "## 2. Subject accounts",
        "",
        "| Account (bank \\| number) | Role | Sent | Amount sent | Received | Amount received |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        "## 3. Grounds for suspicion",
        "",
        *[f"- {g}" for g in grounds],
        "",
        f"## 4. Transactions ({len(listed)} of {len(txs)}, chronological)",
        "",
        "| Time | From | To | Amount paid | Format | Model score |",
        "|---|---|---|---|---|---|",
        *tx_rows,
        "",
        "## 5. Analyst checklist",
        "",
        "- [ ] Verify KYC and beneficial ownership of the subject accounts.",
        "- [ ] Review account activity outside the flagged window.",
        "- [ ] Confirm or dismiss the typology and document the rationale.",
        "- [ ] Decide on filing per the bank's STR policy and PMLA obligations.",
        "",
        "## 6. Model information",
        "",
        f"{metrics.get('model', 'Multi-PNA')} — test F1 {100 * metrics.get('test_f1', 0):.1f}%, "
        f"precision {100 * metrics.get('test_precision', 0):.1f}%, recall {100 * metrics.get('test_recall', 0):.1f}% "
        "on the IBM AMLworld HI-Small benchmark (synthetic data).",
        "",
    ])
