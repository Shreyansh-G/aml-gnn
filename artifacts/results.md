# AML-GNN results — HI-Small

## Scores

5,078,345 transactions scored · **3,964** high risk (≥ 0.5, 0.08%) · 21,111 medium risk · 99th-percentile score 0.032

## Model quality (laundering class)

863,900 evaluated transactions, 1,611 laundering.

| Metric | Threshold 0.5 | Threshold 0.603 (tuned on validation) |
|---|---|---|
| Precision | 65.91% | 71.17% |
| Recall | 68.90% | 67.41% |
| **F1** | **67.37%** | **69.24%** |
| Laundering caught / missed | 1110 / 501 | 1086 / 525 |
| False alarms | 574 | 440 |

AUPRC **0.6963** · ROC-AUC **0.9924**

## Comparison with published results (F1 at the default threshold)

| Model | HI-Small F1 (%) |
|---|---|
| **This project (Multi-PNA)** | **67.37** |
| Multi-PNA + edge updates (Egressy et al., AAAI 2024) | 68.16 ± 2.65 |
| Multi-PNA (Egressy et al., AAAI 2024) | 64.59 ± 3.60 |
| XGBoost + graph features (Altman et al., NeurIPS 2023) | 63.23 ± 0.17 |
| GIN, plain GNN (Egressy et al., AAAI 2024) | 28.70 ± 1.13 |
| MLP, no graph (Lin et al., ICAIF 2024) | 0.42 ± 0.04 |

## Alert budget (top-K transactions reviewed per day)

| Alerts / day | Reviewed | Laundering caught | Precision | Recall |
|---|---|---|---|---|
| 100 | 726 | 588 | 80.99% | 36.50% |
| 500 | 2,108 | 1,175 | 55.74% | 72.94% |
| 1,000 | 3,108 | 1,255 | 40.38% | 77.90% |

## Recall by laundering typology

| Typology | Laundering transactions | Caught | Recall |
|---|---|---|---|
| GATHER-SCATTER | 381 | 371 | 97.38% |
| UNLISTED | 371 | 5 | 1.35% |
| SCATTER-GATHER | 239 | 220 | 92.05% |
| FAN-OUT | 131 | 129 | 98.47% |
| FAN-IN | 123 | 82 | 66.67% |
| STACK | 117 | 84 | 71.79% |
| CYCLE | 99 | 85 | 85.86% |
| RANDOM | 84 | 72 | 85.71% |
| BIPARTITE | 66 | 38 | 57.58% |

## Cases (high-risk transactions grouped into rings, score ≥ 0.5)

| Measure | Value |
|---|---|
| Flagged transactions | 1,693 |
| Cases to review | **431** (3.9 transactions per case) |
| Largest case | 76 transactions |
| Cases containing confirmed laundering | 211 (48.96%) |
| Typology explanation accuracy, all cases | 29.41% (on 204 cases with a known pattern) |
| Typology explanation accuracy, cases with ≥ 3 transactions | **54.37%** (on 103 cases) |

Detected typologies: PAIR 282, RANDOM 43, FAN-OUT 28, CYCLE 24, FAN-IN 19, GATHER-SCATTER 18, STACK 6, BIPARTITE 5, SCATTER-GATHER 4, SELF-TRANSFER 2

## Model and run

| Item | Value |
|---|---|
| model | Multi-PNA (reverse MP + ports + ego IDs + edge updates) |
| parameters | 57,297 |
| gpu | NVIDIA A100-SXM4-40GB |
| peak gpu memory gb | 17.06 |
| seed | 1 |
| train batches per epoch | 256 |
| time budget minutes | 25 |
| stopped by | time_budget |
| epochs trained | 12 |
| best epoch | 8 |
| best val f1 | 0.5755 |
| mean epoch train seconds | 107.1 |
| mean epoch val seconds | 12.3 |
| train transactions per second | 19,581 |
| scoring transactions per second | 75,115 |
| train minutes | 25.4 |
