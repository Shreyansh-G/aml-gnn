"""
config.py — every path and setting of the project, in one place.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR      = PROJECT_ROOT / "web"                 # static web app (code + HI-Small data)
DIST_DIR     = PROJECT_ROOT / "dist"                # Colab upload bundles
RUNS_DIR     = PROJECT_ROOT / "runs"                # one workspace per newly scored dataset
MODEL_DIR    = PROJECT_ROOT / "artifacts"           # the trained (HI-Small) model used for scoring

DATASET_NAME = "HI-Small"


def use_workspace(root: Path, dataset_name: str = "HI-Small") -> None:
    """
    Point every data and artifact path at *root*.

    The project folder itself is the HI-Small training workspace; scoring a
    new dataset uses runs/<name>/ so nothing of the trained model is overwritten.
    """
    global WORKSPACE, DATASET_NAME, RAW_DIR, GRAPH_DIR, ARTIFACTS_DIR, TRANSACTIONS_CSV, PATTERNS_TXT
    global NODES_PATH, EDGES_PATH, GRAPH_META_PATH, MODEL_PATH, FEATURE_SPEC_PATH, METRICS_PATH
    global SCORES_PATH, PREDICTIONS_PATH, RESULTS_JSON, RESULTS_MD, SEEDS_DIR, WEB_DATA_DIR

    WORKSPACE, DATASET_NAME = Path(root), dataset_name
    RAW_DIR       = WORKSPACE / "data" / "raw"
    GRAPH_DIR     = WORKSPACE / "data" / "processed" / "graph"
    ARTIFACTS_DIR = WORKSPACE / "artifacts"

    TRANSACTIONS_CSV = RAW_DIR / "HI-Small_Trans.csv"
    PATTERNS_TXT     = RAW_DIR / "HI-Small_Patterns.txt"
    NODES_PATH       = GRAPH_DIR / "nodes_mapped.parquet"
    EDGES_PATH       = GRAPH_DIR / "edges_labeled.parquet"
    GRAPH_META_PATH  = GRAPH_DIR / "graph_meta.json"      # source, labels present, split mode

    MODEL_PATH        = ARTIFACTS_DIR / "multipna_model.pt"
    FEATURE_SPEC_PATH = ARTIFACTS_DIR / "feature_spec.json"   # training-time encoding, reused for new data
    METRICS_PATH      = ARTIFACTS_DIR / "training_metrics.json"
    SCORES_PATH       = ARTIFACTS_DIR / "edge_scores.parquet"
    PREDICTIONS_PATH  = ARTIFACTS_DIR / "predictions.parquet"  # scored val/test probabilities
    RESULTS_JSON      = ARTIFACTS_DIR / "results.json"
    RESULTS_MD        = ARTIFACTS_DIR / "results.md"
    SEEDS_DIR         = ARTIFACTS_DIR / "seeds"                # training_metrics.json of each seed run

    # The main workspace feeds the deployed web app; others get a self-contained copy
    WEB_DATA_DIR = WEB_DIR / "data" if WORKSPACE == PROJECT_ROOT else WORKSPACE / "web" / "data"


use_workspace(PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

SPLIT_FRACTIONS      = (0.6, 0.2, 0.2)   # train / val / test, cut at day boundaries
CONTINUOUS_FEATURES  = ["Amount Received", "Amount Paid", "in_port", "out_port"]
CATEGORICAL_FEATURES = ["Receiving Currency", "Payment Currency", "Payment Format"]

# ---------------------------------------------------------------------------
# Model — IBM Multi-GNN's tuned PNA hyper-parameters
# ---------------------------------------------------------------------------

HIDDEN        = 20
NUM_LAYERS    = 2
DROPOUT       = 0.083
FINAL_DROPOUT = 0.288

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

LR            = 6.1e-4
CLASS_WEIGHTS = [1.0, 7.08]      # normal, laundering
BATCH_SIZE    = 8192             # target transactions per batch
NUM_NEIGHBORS = [100, 100]       # sampled transactions per account, per hop
EPOCHS        = 50
PATIENCE      = 10               # early stopping on validation F1
TRAIN_BATCHES = 256              # training batches per epoch, as in FraudGT/MEGA configs (0 = full pass)
TIME_BUDGET   = 25               # minutes of training, then keep the best epoch (0 = no limit)
SEED          = 1

# ---------------------------------------------------------------------------
# Results report
# ---------------------------------------------------------------------------

ALERT_BUDGETS = [100, 500, 1000]   # transactions investigators review per day

# Published HI-Small results (laundering-class F1, %, mean ± std over 5 runs), for comparisons
PUBLISHED_F1 = [
    ("Multi-PNA + edge updates (Egressy et al., AAAI 2024)",   "68.16 ± 2.65"),
    ("Multi-PNA (Egressy et al., AAAI 2024)",                  "64.59 ± 3.60"),
    ("XGBoost + graph features (Altman et al., NeurIPS 2023)", "63.23 ± 0.17"),
    ("GIN, plain GNN (Egressy et al., AAAI 2024)",             "28.70 ± 1.13"),
    ("MLP, no graph (Lin et al., ICAIF 2024)",                 "0.42 ± 0.04"),
]

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

HIGH_RISK   = 0.5
MEDIUM_RISK = 0.1
