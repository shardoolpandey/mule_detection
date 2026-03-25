"""
config/config.py
================
Central configuration for the entire mule detection system.
All thresholds, paths, and hyperparameters live here so that
nothing is hardcoded inside business logic modules.
"""

import os
from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_RAW        = ROOT_DIR / "data" / "raw"
DATA_PROCESSED  = ROOT_DIR / "data" / "processed"
DATA_SYNTHETIC  = ROOT_DIR / "data" / "synthetic"
OUTPUTS_PLOTS   = ROOT_DIR / "outputs" / "plots"
OUTPUTS_REPORTS = ROOT_DIR / "outputs" / "reports"
OUTPUTS_MODELS  = ROOT_DIR / "outputs" / "models"
OUTPUTS_RESULTS = ROOT_DIR / "outputs" / "results"

for d in [DATA_RAW, DATA_PROCESSED, DATA_SYNTHETIC,
          OUTPUTS_PLOTS, OUTPUTS_REPORTS, OUTPUTS_MODELS, OUTPUTS_RESULTS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Dataset settings ──────────────────────────────────────────────────────────
PAYSIM_PATH  = DATA_RAW / "paysim.csv"
AMLSIM_PATH  = DATA_RAW / "amlsim.csv"
SYNTHETIC_TX = DATA_SYNTHETIC / "synthetic_transactions.csv"

# Columns expected in transaction data (canonical schema)
TX_SCHEMA = {
    "sender":    "sender_account",
    "receiver":  "receiver_account",
    "amount":    "transaction_amount",
    "timestamp": "timestamp",
    "label":     "is_fraud",      # optional; may not exist in raw data
}

# ── Data generation (synthetic dataset) ──────────────────────────────────────
SYNTH_N_ACCOUNTS       = 5_000     # total accounts in synthetic dataset
SYNTH_N_TRANSACTIONS   = 50_000    # total transactions
SYNTH_MULE_FRACTION    = 0.04      # 4% of accounts are mules
SYNTH_N_MULE_NETWORKS  = 15        # number of coordinated mule networks
SYNTH_NETWORK_SIZE_MIN = 5         # min accounts per mule network
SYNTH_NETWORK_SIZE_MAX = 25        # max accounts per mule network
SYNTH_SEED             = 42

# ── Graph construction ────────────────────────────────────────────────────────
GRAPH_MIN_EDGE_WEIGHT  = 0         # minimum transaction count to keep an edge
GRAPH_SELF_LOOPS       = False     # whether to allow self-loops

# ── Feature engineering ───────────────────────────────────────────────────────
BETWEENNESS_K          = 200       # number of samples for approx. betweenness
PAGERANK_ALPHA         = 0.85
PAGERANK_MAX_ITER      = 200

# ── Lifecycle detection ───────────────────────────────────────────────────────
WINDOW_SIZES_DAYS      = [1, 3, 7, 14, 30]   # sliding windows for temporal features
TEST_TX_MAX_AMOUNT     = 500        # amounts below this = potential test transaction
TEST_TX_ROUND_MODULO   = 100        # round amount modulo threshold
DORMANT_GAP_DAYS       = 14         # silence before "sudden wakeup"
BURST_RATIO_THRESHOLD  = 4.0        # peak / avg weekly txns = burst

# ── Community detection ───────────────────────────────────────────────────────
LOUVAIN_RESOLUTION     = 1.0
MIN_COMMUNITY_SIZE     = 3          # ignore singleton/pair communities
SUSPICIOUS_COMMUNITY_MULE_RATE = 0.30  # community with >30% mules is flagged

# ── Model hyperparameters ─────────────────────────────────────────────────────
RF_N_ESTIMATORS        = 300
RF_MAX_DEPTH           = None
RF_MIN_SAMPLES_LEAF    = 2
RF_CLASS_WEIGHT        = "balanced"

GB_N_ESTIMATORS        = 200
GB_LEARNING_RATE       = 0.05
GB_MAX_DEPTH           = 5

ISO_N_ESTIMATORS       = 200
ISO_CONTAMINATION      = "auto"     # set to float if mule rate is known

# ── Evaluation ────────────────────────────────────────────────────────────────
TEST_SIZE              = 0.20
RANDOM_STATE           = 42
CV_FOLDS               = 5

# ── Visualization ─────────────────────────────────────────────────────────────
PLOT_DPI               = 150
PLOT_FIGSIZE_LARGE     = (14, 10)
PLOT_FIGSIZE_MEDIUM    = (10, 7)
PLOT_FIGSIZE_SMALL     = (8, 5)
NODE_COLOR_NORMAL      = "#85B7EB"
NODE_COLOR_MULE        = "#E24B4A"
NODE_COLOR_SUSPECTED   = "#EF9F27"
EDGE_COLOR_NORMAL      = "#B4B2A9"
EDGE_COLOR_SUSPICIOUS  = "#D85A30"
