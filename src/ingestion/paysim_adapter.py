"""
src/ingestion/paysim_adapter.py
================================
Drop-in adapter for running the full pipeline on your PaySim dataset.

Your PaySim columns:
  type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
  nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

Usage
-----
  python src/ingestion/paysim_adapter.py --path data/raw/paysim.csv

Or import and call prepare_paysim() from main_pipeline.py via --source paysim.
"""

import pandas as pd
import numpy as np
import argparse
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import *


def prepare_paysim(path: str = None, sample_n: int = None) -> pd.DataFrame:
    """
    Load PaySim CSV and enrich it with derived features before
    passing it into the canonical pipeline.

    Steps
    -----
    1. Load CSV
    2. Filter to TRANSFER and CASH_OUT (only these carry fraud labels)
    3. Reconstruct synthetic timestamps from row order (your dataset has no 'step')
    4. Derive balance-based features (drain_rate, pass-through flag)
    5. Rename to canonical schema:
         nameOrig  → sender_account
         nameDest  → receiver_account
         amount    → transaction_amount
         isFraud   → is_fraud
    6. Return canonical DataFrame ready for build_transaction_graph()
    """
    path = path or str(PAYSIM_PATH)
    print(f"Loading PaySim from: {path}")
    df = pd.read_csv(path)
    print(f"  Raw rows: {len(df):,} | Columns: {list(df.columns)}")

    # ── 1. Filter to money-moving types ──────────────────────────────────────
    if "type" in df.columns:
        df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
        print(f"  After TRANSFER/CASH_OUT filter: {len(df):,} rows")
    else:
        df = df.copy()

    # ── 2. Reconstruct timestamp ──────────────────────────────────────────────
    # Your PaySim has no 'step' column — use row order as proxy
    # (PaySim rows are loosely chronological by design)
    if "step" in df.columns:
        base = pd.Timestamp("2023-01-01")
        df["timestamp"] = base + pd.to_timedelta(df["step"], unit="h")
    else:
        df = df.reset_index(drop=True)
        base = pd.Timestamp("2023-01-01")
        df["timestamp"] = base + pd.to_timedelta(df.index, unit="h")
        print("  No 'step' column — reconstructed timestamps from row order")

    # ── 3. Balance-derived features ───────────────────────────────────────────
    # sender_drained: account fully emptied after sending (mule signal)
    if "oldbalanceOrg" in df.columns and "newbalanceOrig" in df.columns:
        df["sender_drained"] = (
            (df["newbalanceOrig"] < 1.0) &
            (df["oldbalanceOrg"] > 0)
        ).astype(int)
    else:
        df["sender_drained"] = 0

    # dest_no_increase: receiver balance unchanged despite receiving funds
    if "oldbalanceDest" in df.columns and "newbalanceDest" in df.columns:
        df["dest_no_increase"] = (
            (df["newbalanceDest"] - df["oldbalanceDest"]) < 1.0
        ).astype(int)
    else:
        df["dest_no_increase"] = 0

    # is_round_amount: test transaction proxy
    df["is_round_amount"] = (df["amount"] % 100 == 0).astype(int)

    # ── 4. Rename to canonical schema ─────────────────────────────────────────
    rename_map = {
        "nameOrig":  "sender_account",
        "nameDest":  "receiver_account",
        "amount":    "transaction_amount",
        "isFraud":   "is_fraud",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure required columns exist
    for col in ["sender_account", "receiver_account", "transaction_amount"]:
        if col not in df.columns:
            raise ValueError(f"Required column missing after rename: {col}")

    if "is_fraud" not in df.columns:
        df["is_fraud"] = 0

    # ── 5. Optional sampling ──────────────────────────────────────────────────
    if sample_n and len(df) > sample_n:
        # Stratified: keep all fraud rows + random normal rows
        fraud_df  = df[df["is_fraud"] == 1]
        normal_df = df[df["is_fraud"] == 0]
        n_normal  = sample_n - len(fraud_df)
        if n_normal > 0 and len(normal_df) > n_normal:
            normal_df = normal_df.sample(n_normal, random_state=RANDOM_STATE)
        df = pd.concat([fraud_df, normal_df]).sort_values("timestamp").reset_index(drop=True)
        print(f"  Sampled to {len(df):,} rows (all {len(fraud_df):,} fraud rows kept)")

    # ── 6. Final summary ──────────────────────────────────────────────────────
    print(f"\n  PaySim dataset prepared:")
    print(f"  Transactions     : {len(df):,}")
    print(f"  Unique senders   : {df['sender_account'].nunique():,}")
    print(f"  Unique receivers : {df['receiver_account'].nunique():,}")
    print(f"  Fraud rate       : {df['is_fraud'].mean():.4%}")
    print(f"  Sender drained   : {df['sender_drained'].sum():,} transactions")
    print(f"  Date range       : {df['timestamp'].min().date()} → "
          f"{df['timestamp'].max().date()}")

    # Save canonical version
    out = df[[
        "sender_account", "receiver_account", "transaction_amount",
        "timestamp", "is_fraud", "sender_drained", "dest_no_increase",
        "is_round_amount"
    ]]
    out.to_csv(DATA_PROCESSED / "paysim_canonical.csv", index=False)
    print(f"  Saved → {DATA_PROCESSED / 'paysim_canonical.csv'}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",   required=True, help="Path to paysim.csv")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    df = prepare_paysim(path=args.path, sample_n=args.sample)
    print("\nFirst 5 rows of prepared data:")
    print(df[["sender_account","receiver_account","transaction_amount",
              "timestamp","is_fraud","sender_drained"]].head().to_string())
