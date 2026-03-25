"""
src/features/feature_engineering.py
=====================================
Computes a rich feature matrix for every account node.

Feature groups
--------------
A. Node behavioral features   — derived from raw transactions
B. Graph topology features    — derived from NetworkX graph structure
C. Balance/flow features      — derived from amount columns
D. Temporal features          — sliding window behavioral drift

Why each feature indicates mule behaviour
-----------------------------------------
  out_degree            : mules forward money to many accounts (high fan-out)
  in_degree             : mule collectors receive from many sources
  passthrough_ratio     : total_sent / total_recv ≈ 1.0 for pass-through mules
  tx_velocity_24h       : rapid burst sending in short window
  drain_rate            : mules empty accounts after receiving
  pagerank              : mule hubs have high influence in the flow graph
  betweenness           : money passes *through* mule accounts (bridges)
  burst_ratio           : sudden spike in activity vs historical baseline
  small_round_tx_rate   : high rate of test transactions before large transfers
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import warnings
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import *

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# A. NODE BEHAVIORAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_node_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-account transaction statistics from the raw edge list.
    Returns DataFrame indexed by account.
    """
    print("  Computing node behavioral features...")

    # Pre-group for performance
    sent = df.groupby("sender_account")
    recv = df.groupby("receiver_account")

    sent_stats = sent["transaction_amount"].agg(
        total_sent="sum", avg_sent="mean", max_sent="max",
        std_sent="std", n_sent="count"
    ).fillna(0)

    recv_stats = recv["transaction_amount"].agg(
        total_recv="sum", avg_recv="mean", max_recv="max",
        std_recv="std", n_recv="count"
    ).fillna(0)

    # Unique counterparties
    unique_dest   = sent["receiver_account"].nunique().rename("unique_dest")
    unique_source = recv["sender_account"].nunique().rename("unique_source")

    # Balance-drain rate: rows where sender balance goes to ~zero
    # (works if dataset has balance columns; otherwise derived from amounts)
    drain_rate = pd.Series(dtype=float, name="drain_rate")
    if "sender_drained" in df.columns:
        drain_rate = sent["sender_drained"].mean().rename("drain_rate")

    # Pass-through ratio
    all_accounts = pd.Index(
        set(df["sender_account"]).union(set(df["receiver_account"]))
    )
    feats = pd.DataFrame(index=all_accounts)
    feats.index.name = "account"

    feats = feats.join(sent_stats,  how="left")
    feats = feats.join(recv_stats,  how="left")
    feats = feats.join(unique_dest, how="left")
    feats = feats.join(unique_source, how="left")
    if len(drain_rate):
        feats = feats.join(drain_rate, how="left")
    else:
        feats["drain_rate"] = 0.0

    feats = feats.fillna(0)

    # Derived
    feats["tx_count"]          = feats["n_sent"] + feats["n_recv"]
    feats["passthrough_ratio"] = feats["total_sent"] / (feats["total_recv"] + 1e-6)
    feats["fanout_ratio"]      = feats["unique_dest"] / (feats["n_sent"] + 1e-6)
    feats["degree_ratio"]      = (feats["n_sent"] + 1) / (feats["n_recv"] + 1)

    # Round-amount rate (test transaction signal)
    round_mask = df["transaction_amount"].apply(
        lambda x: x > 0 and x % TEST_TX_ROUND_MODULO == 0
    )
    round_sent = df[round_mask].groupby("sender_account").size().rename("round_tx_count")
    feats = feats.join(round_sent, how="left")
    feats["round_tx_count"] = feats["round_tx_count"].fillna(0)
    feats["small_round_tx_rate"] = feats["round_tx_count"] / (feats["n_sent"] + 1e-6)

    # Cash-out ratio (if type column is available)
    if "type" in df.columns:
        cashout = df[df["type"] == "CASH_OUT"].groupby("sender_account").size()
        feats   = feats.join(cashout.rename("n_cashout"), how="left")
        feats["n_cashout"]    = feats["n_cashout"].fillna(0)
        feats["cashout_rate"] = feats["n_cashout"] / (feats["n_sent"] + 1e-6)
    else:
        feats["cashout_rate"] = 0.0

    print(f"    Computed {len(feats.columns)} behavioral features for "
          f"{len(feats):,} accounts")
    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# B. GRAPH TOPOLOGY FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_graph_features(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute graph topology features for every node.
    """
    print("  Computing graph topology features...")

    nodes = list(G.nodes())

    # Degree features
    out_deg = dict(G.out_degree())
    in_deg  = dict(G.in_degree())

    # PageRank — high value = central hub in money flow (mule hubs are high)
    print("    PageRank...", end=" ", flush=True)
    pagerank = nx.pagerank(G, alpha=PAGERANK_ALPHA, max_iter=PAGERANK_MAX_ITER,
                           weight="total_amount")
    print("done")

    # Betweenness centrality (approx) — money passes *through* mule bridges
    print(f"    Betweenness centrality (k={BETWEENNESS_K})...", end=" ", flush=True)
    betweenness = nx.betweenness_centrality(
        G, k=min(BETWEENNESS_K, len(nodes)), normalized=True, weight="total_amount"
    )
    print("done")

    # Clustering coefficient (undirected view)
    print("    Clustering coefficient...", end=" ", flush=True)
    G_und = G.to_undirected()
    clustering = nx.clustering(G_und, weight="total_amount")
    print("done")

    # Weakly connected component size — mule clusters form large components
    comp_map = {}
    for comp in nx.weakly_connected_components(G):
        size = len(comp)
        for n in comp:
            comp_map[n] = size

    feats = pd.DataFrame({
        "account":         nodes,
        "out_degree":      [out_deg.get(n, 0) for n in nodes],
        "in_degree":       [in_deg.get(n, 0)  for n in nodes],
        "pagerank":        [pagerank.get(n, 0)     for n in nodes],
        "betweenness":     [betweenness.get(n, 0)  for n in nodes],
        "clustering":      [clustering.get(n, 0)   for n in nodes],
        "component_size":  [comp_map.get(n, 1)     for n in nodes],
    }).set_index("account")

    feats["degree_ratio_graph"] = (feats["out_degree"] + 1) / (feats["in_degree"] + 1)

    print(f"    Computed {len(feats.columns)} topology features for "
          f"{len(feats):,} nodes")
    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# C. TEMPORAL FEATURES (sliding window)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time-based behavioral features:
      - Transaction velocity in 1/7/30-day windows
      - Burst score (peak week / avg week)
      - Max inactivity gap (dormancy signal)
      - Sudden wakeup indicator
      - Days active
    """
    print("  Computing temporal features...")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"]      = df["timestamp"].dt.date

    all_accounts = list(
        set(df["sender_account"]).union(set(df["receiver_account"]))
    )

    records = []

    # Pre-group for performance
    by_sender = {k: v for k, v in df.groupby("sender_account")}
    by_recv   = {k: v for k, v in df.groupby("receiver_account")}

    for acc in all_accounts:
        sent_grp = by_sender.get(acc, pd.DataFrame())
        recv_grp = by_recv.get(acc, pd.DataFrame())

        cols_needed = ["timestamp", "transaction_amount"]
        sent_slice  = sent_grp[cols_needed] if len(sent_grp) > 0 else pd.DataFrame(columns=cols_needed)
        recv_slice  = recv_grp[cols_needed] if len(recv_grp) > 0 else pd.DataFrame(columns=cols_needed)
        all_tx = pd.concat([sent_slice, recv_slice])

        if len(all_tx) == 0:
            records.append({"account": acc})
            continue

        all_tx = all_tx.copy()
        all_tx["timestamp"] = pd.to_datetime(all_tx["timestamp"], errors="coerce")
        all_tx = all_tx.dropna(subset=["timestamp"]).sort_values("timestamp")

        if len(all_tx) == 0:
            records.append({"account": acc})
            continue

        first  = all_tx["timestamp"].min()
        last   = all_tx["timestamp"].max()
        active_days = max((last - first).days, 1)

        # Max inactivity gap (dormancy before activation)
        gaps = all_tx["timestamp"].diff().dt.total_seconds() / 86400
        max_gap = float(gaps.max()) if len(gaps) > 1 else 0.0

        # Weekly transaction counts → burst ratio
        all_tx["week"] = all_tx["timestamp"].dt.isocalendar().week.astype(int)
        wc = all_tx.groupby("week").size()
        max_weekly  = int(wc.max())
        mean_weekly = float(wc.mean())
        burst_ratio = max_weekly / max(mean_weekly, 1)

        # Coefficient of variation (stability — high = erratic, mule-like)
        weekly_cv = float(wc.std() / mean_weekly) if mean_weekly > 0 else 0.0

        # Transaction velocity per window (sent only)
        n_sent = len(sent_grp)
        vel_1d = vel_7d = vel_30d = 0
        if n_sent > 0:
            sent_grp = sent_grp.copy()
            sent_grp["timestamp"] = pd.to_datetime(sent_grp["timestamp"])
            ref = sent_grp["timestamp"].max()
            vel_1d  = int((sent_grp["timestamp"] >= ref - pd.Timedelta(days=1) ).sum())
            vel_7d  = int((sent_grp["timestamp"] >= ref - pd.Timedelta(days=7) ).sum())
            vel_30d = int((sent_grp["timestamp"] >= ref - pd.Timedelta(days=30)).sum())

        # Sudden wakeup: long gap then burst
        sudden_wakeup = int(max_gap > DORMANT_GAP_DAYS and burst_ratio > BURST_RATIO_THRESHOLD)

        # Small test transaction detection
        small_round = 0
        if len(sent_grp) > 0:
            early = sent_grp.head(5)
            small_round = int(
                ((early["transaction_amount"] < TEST_TX_MAX_AMOUNT) &
                 (early["transaction_amount"] % TEST_TX_ROUND_MODULO == 0)).sum()
            )

        records.append({
            "account":          acc,
            "active_days":      active_days,
            "max_gap_days":     max_gap,
            "burst_ratio":      burst_ratio,
            "weekly_cv":        weekly_cv,
            "max_weekly_txns":  max_weekly,
            "mean_weekly_txns": mean_weekly,
            "tx_velocity_1d":   vel_1d,
            "tx_velocity_7d":   vel_7d,
            "tx_velocity_30d":  vel_30d,
            "sudden_wakeup":    sudden_wakeup,
            "small_round_txns": small_round,
        })

    feats = pd.DataFrame(records).set_index("account").fillna(0)
    print(f"    Computed {len(feats.columns)} temporal features for "
          f"{len(feats):,} accounts")
    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# D. ASSEMBLE MASTER FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(
    df: pd.DataFrame,
    G: nx.DiGraph,
    labels: pd.Series = None,
) -> pd.DataFrame:
    """
    Merge all feature groups into a single feature matrix.

    Parameters
    ----------
    df     : canonical transaction DataFrame
    G      : summary weighted DiGraph
    labels : optional Series {account → is_fraud}

    Returns
    -------
    feature_matrix : DataFrame with one row per account, all features + label
    """
    print("\nBuilding master feature matrix...")

    node_feats  = compute_node_features(df)
    graph_feats = compute_graph_features(G)
    temp_feats  = compute_temporal_features(df)

    master = node_feats.join(graph_feats, how="outer", rsuffix="_g")
    master = master.join(temp_feats,       how="outer")
    master = master.fillna(0)

    if labels is not None:
        master["is_fraud"] = labels.reindex(master.index).fillna(0).astype(int)
    elif "is_fraud" in df.columns:
        fraud_map = (
            df.groupby("sender_account")["is_fraud"]
            .max()
            .reindex(master.index)
            .fillna(0)
            .astype(int)
        )
        master["is_fraud"] = fraud_map
    else:
        master["is_fraud"] = 0

    master.index.name = "account"
    master = master.reset_index()

    print(f"\n  Master feature matrix: {master.shape[0]:,} accounts × "
          f"{master.shape[1]-2} features (+account, +label)")
    print(f"  Fraud accounts in matrix: {master['is_fraud'].sum():,} "
          f"({master['is_fraud'].mean():.2%})")

    # Save
    out_path = DATA_PROCESSED / "feature_matrix.csv"
    master.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    return master


# ── Feature column list (used by models) ──────────────────────────────────────
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of numeric feature columns, excluding id and label."""
    drop = {"account", "is_fraud", "source"}
    return [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
