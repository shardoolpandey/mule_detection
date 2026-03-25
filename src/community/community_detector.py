"""
src/community/community_detector.py
=====================================
Detects coordinated mule networks as graph communities.

Methods
-------
1. Louvain community detection  — finds densely connected clusters
2. Weakly connected components  — finds all reachable account groups
3. Suspicious cluster scoring   — ranks communities by mule density

Why mule networks form clusters
--------------------------------
Money mules operate in coordinated networks. Within a network:
  - Collectors receive from the same fraud sources
  - Distributors are all connected to the same collectors
  - The network has higher internal transaction density than random accounts

This creates a detectable "community structure" in the graph where
mule accounts appear in the same dense cluster.
"""

import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import *

# Optional: python-louvain package
try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False
    print("  [Warning] python-louvain not installed. "
          "Install with: pip install python-louvain")
    print("  Falling back to greedy modularity communities.")


def detect_communities(
    G: nx.DiGraph,
    method: str = "louvain",
) -> dict[str, int]:
    """
    Assign each node to a community ID.

    Parameters
    ----------
    G      : directed transaction graph
    method : 'louvain' | 'components' | 'greedy'

    Returns
    -------
    partition : {account → community_id}
    """
    print(f"\nDetecting communities (method={method})...")
    G_und = G.to_undirected()

    if method == "louvain":
        if LOUVAIN_AVAILABLE:
            partition = community_louvain.best_partition(
                G_und, resolution=LOUVAIN_RESOLUTION, random_state=RANDOM_STATE
            )
        else:
            method = "greedy"

    if method == "greedy":
        communities = nx.community.greedy_modularity_communities(G_und)
        partition   = {}
        for cid, comm in enumerate(communities):
            for node in comm:
                partition[node] = cid

    if method == "components":
        partition = {}
        for cid, comp in enumerate(nx.weakly_connected_components(G)):
            for node in comp:
                partition[node] = cid

    n_communities = len(set(partition.values()))
    print(f"  Found {n_communities:,} communities")
    return partition


def score_communities(
    partition: dict[str, int],
    feature_matrix: pd.DataFrame,
    G: nx.DiGraph,
) -> pd.DataFrame:
    """
    Score each community on multiple suspicion indicators.

    Suspicion indicators:
      mule_rate         — fraction of labelled mules in community
      avg_pagerank      — mean PageRank (hub centrality)
      avg_betweenness   — mean betweenness centrality
      avg_passthrough   — mean pass-through ratio
      avg_burst_ratio   — mean burst score
      internal_density  — edges within community / possible edges
      size              — number of accounts

    Returns DataFrame of communities sorted by suspicion_score.
    """
    print("  Scoring community suspicion levels...")

    # Map accounts to features
    fm = feature_matrix.set_index("account") if "account" in feature_matrix.columns \
         else feature_matrix

    community_records = []

    # Group accounts by community
    comm_to_nodes: dict[int, list] = defaultdict(list)
    for node, cid in partition.items():
        comm_to_nodes[cid].append(node)

    for cid, nodes in comm_to_nodes.items():
        if len(nodes) < MIN_COMMUNITY_SIZE:
            continue

        # Account subset in feature matrix
        node_feats = fm.loc[fm.index.isin(nodes)]

        mule_rate    = node_feats["is_fraud"].mean()  if "is_fraud" in node_feats.columns else 0
        avg_pr       = node_feats["pagerank"].mean()  if "pagerank" in node_feats.columns else 0
        avg_btw      = node_feats["betweenness"].mean() if "betweenness" in node_feats.columns else 0
        avg_pass     = node_feats["passthrough_ratio"].mean() if "passthrough_ratio" in node_feats.columns else 0
        avg_burst    = node_feats["burst_ratio"].mean() if "burst_ratio" in node_feats.columns else 0
        avg_velocity = node_feats["tx_velocity_7d"].mean() if "tx_velocity_7d" in node_feats.columns else 0

        # Internal edge density
        subgraph = G.subgraph(nodes)
        n = len(nodes)
        max_edges = n * (n - 1)
        density = subgraph.number_of_edges() / max_edges if max_edges > 0 else 0

        # Total flow through community
        total_flow = sum(
            d.get("total_amount", 0)
            for _, _, d in subgraph.edges(data=True)
        )

        # Composite suspicion score (weighted combination)
        suspicion = (
            0.30 * mule_rate +
            0.20 * min(avg_pr * 100, 1.0) +
            0.15 * min(avg_btw * 10, 1.0) +
            0.15 * min(avg_pass, 1.0) +
            0.10 * min(avg_burst / 10.0, 1.0) +
            0.10 * min(density * 10, 1.0)
        )

        community_records.append({
            "community_id":    cid,
            "size":            n,
            "mule_rate":       round(mule_rate, 4),
            "avg_pagerank":    round(avg_pr, 6),
            "avg_betweenness": round(avg_btw, 6),
            "avg_passthrough": round(avg_pass, 4),
            "avg_burst_ratio": round(avg_burst, 3),
            "avg_velocity_7d": round(avg_velocity, 2),
            "internal_density":round(density, 4),
            "total_flow":      round(total_flow, 2),
            "suspicion_score": round(suspicion, 4),
            "is_suspicious":   int(suspicion > 0.25 or mule_rate > SUSPICIOUS_COMMUNITY_MULE_RATE),
            "accounts":        nodes,
        })

    comm_df = pd.DataFrame(community_records)
    comm_df = comm_df.sort_values("suspicion_score", ascending=False).reset_index(drop=True)

    n_suspicious = comm_df["is_suspicious"].sum()
    print(f"  Communities analysed  : {len(comm_df):,}")
    print(f"  Flagged as suspicious : {n_suspicious:,}")

    return comm_df


def get_suspicious_accounts(
    partition: dict[str, int],
    comm_df: pd.DataFrame,
) -> list[str]:
    """Return list of accounts belonging to suspicious communities."""
    suspicious_cids = set(
        comm_df[comm_df["is_suspicious"] == 1]["community_id"].tolist()
    )
    return [acc for acc, cid in partition.items() if cid in suspicious_cids]
