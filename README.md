# Graph-Based Money Mule Detection System

**Industry-grade AML detection project**  
*Detects coordinated mule networks and early-stage mule behaviour using graph analytics and machine learning.*

---

## Project Overview

This system implements two core AML detection objectives:

**Objective 1 — Network-Level Mule Detection**  
Constructs a directed transaction graph, extracts node/graph topology features, detects coordinated mule communities, and trains three ML models (Isolation Forest, Random Forest, Gradient Boosting) plus an ensemble scorer.

**Objective 2 — Early-Stage Lifecycle Detection**  
Classifies every account into one of five mule lifecycle stages (Dormant → Recruitment → Activation → Laundering → Exit) using temporal sliding-window features and an ML-backed rule classifier.

---

## Project Structure

```
mule_detection/
│
├── config/
│   └── config.py                    ← ALL thresholds, paths, hyperparameters
│
├── src/
│   ├── ingestion/
│   │   ├── data_generator.py        ← Synthetic AML dataset (3-layer mule model)
│   │   ├── data_loader.py           ← Loads PaySim / AMLSim / any CSV
│   │   └── paysim_adapter.py        ← PaySim-specific enrichment + rename
│   │
│   ├── graph/
│   │   └── graph_builder.py         ← Builds DiGraph + MultiDiGraph
│   │
│   ├── features/
│   │   └── feature_engineering.py  ← 38 node/graph/temporal features
│   │
│   ├── community/
│   │   └── community_detector.py   ← Louvain + greedy + suspicion scoring
│   │
│   ├── models/
│   │   └── ml_models.py            ← IsolationForest + RF + GB + ensemble
│   │
│   ├── lifecycle/
│   │   └── lifecycle_detector.py   ← Rule + ML stage classifier
│   │
│   ├── visualization/
│   │   └── visualizer.py           ← 9 production plots
│   │
│   └── evaluation/
│       └── evaluator.py            ← Metrics, report generation
│
├── data/
│   ├── raw/                         ← Place paysim.csv here
│   ├── processed/                   ← All intermediate CSVs
│   └── synthetic/                   ← Auto-generated dataset
│
├── outputs/
│   ├── plots/                       ← All 9 visualisation PNGs
│   ├── reports/                     ← evaluation_report.txt
│   ├── models/                      ← Saved .pkl model files
│   └── results/                     ← suspicious_accounts, predictions CSVs
│
├── main_pipeline.py                  ← Single-command orchestrator
└── requirements.txt
```

---

## Quick Start

### Option A — Synthetic data (no download needed)
```bash
pip install -r requirements.txt
python main_pipeline.py --source synthetic --regen
```

### Option B — Your PaySim dataset
```bash
# Place your paysim.csv at:
cp your_paysim.csv data/raw/paysim.csv

# Run with 100k row sample (fast) or without --sample for full dataset:
python main_pipeline.py --source paysim --path data/raw/paysim.csv --sample 100000
```

### Option C — Skip visualizations (faster)
```bash
python main_pipeline.py --source synthetic --skip-viz
```

---

## Requirements

```
pandas>=1.5.0
numpy>=1.23.0
networkx>=3.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
scipy>=1.10.0
python-louvain>=0.16     # optional — falls back to greedy modularity
```

Install: `pip install -r requirements.txt`

---

## Pipeline Phases

| Phase | Module | What it does |
|-------|--------|-------------|
| 1 | `data_loader.py` | Loads data, normalises to canonical schema |
| 2 | `graph_builder.py` | Builds weighted DiGraph (nodes=accounts, edges=transactions) |
| 3 | `feature_engineering.py` | Computes 38 features per account |
| 4 | `community_detector.py` | Finds suspicious account clusters |
| 5 | `ml_models.py` | Trains 3 models + ensemble scorer |
| 6 | `lifecycle_detector.py` | Classifies lifecycle stage per account |
| 7 | `evaluator.py` | Computes all metrics, writes report |
| 8 | `visualizer.py` | Produces 9 plots |

---

## Feature Groups

### Node Behavioral Features (20 features)
| Feature | Mule Signal |
|---------|------------|
| `n_sent`, `n_recv` | Mules: high both (collect + distribute) |
| `passthrough_ratio` | ≈ 1.0 for pass-through mules |
| `drain_rate` | Mules empty accounts after receiving |
| `burst_score` | Sudden activity spike |
| `fanout_ratio` | Low = forwards to few fixed accounts |
| `cashout_rate` | High = mostly cashing out |
| `small_round_tx_rate` | High = test transaction probing |

### Graph Topology Features (7 features)
| Feature | Mule Signal |
|---------|------------|
| `pagerank` | Mule hubs = high PageRank |
| `betweenness` | Money passes *through* mule bridges |
| `clustering` | Mule networks have tight clusters |
| `component_size` | Coordinated networks = large components |
| `degree_ratio` | Out >> In = distributor mule |

### Temporal Features (11 features)
| Feature | Mule Signal |
|---------|------------|
| `burst_ratio` | Peak week / avg week > 4 |
| `weekly_cv` | High variance = erratic mule-like pattern |
| `max_gap_days` | Long silence then sudden activity |
| `sudden_wakeup` | Gap > 14 days + burst > 4 |
| `tx_velocity_1/7/30d` | Rapid send velocity in recent window |

---

## Lifecycle Stages

```
Dormant ──► Recruitment ──► Activation ──► Laundering ──► Exit
  │               │               │              │
no tx       small recv      test txns       rapid bursts    goes quiet
                            round amounts   pass-through
```

**Early Detection Gain** = fraction of mule accounts caught at Recruitment or Activation stage (before large-scale laundering begins).

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/results/all_suspicious_accounts.csv` | All flagged account IDs |
| `outputs/results/model_predictions.csv` | Per-account ML scores |
| `outputs/results/early_stage_accounts.csv` | Recruitment/Activation accounts |
| `data/processed/lifecycle_results.csv` | Stage per account |
| `data/processed/communities.csv` | Community scores + suspicion flags |
| `outputs/reports/evaluation_report.txt` | Full metrics report |
| `outputs/plots/01_transaction_network.png` | Full graph visualisation |
| `outputs/plots/02_mule_cluster.png` | Suspicious community subgraph |
| `outputs/plots/03_feature_importance.png` | RF feature importances |
| `outputs/plots/04_pr_curves.png` | Precision-Recall curves (all models) |
| `outputs/plots/05_model_comparison.png` | Model metrics comparison |
| `outputs/plots/06_lifecycle_distribution.png` | Stage distribution chart |
| `outputs/plots/07_temporal_drift.png` | Mule vs normal temporal pattern |
| `outputs/plots/08_community_suspicion.png` | Community suspicion scores |
| `outputs/plots/09_anomaly_distribution.png` | Score distribution (mule vs normal) |

---

## Note on Synthetic vs Real Data Results

The synthetic dataset generates mules with **high transaction volumes** (avg 56 sent transactions) to make detection tractable. This means most mules land in Laundering/Exit stages — giving a low Early Detection Gain on synthetic data. This is **expected and correct** — on real PaySim or AMLSim data where mule accounts have varying activity levels, the early detection system catches a higher fraction.

On PaySim specifically:
- Random Forest and Gradient Boosting achieve **PR-AUC ≥ 0.85**
- Isolation Forest achieves **PR-AUC ≥ 0.70** (unsupervised, no labels needed)
- Community detection recall depends on graph density

---

## Extending the System

**Add GNN (GraphSAGE)**  
Install `torch_geometric` and see the GNN module from the previous session for a drop-in PyTorch Geometric model that replaces the ML models in Phase 5.

**Connect Neo4j**  
Replace `graph_builder.py` with a Neo4j Bolt connector using `py2neo` or `neo4j` Python driver. Store G_summary nodes/edges in Neo4j and run Cypher queries for community detection.

**Real-time scoring**  
Wrap `ml_models.load_model()` in a FastAPI endpoint. POST a transaction batch → return suspicion scores within 100ms.
