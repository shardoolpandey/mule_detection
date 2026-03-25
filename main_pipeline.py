"""
main_pipeline.py
=================
Master orchestration script for the Money Mule Detection System.

Runs all phases in sequence:
  Phase 1  — Data ingestion (load or generate dataset)
  Phase 2  — Graph construction
  Phase 3  — Feature engineering
  Phase 4  — Community detection
  Phase 5  — Machine learning models (Isolation Forest, RF, GB)
  Phase 6  — Lifecycle stage classification
  Phase 7  — Evaluation & reporting
  Phase 8  — Visualisations

Usage
-----
  # Use synthetic data (default — no downloads required):
  python main_pipeline.py

  # Use your PaySim CSV:
  python main_pipeline.py --source paysim --path data/raw/paysim.csv

  # Sample 50K rows from PaySim:
  python main_pipeline.py --source paysim --sample 50000
"""

import argparse
import time
import sys
import os
import pandas as pd
import numpy as np

# Make src importable from project root
sys.path.insert(0, os.path.dirname(__file__))

from config.config import *
from src.ingestion.data_loader       import load_synthetic, load_paysim, load_amlsim
from src.graph.graph_builder         import build_transaction_graph, graph_summary
from src.features.feature_engineering import build_feature_matrix, get_feature_columns
from src.community.community_detector import (
    detect_communities, score_communities, get_suspicious_accounts
)
from src.models.ml_models            import (
    train_isolation_forest, train_random_forest,
    train_gradient_boosting, build_ensemble_scores, save_model
)
from src.lifecycle.lifecycle_detector import (
    detect_lifecycle_stages, get_early_stage_accounts
)
from src.evaluation.evaluator        import (
    evaluate_model, evaluate_early_detection,
    evaluate_community_detection, compile_report
)
from src.visualization.visualizer    import (
    plot_transaction_network, plot_mule_cluster,
    plot_feature_importance, plot_pr_curves,
    plot_model_comparison, plot_lifecycle_distribution,
    plot_temporal_drift, plot_community_suspicion,
    plot_anomaly_scores,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="synthetic",
                   choices=["synthetic", "paysim", "amlsim"],
                   help="Data source")
    p.add_argument("--path",   default=None, help="Path to CSV file")
    p.add_argument("--sample", type=int, default=None,
                   help="Sample N rows (for large datasets)")
    p.add_argument("--regen",  action="store_true",
                   help="Regenerate synthetic dataset")
    p.add_argument("--skip-viz", action="store_true",
                   help="Skip visualisation (faster)")
    return p.parse_args()


def banner(text):
    print(f"\n{'═'*65}")
    print(f"  {text}")
    print(f"{'═'*65}")


def run_pipeline(args=None):
    if args is None:
        args = parse_args()

    t_total = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — DATA INGESTION
    # ══════════════════════════════════════════════════════════════════════════
    banner("Phase 1 — Data Ingestion")
    t0 = time.time()

    if args.source == "paysim":
        df = load_paysim(path=args.path, sample_n=args.sample)
    elif args.source == "amlsim":
        df = load_amlsim(path=args.path, sample_n=args.sample)
    else:
        df = load_synthetic(regenerate=args.regen)
        if args.sample and len(df) > args.sample:
            df = df.sample(args.sample, random_state=RANDOM_STATE).reset_index(drop=True)

    # Save canonical dataset
    df.to_csv(DATA_PROCESSED / "canonical_transactions.csv", index=False)
    print(f"  Phase 1 done in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — GRAPH CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════════
    banner("Phase 2 — Graph Construction")
    t0 = time.time()

    G_summary, G_full = build_transaction_graph(df)

    stats = graph_summary(G_summary)
    print("\nGraph statistics:")
    for k, v in stats.items():
        print(f"  {k:<22}: {v}")

    print(f"  Phase 2 done in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════════════════════════
    banner("Phase 3 — Feature Engineering")
    t0 = time.time()

    feature_matrix = build_feature_matrix(df, G_summary)
    feat_cols      = get_feature_columns(feature_matrix)
    print(f"  Feature matrix : {feature_matrix.shape}")
    print(f"  Feature columns: {len(feat_cols)}")
    print(f"  Phase 3 done in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 4 — COMMUNITY DETECTION
    # ══════════════════════════════════════════════════════════════════════════
    banner("Phase 4 — Community Detection")
    t0 = time.time()

    partition = detect_communities(G_summary, method="louvain")
    comm_df   = score_communities(partition, feature_matrix, G_summary)
    susp_accs = get_suspicious_accounts(partition, comm_df)

    # Attach community ID to feature matrix
    feature_matrix["community_id"] = feature_matrix["account"].map(partition)

    # Save community results
    comm_save = comm_df.drop(columns=["accounts"], errors="ignore")
    comm_save.to_csv(DATA_PROCESSED / "communities.csv", index=False)
    pd.Series(susp_accs, name="account").to_csv(
        OUTPUTS_RESULTS / "suspicious_from_communities.csv", index=False
    )
    print(f"  Suspicious accounts from communities: {len(susp_accs):,}")
    print(f"  Phase 4 done in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 5 — MACHINE LEARNING MODELS
    # ══════════════════════════════════════════════════════════════════════════
    banner("Phase 5 — Machine Learning Models")
    t0 = time.time()

    has_labels = feature_matrix["is_fraud"].sum() > 0

    # 5a. Isolation Forest (always runs)
    iso_model, iso_scaler, iso_scores, iso_preds = train_isolation_forest(feature_matrix)
    save_model({"model": iso_model, "scaler": iso_scaler}, "isolation_forest")

    model_results = {}

    # Isolation Forest eval
    if has_labels:
        accs_indexed = feature_matrix.set_index("account")
        common_iso   = iso_scores.index.intersection(accs_indexed.index)
        iso_res = evaluate_model(
            accs_indexed.loc[common_iso, "is_fraud"].values,
            iso_preds.loc[common_iso].values,
            iso_scores.loc[common_iso].values,
            "IsolationForest",
        )
        model_results["IsolationForest"] = {
            "y_test":      accs_indexed.loc[common_iso, "is_fraud"].values,
            "y_prob_test": iso_scores.loc[common_iso].values,
            "y_pred_test": iso_preds.loc[common_iso].values,
        }
    else:
        iso_res = {"Model": "IsolationForest"}

    # 5b. Random Forest
    rf_model = rf_scaler = rf_res_dict = fi = None
    rf_results_eval = {"Model": "RandomForest"}

    if has_labels:
        rf_model, rf_scaler, rf_res_dict = train_random_forest(feature_matrix)
        save_model({"model": rf_model, "scaler": rf_scaler, "feat_cols": feat_cols}, "random_forest")
        fi = rf_res_dict["feat_importance"]
        rf_results_eval = evaluate_model(
            rf_res_dict["y_test"],
            rf_res_dict["y_pred_test"],
            rf_res_dict["y_prob_test"],
            "RandomForest",
        )
        model_results["RandomForest"] = {
            "y_test":      rf_res_dict["y_test"],
            "y_prob_test": rf_res_dict["y_prob_test"],
            "y_pred_test": rf_res_dict["y_pred_test"],
        }

    # 5c. Gradient Boosting
    gb_model = gb_scaler = gb_res_dict = None
    gb_results_eval = {"Model": "GradientBoosting"}

    if has_labels:
        gb_model, gb_scaler, gb_res_dict = train_gradient_boosting(feature_matrix)
        save_model({"model": gb_model, "scaler": gb_scaler, "feat_cols": feat_cols}, "gradient_boosting")
        gb_results_eval = evaluate_model(
            gb_res_dict["y_test"],
            gb_res_dict["y_pred_test"],
            gb_res_dict["y_prob_test"],
            "GradientBoosting",
        )
        model_results["GradientBoosting"] = {
            "y_test":      gb_res_dict["y_test"],
            "y_prob_test": gb_res_dict["y_prob_test"],
            "y_pred_test": gb_res_dict["y_pred_test"],
        }

    # 5d. Ensemble score
    if rf_res_dict and gb_res_dict:
        ensemble_scores = build_ensemble_scores(
            iso_scores,
            rf_res_dict["proba"],
            gb_res_dict["proba"],
        )
    else:
        ensemble_scores = iso_scores.rename("ensemble_score")

    # Build final suspicious account list (ensemble threshold)
    threshold        = float(np.percentile(ensemble_scores.values, 95))
    suspicious_ml    = ensemble_scores[ensemble_scores > threshold].index.tolist()
    suspicious_all   = list(set(susp_accs) | set(suspicious_ml))

    # Save
    result_df = feature_matrix[["account", "is_fraud"]].copy()
    result_df["iso_score"]      = iso_scores.reindex(result_df["account"]).values
    result_df["iso_pred"]       = iso_preds.reindex(result_df["account"]).values
    if rf_res_dict:
        result_df["rf_proba"]   = rf_res_dict["proba"].reindex(result_df["account"]).values
        result_df["rf_pred"]    = rf_res_dict["preds"].reindex(result_df["account"]).values
    if gb_res_dict:
        result_df["gb_proba"]   = gb_res_dict["proba"].reindex(result_df["account"]).values
        result_df["gb_pred"]    = gb_res_dict["preds"].reindex(result_df["account"]).values
    result_df["ensemble_score"] = ensemble_scores.reindex(result_df["account"]).values
    result_df["ml_suspicious"]  = result_df["ensemble_score"] > threshold
    result_df.to_csv(OUTPUTS_RESULTS / "model_predictions.csv", index=False)

    pd.Series(suspicious_all, name="account").to_csv(
        OUTPUTS_RESULTS / "all_suspicious_accounts.csv", index=False
    )
    print(f"\n  Total suspicious accounts (ensemble + community): {len(suspicious_all):,}")
    print(f"  Phase 5 done in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 6 — LIFECYCLE DETECTION
    # ══════════════════════════════════════════════════════════════════════════
    banner("Phase 6 — Lifecycle Stage Detection")
    t0 = time.time()

    lifecycle_df    = detect_lifecycle_stages(feature_matrix)
    early_stage_df  = get_early_stage_accounts(lifecycle_df)
    early_stage_df[["account","lifecycle_stage","risk_level"]].to_csv(
        OUTPUTS_RESULTS / "early_stage_accounts.csv", index=False
    )
    print(f"  Phase 6 done in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 7 — EVALUATION & REPORTING
    # ══════════════════════════════════════════════════════════════════════════
    banner("Phase 7 — Evaluation & Reporting")
    t0 = time.time()

    model_metrics = [iso_res, rf_results_eval, gb_results_eval]
    early_eval    = evaluate_early_detection(lifecycle_df)
    comm_eval     = evaluate_community_detection(partition, feature_matrix, comm_df)

    comparison_df = pd.DataFrame([
        m for m in model_metrics if "Precision" in m
    ])

    compile_report(model_metrics, early_eval, comm_eval, suspicious_all)
    print(f"  Phase 7 done in {time.time()-t0:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 8 — VISUALISATIONS
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_viz:
        banner("Phase 8 — Visualisations")
        t0 = time.time()

        plot_transaction_network(G_summary)

        # Pick largest suspicious community for cluster plot
        if len(comm_df) > 0:
            top_comm     = comm_df.iloc[0]
            cluster_nodes = top_comm["accounts"][:50]   # limit for clarity
            plot_mule_cluster(G_summary, cluster_nodes)

        if fi is not None:
            plot_feature_importance(fi)

        if model_results:
            plot_pr_curves(model_results)

        if len(comparison_df) > 0:
            plot_model_comparison(comparison_df)

        plot_lifecycle_distribution(lifecycle_df)

        # Temporal drift: pick one mule + one normal account
        if has_labels:
            mule_acc   = feature_matrix[feature_matrix["is_fraud"] == 1]["account"].iloc[0]
            normal_acc = feature_matrix[feature_matrix["is_fraud"] == 0]["account"].iloc[0]
            plot_temporal_drift(df, mule_acc, normal_acc)

        plot_community_suspicion(comm_df)

        iso_lab = feature_matrix["is_fraud"] if has_labels else None
        plot_anomaly_scores(iso_scores, labels=iso_lab, threshold=threshold)

        print(f"  Phase 8 done in {time.time()-t0:.1f}s")
    else:
        print("\nVisualisations skipped (--skip-viz flag set)")

    # ── Final summary ─────────────────────────────────────────────────────────
    banner("Pipeline Complete")
    print(f"  Total time           : {time.time()-t_total:.1f}s")
    print(f"  Suspicious accounts  : {len(suspicious_all):,}")
    print(f"  Early-stage flagged  : {len(early_stage_df):,}")
    print(f"\nOutputs written to:")
    print(f"  {DATA_PROCESSED}")
    print(f"  {OUTPUTS_RESULTS}")
    print(f"  {OUTPUTS_PLOTS}")
    print(f"  {OUTPUTS_REPORTS}")

    return {
        "df":               df,
        "G":                G_summary,
        "feature_matrix":   feature_matrix,
        "communities":      comm_df,
        "lifecycle_df":     lifecycle_df,
        "suspicious_all":   suspicious_all,
        "ensemble_scores":  ensemble_scores,
    }


if __name__ == "__main__":
    run_pipeline()
