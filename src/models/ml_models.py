"""
src/models/ml_models.py
========================
Trains three complementary ML models for mule account detection.

Model roles
-----------
  IsolationForest  : Unsupervised anomaly detection.
                     Flags statistically unusual accounts without labels.
                     Use when labels are sparse or unavailable.

  RandomForest     : Supervised ensemble classifier.
                     High interpretability (feature importances).
                     Robust to class imbalance via class_weight='balanced'.

  GradientBoosting : Supervised sequential ensemble.
                     Typically highest precision on structured tabular data.
                     More sensitive to hyperparameter tuning than RF.

All three are combined into an ensemble via soft-voting on probability scores.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    IsolationForest,
    VotingClassifier,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    classification_report, precision_recall_curve,
    average_precision_score, roc_auc_score,
    confusion_matrix, f1_score
)
from sklearn.calibration import CalibratedClassifierCV
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import *

warnings.filterwarnings("ignore")


# ── Feature column resolver ───────────────────────────────────────────────────
def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    drop = {"account", "is_fraud", "source", "tx_id"}
    return [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ISOLATION FOREST (unsupervised)
# ═══════════════════════════════════════════════════════════════════════════════

def train_isolation_forest(
    feature_matrix: pd.DataFrame,
    contamination: float = None,
) -> tuple:
    """
    Train Isolation Forest on all accounts (no labels required).

    Returns (model, scaler, anomaly_scores_series, predictions_series)
    """
    print("\n--- Training Isolation Forest ---")
    feat_cols = _get_feature_cols(feature_matrix)
    X = feature_matrix[feat_cols].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use known fraud rate as contamination if available
    if contamination is None:
        if "is_fraud" in feature_matrix.columns and feature_matrix["is_fraud"].sum() > 0:
            contamination = float(np.clip(feature_matrix["is_fraud"].mean(), 0.001, 0.1))
        else:
            contamination = "auto"

    iso = IsolationForest(
        n_estimators=ISO_N_ESTIMATORS,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    raw_scores  = iso.score_samples(X_scaled)   # lower = more anomalous
    anom_scores = -raw_scores                    # flip: higher = more suspicious
    iso_preds   = (iso.predict(X_scaled) == -1).astype(int)

    accs = feature_matrix["account"].values if "account" in feature_matrix.columns \
           else feature_matrix.index

    scores_s = pd.Series(anom_scores, index=accs, name="iso_score")
    preds_s  = pd.Series(iso_preds,   index=accs, name="iso_pred")

    if "is_fraud" in feature_matrix.columns:
        y_true = feature_matrix["is_fraud"].values
        if y_true.sum() > 0:
            pr_auc = average_precision_score(y_true, anom_scores)
            roc    = roc_auc_score(y_true, anom_scores)
            print(f"  PR-AUC  : {pr_auc:.4f}")
            print(f"  ROC-AUC : {roc:.4f}")
            print(classification_report(y_true, iso_preds,
                                        target_names=["Normal", "Mule"], digits=4))

    print(f"  Flagged anomalies: {iso_preds.sum():,} / {len(iso_preds):,}")
    return iso, scaler, scores_s, preds_s


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RANDOM FOREST (supervised)
# ═══════════════════════════════════════════════════════════════════════════════

def train_random_forest(
    feature_matrix: pd.DataFrame,
) -> tuple:
    """
    Train Random Forest classifier. Returns (model, scaler, results_dict).
    """
    print("\n--- Training Random Forest ---")
    feat_cols = _get_feature_cols(feature_matrix)
    X = feature_matrix[feat_cols].values
    y = feature_matrix["is_fraud"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight=RF_CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)

    y_prob = rf.predict_proba(X_te)[:, 1]
    y_pred = rf.predict(X_te)

    pr_auc  = average_precision_score(y_te, y_prob)
    roc_auc = roc_auc_score(y_te, y_prob)

    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(classification_report(y_te, y_pred,
                                target_names=["Normal", "Mule"], digits=4))

    # Feature importances
    fi = pd.Series(rf.feature_importances_, index=feat_cols)
    fi = fi.sort_values(ascending=False)
    print(f"\n  Top 10 features:")
    print(fi.head(10).to_string())

    # Full-dataset probabilities
    all_proba = rf.predict_proba(X_scaled)[:, 1]
    all_preds = rf.predict(X_scaled)
    accs      = feature_matrix["account"].values

    results = {
        "pr_auc":     pr_auc,
        "roc_auc":    roc_auc,
        "proba":      pd.Series(all_proba, index=accs, name="rf_proba"),
        "preds":      pd.Series(all_preds, index=accs, name="rf_pred"),
        "feat_importance": fi,
        "y_test":     y_te,
        "y_prob_test": y_prob,
        "y_pred_test": y_pred,
    }
    return rf, scaler, results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GRADIENT BOOSTING (supervised)
# ═══════════════════════════════════════════════════════════════════════════════

def train_gradient_boosting(
    feature_matrix: pd.DataFrame,
) -> tuple:
    """
    Train Gradient Boosting classifier. Returns (model, scaler, results_dict).
    """
    print("\n--- Training Gradient Boosting ---")
    feat_cols = _get_feature_cols(feature_matrix)
    X = feature_matrix[feat_cols].values
    y = feature_matrix["is_fraud"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )

    # Class weight via sample_weight
    n_neg = (y_tr == 0).sum()
    n_pos = (y_tr == 1).sum()
    sample_weight = np.where(y_tr == 1, n_neg / max(n_pos, 1), 1.0)

    gb = GradientBoostingClassifier(
        n_estimators=GB_N_ESTIMATORS,
        learning_rate=GB_LEARNING_RATE,
        max_depth=GB_MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    gb.fit(X_tr, y_tr, sample_weight=sample_weight)

    y_prob = gb.predict_proba(X_te)[:, 1]
    y_pred = gb.predict(X_te)

    pr_auc  = average_precision_score(y_te, y_prob)
    roc_auc = roc_auc_score(y_te, y_prob)

    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(classification_report(y_te, y_pred,
                                target_names=["Normal", "Mule"], digits=4))

    all_proba = gb.predict_proba(X_scaled)[:, 1]
    all_preds = gb.predict(X_scaled)
    accs      = feature_matrix["account"].values

    results = {
        "pr_auc":      pr_auc,
        "roc_auc":     roc_auc,
        "proba":       pd.Series(all_proba, index=accs, name="gb_proba"),
        "preds":       pd.Series(all_preds, index=accs, name="gb_pred"),
        "y_test":      y_te,
        "y_prob_test": y_prob,
        "y_pred_test": y_pred,
    }
    return gb, scaler, results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ENSEMBLE (soft voting)
# ═══════════════════════════════════════════════════════════════════════════════

def build_ensemble_scores(
    iso_scores: pd.Series,
    rf_proba:   pd.Series,
    gb_proba:   pd.Series,
    weights:    tuple = (0.20, 0.45, 0.35),
) -> pd.Series:
    """
    Combine model scores into a single ensemble suspicion score.
    Normalise each model's scores to [0, 1] before weighting.
    """
    def _norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    common = iso_scores.index.intersection(rf_proba.index).intersection(gb_proba.index)
    score  = (
        weights[0] * _norm(iso_scores.loc[common]) +
        weights[1] * rf_proba.loc[common] +
        weights[2] * gb_proba.loc[common]
    )
    return score.rename("ensemble_score")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════════════════

def save_model(obj, name: str):
    path = OUTPUTS_MODELS / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved model → {path}")


def load_model(name: str):
    path = OUTPUTS_MODELS / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
