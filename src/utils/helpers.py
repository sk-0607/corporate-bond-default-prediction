"""
src/utils/helpers.py
---------------------
Shared evaluation and threshold-optimisation utilities used across all model
notebooks (CatBoost, LSTM, GRU).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    y_true,
    y_pred,
    y_pred_proba,
    dataset_name: str = "Dataset",
) -> dict:
    """
    Compute and print AUC-ROC, Precision, Recall, and F1 for a dataset split.

    Parameters
    ----------
    y_true       : array-like of int   Ground-truth binary labels.
    y_pred       : array-like of int   Hard predictions.
    y_pred_proba : array-like of float Predicted probabilities for class 1.
    dataset_name : str                 Label printed in the output header.

    Returns
    -------
    dict with keys ``auc``, ``precision``, ``recall``, ``f1``.
    """
    auc       = roc_auc_score(y_true, y_pred_proba)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{dataset_name} Metrics:")
    print("=" * 50)
    print(f"AUC-ROC   : {auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("=" * 50)

    return {"auc": auc, "precision": precision, "recall": recall, "f1": f1}


def metrics_at_threshold(y_true, y_proba, t: float) -> dict:
    """
    Compute classification metrics for a specific probability threshold.

    Parameters
    ----------
    y_true  : array-like of int   Ground-truth binary labels.
    y_proba : array-like of float Predicted probabilities.
    t       : float               Decision threshold in [0, 1].

    Returns
    -------
    dict with keys ``threshold``, ``precision``, ``recall``, ``f1``, ``fn``, ``fp``.
    """
    y_pred = (np.asarray(y_proba) >= t).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold" : t,
        "precision" : precision_score(y_true, y_pred, zero_division=0),
        "recall"    : recall_score(y_true, y_pred, zero_division=0),
        "f1"        : f1_score(y_true, y_pred, zero_division=0),
        "fn"        : int(fn),
        "fp"        : int(fp),
    }


def find_threshold_f1(y_true, y_proba) -> float:
    """
    Find the probability threshold that maximises F1 on the given set.

    Parameters
    ----------
    y_true  : array-like of int
    y_proba : array-like of float

    Returns
    -------
    float  Optimal threshold.
    """
    _, _, thresholds = precision_recall_curve(y_true, y_proba)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f = f1_score(y_true, (np.asarray(y_proba) >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t


def find_threshold_cost(
    y_true,
    y_proba,
    fn_cost: float = 10,
    fp_cost: float = 1,
) -> tuple[float, float]:
    """
    Find the threshold that minimises total weighted cost.

    In bond default prediction a missed default (FN) is far more costly than
    a false alarm (FP). The default assumption ``fn_cost=10`` means one missed
    default is as expensive as 10 false alarms.

    Parameters
    ----------
    y_true   : array-like of int
    y_proba  : array-like of float
    fn_cost  : relative cost of a false negative
    fp_cost  : relative cost of a false positive

    Returns
    -------
    tuple[float, float]  (best_threshold, minimum_cost)
    """
    _, _, thresholds = precision_recall_curve(y_true, y_proba)
    best_t, best_cost = 0.5, float("inf")
    for t in thresholds:
        y_pred = (np.asarray(y_proba) >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cost = fn * fn_cost + fp * fp_cost
        if cost < best_cost:
            best_cost, best_t = float(cost), float(t)
    return best_t, best_cost
