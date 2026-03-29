"""
src/models/train_catboost.py
-----------------------------
CatBoost training utilities for the corporate bond default prediction project.
"""
from __future__ import annotations

import pandas as pd
from catboost import CatBoostClassifier

# ---------------------------------------------------------------------------
# Default hyperparameters (as tuned in notebook 03)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: dict = {
    "iterations"           : 1000,
    "learning_rate"        : 0.01,
    "depth"                : 5,
    "l2_leaf_reg"          : 3,
    "random_strength"      : 1,
    "bagging_temperature"  : 1,
    "auto_class_weights"   : "Balanced",
    "early_stopping_rounds": 50,
    "eval_metric"          : "AUC",
    "random_seed"          : 42,
    "verbose"              : 100,
    "task_type"            : "CPU",
    "thread_count"         : -1,
}


def train_catboost(
    X_train: pd.DataFrame,
    y_train,
    X_val: pd.DataFrame,
    y_val,
    params: dict | None = None,
) -> CatBoostClassifier:
    """
    Train a CatBoostClassifier with early stopping on the validation set.

    Parameters
    ----------
    X_train, y_train : Training features and labels.
    X_val,   y_val   : Validation features and labels (used for early stopping).
    params           : CatBoost hyperparameter dict. Defaults to ``DEFAULT_PARAMS``.

    Returns
    -------
    CatBoostClassifier
        Fitted model (best iteration selected automatically).
    """
    if params is None:
        params = DEFAULT_PARAMS

    model = CatBoostClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        plot=False,
    )
    return model
