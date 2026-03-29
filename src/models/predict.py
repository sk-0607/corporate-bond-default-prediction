"""
src/models/predict.py
----------------------
Thin prediction helpers that provide a uniform interface over both
CatBoost and Keras (LSTM/GRU) models.
"""
from __future__ import annotations

import numpy as np


def predict_proba(model, X) -> np.ndarray:
    """
    Return a 1-D array of predicted probabilities for class 1.

    Works with:
    - ``CatBoostClassifier``  (uses ``.predict_proba(X)[:, 1]``)
    - Keras ``Sequential``    (uses ``.predict(X).flatten()``)

    Parameters
    ----------
    model : fitted model object
    X     : Feature array / DataFrame accepted by the model.

    Returns
    -------
    np.ndarray of shape (n_samples,)
    """
    if hasattr(model, "predict_proba"):
        # scikit-learn / CatBoost API
        return np.asarray(model.predict_proba(X))[:, 1]
    else:
        # Keras API
        return model.predict(X, verbose=0).flatten()


def apply_threshold(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert probability scores to binary class labels.

    Parameters
    ----------
    proba     : 1-D array of probabilities in [0, 1].
    threshold : Decision boundary (default 0.5).

    Returns
    -------
    np.ndarray of int (0 or 1).
    """
    return (np.asarray(proba) >= threshold).astype(int)
