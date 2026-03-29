"""
src/features/build_features.py
--------------------------------
Feature engineering utilities:
  - Column constants
  - Correlated-feature removal / renaming
  - Sliding-window sequence construction for LSTM / GRU
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns identified as highly correlated during EDA (notebook 01) and
# therefore excluded from model feature matrices.
COLS_TO_DROP: list[str] = ["x6", "x7", "x19", "x24", "x25"]

# Metadata columns that are kept in the processed files as labels/identifiers
# but MUST NOT be fed as model inputs.
META_COLS: list[str] = ["obs_id", "company_id", "time", "year", "default"]


def engineer_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols_to_drop: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply feature engineering steps shared across all three models:
      1. Rename ``class`` → ``company_id``
      2. Drop highly correlated columns defined in ``COLS_TO_DROP``

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
        DataFrames produced by :func:`src.data.preprocessing.split_train_test`.
    cols_to_drop : list[str], optional
        Override the default ``COLS_TO_DROP`` list.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) after transformation, unchanged index.
    """
    if cols_to_drop is None:
        cols_to_drop = COLS_TO_DROP

    train_df = train_df.rename(columns={"class": "company_id"})
    test_df  = test_df.rename(columns={"class": "company_id"})

    train_df = train_df.drop(columns=cols_to_drop, errors="ignore")
    test_df  = test_df.drop(columns=cols_to_drop, errors="ignore")

    return train_df, test_df


def create_sequences(
    df: pd.DataFrame,
    sequence_length: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences strictly within each company's time series.

    Groups by ``company_id``, sorts each group by ``time``, then applies a
    sliding window of width ``sequence_length``. The target label is taken
    from the **last** time step of each window.

    Columns in ``META_COLS`` (``obs_id``, ``company_id``, ``time``, ``year``,
    ``default``) are excluded from the feature array ``X``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``company_id``, ``time``, ``default``, and feature columns.
    sequence_length : int
        Number of consecutive time steps per sequence window.

    Returns
    -------
    X : np.ndarray  shape (n_sequences, sequence_length, n_features)
    y : np.ndarray  shape (n_sequences,)
    """
    features = [c for c in df.columns if c not in META_COLS]

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for _company_id, group in df.groupby("company_id", sort=False):
        group = group.sort_values("time")

        feat   = group[features].values
        target = group["default"].values
        n      = len(feat)

        if n < sequence_length:
            continue  # not enough time steps — skip this company

        for i in range(n - sequence_length + 1):
            X_list.append(feat[i : i + sequence_length])
            y_list.append(target[i + sequence_length - 1])

    return np.array(X_list), np.array(y_list)
