"""
src/data/preprocessing.py
--------------------------
Data preparation utilities: train/test splitting, metadata column removal,
and class-weight computation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


def split_train_test(df: pd.DataFrame):
    """
    Split the raw DataFrame into train and test sets using the
    ``training_set`` / ``testing_set`` flags present in the raw dataset.
    Resets index and drops the split-indicator columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame that contains ``training_set`` and ``testing_set`` columns.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df) — both without the split-indicator columns.
    """
    train_df = (
        df[df["training_set"] == 1]
        .drop(columns=["training_set", "testing_set"])
        .reset_index(drop=True)
    )
    test_df = (
        df[df["testing_set"] == 1]
        .drop(columns=["training_set", "testing_set"])
        .reset_index(drop=True)
    )
    return train_df, test_df


def drop_metadata_cols(
    df: pd.DataFrame,
    meta_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Drop metadata / non-feature columns from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    meta_cols : list[str], optional
        Columns to drop. Defaults to ``['obs_id', 'time', 'year']``.

    Returns
    -------
    pd.DataFrame
        DataFrame with the specified columns removed (missing ones are ignored).
    """
    if meta_cols is None:
        meta_cols = ["obs_id", "time", "year"]
    return df.drop(columns=meta_cols, errors="ignore")


def compute_class_weights(y: np.ndarray) -> dict:
    """
    Compute balanced class weights suitable for Keras ``class_weight``
    or scikit-learn ``class_weight`` parameters.

    Parameters
    ----------
    y : array-like of int
        Binary target labels (0 / 1).

    Returns
    -------
    dict
        ``{0: weight_for_0, 1: weight_for_1}``
    """
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=np.asarray(y))
    return {0: weights[0], 1: weights[1]}
