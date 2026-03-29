"""
src/data/load_data.py
---------------------
Functions to load raw and processed CSV data for the corporate bond default
prediction project.
"""
import pandas as pd


def load_raw(path: str) -> pd.DataFrame:
    """
    Load the original whitespace-separated raw dataset.

    Parameters
    ----------
    path : str
        Path to the raw CSV file (e.g. 'data/raw/dataset_csv.csv').

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    return df


def load_processed(train_path: str, test_path: str):
    """
    Load the pre-processed feature files saved by notebook 02.

    Parameters
    ----------
    train_path : str
        Path to 'data/processed/features_train.csv'.
    test_path : str
        Path to 'data/processed/features_test.csv'.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    return train_df, test_df
