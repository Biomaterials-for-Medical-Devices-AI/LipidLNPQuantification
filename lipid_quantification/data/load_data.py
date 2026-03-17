from pathlib import Path
from typing import List, Tuple

import pandas as pd


def load_train_df(
    path: Path,
    use_feats: list | None = None,
    remove_feats: list | None = None,
    target_vars: list | None = None,
    index_col: int = 0,
) -> pd.DataFrame:
    data = pd.read_csv(path, index_col=index_col)

    # Remove unwanted features safely
    if remove_feats is not None:
        data = data.drop(columns=remove_feats, errors="ignore")

    # Drop rows with NaNs
    data = data.dropna()

    # Select only desired features
    if use_feats is not None:
        missing = [f for f in use_feats if f not in data.columns]
        if missing:
            raise ValueError(f"Features not found in DataFrame: {missing}")
        data = data[target_vars + use_feats]

    return data


def load_test_df_aligned(
    path: Path, train_columns: List, index_col: int = 0
) -> pd.DataFrame:
    """Exactly your working line: test = test[data.columns]."""
    test = pd.read_csv(path, index_col=index_col)
    return test[train_columns]


def split_xy(df: pd.DataFrame, n_targets=4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = df.iloc[:, :n_targets]
    X = df.iloc[:, n_targets:]
    return X, y
