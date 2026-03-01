from pathlib import Path
from typing import List, Tuple

import pandas as pd


def load_train_df(path: Path, index_col: int = 0) -> pd.DataFrame:
    return pd.read_csv(path, index_col=index_col)


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
