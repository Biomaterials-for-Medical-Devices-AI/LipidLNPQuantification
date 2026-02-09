import pandas as pd


def load_train_df(path, index_col=0):
    return pd.read_csv(path, index_col=index_col)


def load_test_df_aligned(path, train_columns, index_col=0):
    """Exactly your working line: test = test[data.columns]."""
    test = pd.read_csv(path, index_col=index_col)
    return test[train_columns]


def split_xy(df, n_targets=4):
    y = df.iloc[:, :n_targets]
    X = df.iloc[:, n_targets:]
    return X, y
