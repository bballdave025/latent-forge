import pandas as pd


def item_mean_fit(ratings: pd.DataFrame, item_col="ASIN", rating_col="Rating"):
    """Course-style rank-1 baseline: each item's training mean."""
    return ratings.groupby(item_col)[rating_col].mean()


def item_mean_predict(rows: pd.DataFrame, item_means, fallback=None, item_col="ASIN"):
    if fallback is None:
        fallback = float(item_means.mean())
    return rows[item_col].map(item_means).fillna(fallback).astype(float)
