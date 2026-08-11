import numpy as np


def masked_mse(prediction, ratings, mask):
    """MSE over observed entries only."""
    observed = np.asarray(mask, dtype=bool)
    if not observed.any():
        raise ValueError("mask contains no observed entries")
    error = np.asarray(ratings)[observed] - np.asarray(prediction)[observed]
    return float(np.mean(error ** 2))


def masked_rmse(prediction, ratings, mask):
    return float(np.sqrt(masked_mse(prediction, ratings, mask)))
