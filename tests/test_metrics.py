import numpy as np
from latent_forge.metrics import masked_mse


def test_masked_mse_ignores_unobserved():
    y = np.array([[5., 0.]])
    p = np.array([[4., 100.]])
    m = np.array([[1, 0]])
    assert masked_mse(p, y, m) == 1.0
