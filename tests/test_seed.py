import numpy as np

from kernel.utils.seed import set_seed


def test_set_seed_makes_numpy_deterministic():
    set_seed(42)
    a = np.random.randn(5)

    set_seed(42)
    b = np.random.randn(5)

    assert np.allclose(a, b)


def test_set_seed_returns_seed():
    out = set_seed(123)
    assert out == 123