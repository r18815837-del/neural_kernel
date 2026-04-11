import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn import LayerNorm


def test_layernorm_output_shape():
    x = Tensor(np.random.randn(2, 5), requires_grad=True)
    ln = LayerNorm(5)

    y = ln(x)

    assert y.data.shape == (2, 5)


def test_layernorm_forward_basic():
    x_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    x = Tensor(x_data, requires_grad=True)
    ln = LayerNorm(3, affine=False)

    y = ln(x)

    expected_mean = x_data.mean(axis=-1, keepdims=True)
    expected_var = x_data.var(axis=-1, keepdims=True)
    expected = (x_data - expected_mean) / np.sqrt(expected_var + 1e-5)

    assert np.allclose(y.data, expected)


def test_layernorm_affine_shapes():
    ln = LayerNorm(4, affine=True)

    assert ln.weight is not None
    assert ln.bias is not None
    assert ln.weight.data.shape == (4,)
    assert ln.bias.data.shape == (4,)


def test_layernorm_backward_runs():
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    ln = LayerNorm(4)

    y = ln(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.data.shape
    assert ln.weight.grad is not None
    assert ln.bias.grad is not None