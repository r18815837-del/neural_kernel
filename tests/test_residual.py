import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn import ResidualBlock


def test_residual_block_output_shape():
    x = Tensor(np.random.randn(2, 8, 16, 16), requires_grad=True)
    block = ResidualBlock(8)

    y = block(x)

    assert y.data.shape == x.data.shape


def test_residual_block_backward_runs():
    x = Tensor(np.random.randn(2, 8, 16, 16), requires_grad=True)
    block = ResidualBlock(8)

    y = block(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.data.shape