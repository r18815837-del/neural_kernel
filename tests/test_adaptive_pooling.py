import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.layers.pooling import AdaptiveAvgPool2d, AdaptiveMaxPool2d


def test_adaptive_avgpool2d_output_shape():
    x = Tensor(np.random.randn(2, 3, 7, 7), requires_grad=True)
    pool = AdaptiveAvgPool2d((1, 1))

    y = pool(x)

    assert y.data.shape == (2, 3, 1, 1)


def test_adaptive_avgpool2d_manual_case():
    x_data = np.array(
        [[[[1.0, 2.0],
           [3.0, 4.0]]]]
    )
    x = Tensor(x_data, requires_grad=True)

    pool = AdaptiveAvgPool2d((1, 1))
    y = pool(x)

    expected = np.array([[[[2.5]]]])

    assert np.allclose(y.data, expected)


def test_adaptive_avgpool2d_backward_runs():
    x = Tensor(np.random.randn(2, 3, 5, 5), requires_grad=True)
    pool = AdaptiveAvgPool2d((1, 1))

    y = pool(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.data.shape


def test_adaptive_maxpool2d_output_shape():
    x = Tensor(np.random.randn(2, 3, 7, 7), requires_grad=True)
    pool = AdaptiveMaxPool2d((1, 1))

    y = pool(x)

    assert y.data.shape == (2, 3, 1, 1)


def test_adaptive_maxpool2d_manual_case():
    x_data = np.array(
        [[[[1.0, 2.0],
           [3.0, 4.0]]]]
    )
    x = Tensor(x_data, requires_grad=True)

    pool = AdaptiveMaxPool2d((1, 1))
    y = pool(x)

    expected = np.array([[[[4.0]]]])

    assert np.allclose(y.data, expected)


def test_adaptive_maxpool2d_backward_runs():
    x = Tensor(np.random.randn(2, 3, 5, 5), requires_grad=True)
    pool = AdaptiveMaxPool2d((1, 1))

    y = pool(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.data.shape