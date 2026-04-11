import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.layers.pooling import AvgPool2d, MaxPool2d


def test_maxpool2d_forward_shape():
    x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
    pool = MaxPool2d(kernel_size=2)

    y = pool(x)

    assert y.data.shape == (2, 3, 4, 4)


def test_maxpool2d_known_case():
    x = Tensor(
        np.array([[[[1.0, 2.0],
                    [3.0, 4.0]]]]),
        requires_grad=True,
    )

    pool = MaxPool2d(kernel_size=2)
    y = pool(x)

    expected = np.array([[[[4.0]]]])
    assert np.allclose(y.data, expected)


def test_maxpool2d_backward():
    x = Tensor(
        np.array([[[[1.0, 2.0],
                    [3.0, 4.0]]]]),
        requires_grad=True,
    )

    pool = MaxPool2d(kernel_size=2)
    y = pool(x).sum()
    y.backward()

    expected_grad = np.array([[[[0.0, 0.0],
                                [0.0, 1.0]]]])
    assert np.allclose(x.grad, expected_grad)


def test_avgpool2d_forward_shape():
    x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
    pool = AvgPool2d(kernel_size=2)

    y = pool(x)

    assert y.data.shape == (2, 3, 4, 4)


def test_avgpool2d_known_case():
    x = Tensor(
        np.array([[[[1.0, 2.0],
                    [3.0, 4.0]]]]),
        requires_grad=True,
    )

    pool = AvgPool2d(kernel_size=2)
    y = pool(x)

    expected = np.array([[[[2.5]]]])
    assert np.allclose(y.data, expected)


def test_avgpool2d_backward():
    x = Tensor(
        np.array([[[[1.0, 2.0],
                    [3.0, 4.0]]]]),
        requires_grad=True,
    )

    pool = AvgPool2d(kernel_size=2)
    y = pool(x).sum()
    y.backward()

    expected_grad = np.array([[[[0.25, 0.25],
                                [0.25, 0.25]]]])
    assert np.allclose(x.grad, expected_grad)