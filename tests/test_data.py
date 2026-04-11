import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.layers import Conv2d


def test_conv2d_output_shape_no_padding_stride1():
    x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
    conv = Conv2d(3, 4, 3, stride=1, padding=0)

    y = conv(x)

    assert y.data.shape == (2, 4, 6, 6)


def test_conv2d_output_shape_with_padding():
    x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
    conv = Conv2d(3, 4, 3, stride=1, padding=1)

    y = conv(x)

    assert y.data.shape == (2, 4, 8, 8)


def test_conv2d_output_shape_with_stride():
    x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
    conv = Conv2d(3, 4, 3, stride=2, padding=0)

    y = conv(x)

    assert y.data.shape == (2, 4, 3, 3)


def test_conv2d_output_shape_with_padding_and_stride():
    x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
    conv = Conv2d(3, 4, 3, stride=2, padding=1)

    y = conv(x)

    assert y.data.shape == (2, 4, 4, 4)


def test_conv2d_forward_manual_case():
    x = Tensor(
        np.array(
            [[[[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0],
               [7.0, 8.0, 9.0]]]]
        ),
        requires_grad=True,
    )

    conv = Conv2d(1, 1, 2, stride=1, padding=0, bias=True)
    conv.weight.data[:] = np.array(
        [[[[1.0, 0.0],
           [0.0, -1.0]]]]
    )
    conv.bias.data[:] = np.array([0.0])

    y = conv(x)

    expected = np.array(
        [[[[-4.0, -4.0],
           [-4.0, -4.0]]]]
    )

    assert np.allclose(y.data, expected)


def test_conv2d_backward_with_padding_and_stride():
    x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
    conv = Conv2d(3, 4, 3, stride=2, padding=1, bias=True)

    y = conv(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert conv.weight.grad is not None
    assert conv.bias.grad is not None

    assert x.grad.shape == x.data.shape
    assert conv.weight.grad.shape == conv.weight.data.shape
    assert conv.bias.grad.shape == conv.bias.data.shape


def test_conv2d_without_bias():
    x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
    conv = Conv2d(3, 4, 3, stride=1, padding=1, bias=False)

    y = conv(x)
    loss = y.sum()
    loss.backward()

    assert y.data.shape == (2, 4, 8, 8)
    assert x.grad is not None
    assert conv.weight.grad is not None
    assert conv.bias is None