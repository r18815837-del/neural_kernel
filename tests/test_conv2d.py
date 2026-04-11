import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.layers.conv import Conv2d


def test_conv2d_forward_shape():
    np.random.seed(42)

    x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
    conv = Conv2d(in_channels=3, out_channels=4, kernel_size=3)

    y = conv(x)

    assert y.data.shape == (2, 4, 6, 6)


def test_conv2d_backward_shapes():
    np.random.seed(42)

    x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
    conv = Conv2d(in_channels=3, out_channels=4, kernel_size=3)

    y = conv(x).sum()
    y.backward()

    assert x.grad is not None
    assert conv.weight.grad is not None
    assert conv.bias.grad is not None

    assert x.grad.shape == x.data.shape
    assert conv.weight.grad.shape == conv.weight.data.shape
    assert conv.bias.grad.shape == conv.bias.data.shape


def test_conv2d_single_channel_known_case():
    x = Tensor(
        np.array([[[[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0]]]]),
        requires_grad=True,
    )

    conv = Conv2d(in_channels=1, out_channels=1, kernel_size=2)

    conv.weight.data[...] = np.array([[[[1.0, 0.0],
                                        [0.0, -1.0]]]])
    conv.bias.data[...] = np.array([0.0])

    y = conv(x)

    expected = np.array([[[[-4.0, -4.0],
                           [-4.0, -4.0]]]])

    assert np.allclose(y.data, expected)