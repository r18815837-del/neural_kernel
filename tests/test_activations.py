import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.activations import ReLU, Sigmoid, LeakyReLU, Tanh, Identity, Softmax


def test_relu_forward():
    x = Tensor(np.array([-1.0, 0.0, 2.0]), requires_grad=True)
    y = x.relu()
    assert np.allclose(y.data, np.array([0.0, 0.0, 2.0]))


def test_relu_backward():
    x = Tensor(np.array([-1.0, 2.0, 3.0]), requires_grad=True)
    y = x.relu().sum()
    y.backward()
    assert np.allclose(x.grad, np.array([0.0, 1.0, 1.0]))


def test_sigmoid_forward():
    x = Tensor(np.array([0.0, 2.0]), requires_grad=True)
    y = x.sigmoid()
    expected = 1.0 / (1.0 + np.exp(-np.array([0.0, 2.0])))
    assert np.allclose(y.data, expected)


def test_sigmoid_backward():
    x = Tensor(np.array([0.0]), requires_grad=True)
    y = x.sigmoid()
    z = y.sum()
    z.backward()

    s = 1.0 / (1.0 + np.exp(-0.0))
    expected_grad = s * (1.0 - s)
    assert np.allclose(x.grad, np.array([expected_grad]))


def test_relu_module():
    x = Tensor(np.array([-2.0, 1.5]), requires_grad=True)
    layer = ReLU()
    y = layer(x)
    assert np.allclose(y.data, np.array([0.0, 1.5]))


def test_sigmoid_module():
    x = Tensor(np.array([0.0]), requires_grad=True)
    layer = Sigmoid()
    y = layer(x)
    assert np.allclose(y.data, np.array([0.5]))

def test_leaky_relu_forward():
    x = Tensor(np.array([[-2.0, -1.0, 0.0, 3.0]]), requires_grad=False)
    act = LeakyReLU(negative_slope=0.1)

    y = act(x)

    expected = np.array([[-0.2, -0.1, 0.0, 3.0]])
    assert np.allclose(y.data, expected)


def test_leaky_relu_backward():
    x_data = np.array([[-2.0, -1.0, 0.0, 3.0]])
    x = Tensor(x_data, requires_grad=True)
    act = LeakyReLU(negative_slope=0.1)

    y = act(x).sum()
    y.backward()

    expected_grad = np.array([[0.1, 0.1, 0.1, 1.0]])
    assert np.allclose(x.grad, expected_grad)


def test_tanh_forward():
    x = Tensor(np.array([[-1.0, 0.0, 1.0]]), requires_grad=False)
    act = Tanh()

    y = act(x)

    expected = np.tanh(np.array([[-1.0, 0.0, 1.0]]))
    assert np.allclose(y.data, expected)


def test_tanh_backward():
    x_data = np.array([[-1.0, 0.0, 1.0]])
    x = Tensor(x_data, requires_grad=True)
    act = Tanh()

    y = act(x).sum()
    y.backward()

    expected_grad = 1.0 - np.tanh(x_data) ** 2
    assert np.allclose(x.grad, expected_grad)

def test_identity_forward_returns_same_values():
    x_data = np.array([[1.0, -2.0], [3.5, 0.0]])
    x = Tensor(x_data, requires_grad=False)

    layer = Identity()
    y = layer(x)

    assert np.allclose(y.data, x_data)


def test_identity_backward_passes_gradient():
    x_data = np.array([[1.0, -2.0], [3.5, 0.0]])
    x = Tensor(x_data, requires_grad=True)

    layer = Identity()
    y = layer(x).sum()
    y.backward()

    expected_grad = np.ones_like(x_data)
    assert np.allclose(x.grad, expected_grad)

def test_softmax_forward_rows_sum_to_one():
    x = Tensor(np.array([[1.0, 2.0, 3.0], [0.5, 0.0, -1.0]]), requires_grad=False)
    layer = Softmax(axis=1)

    y = layer(x)

    row_sums = y.data.sum(axis=1)
    assert np.allclose(row_sums, np.ones_like(row_sums))


def test_softmax_output_is_positive():
    x = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=False)
    layer = Softmax(axis=1)

    y = layer(x)

    assert np.all(y.data > 0.0)