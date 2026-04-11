import numpy as np

from kernel.core.tensor import Tensor
from kernel.loss.regression import MSELoss
from kernel.nn.layers.linear import Linear
from kernel.optim.sgd import SGD


def test_sgd_updates_parameter():
    w = Tensor([1.0], requires_grad=True)
    w.grad = np.array([0.5])
    opt = SGD([w], lr=0.1)
    opt.step()
    assert np.allclose(w.data, np.array([0.95]))


def test_training_step_reduces_loss():
    np.random.seed(0)
    x_np = np.linspace(-1, 1, 20).reshape(-1, 1)
    y_np = 3 * x_np + 2

    x = Tensor(x_np)
    y = Tensor(y_np)

    model = Linear(1, 1)
    criterion = MSELoss()
    optim = SGD(model.parameters(), lr=0.1)

    initial = criterion(model(x), y).data.item()
    for _ in range(50):
        optim.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optim.step()
    final = criterion(model(x), y).data.item()

    assert final < initial
