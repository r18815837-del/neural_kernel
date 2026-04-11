import numpy as np

from kernel.core.tensor import Tensor
from kernel.loss.classification import CrossEntropyLoss
from kernel.nn.layers.linear import Linear
from kernel.optim.adam import Adam


def test_cross_entropy_returns_scalar():
    logits = Tensor([[2.0, 1.0, 0.1], [0.5, 1.5, 2.5]], requires_grad=True)
    targets = np.array([0, 2])

    criterion = CrossEntropyLoss()
    loss = criterion(logits, targets)

    assert loss.data.shape == ()
    assert loss.data.item() > 0.0


def test_cross_entropy_backward_shape():
    logits = Tensor([[2.0, 1.0, 0.1], [0.5, 1.5, 2.5]], requires_grad=True)
    targets = np.array([0, 2])

    criterion = CrossEntropyLoss()
    loss = criterion(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert logits.grad.shape == logits.data.shape


def test_adam_updates_parameter():
    w = Tensor([1.0], requires_grad=True)
    w.grad = np.array([0.5])

    opt = Adam([w], lr=0.1)
    before = w.data.copy()
    opt.step()

    assert not np.allclose(w.data, before)


def test_classification_training_reduces_loss():
    np.random.seed(0)

    x_np = np.array([
        [-2.0, -1.0],
        [-1.5, -1.0],
        [-1.0, -2.0],
        [1.0, 1.5],
        [1.5, 1.0],
        [2.0, 1.0],
    ])
    y_np = np.array([0, 0, 0, 1, 1, 1])

    x = Tensor(x_np)
    model = Linear(2, 2)
    criterion = CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=0.05)

    initial = criterion(model(x), y_np).data.item()

    for _ in range(200):
        optim.zero_grad()
        logits = model(x)
        loss = criterion(logits, y_np)
        loss.backward()
        optim.step()

    final = criterion(model(x), y_np).data.item()

    assert final < initial