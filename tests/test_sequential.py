import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.activations import ReLU
from kernel.nn.containers import Sequential
from kernel.nn.layers.linear import Linear


def test_sequential_forward_shape():
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1),
    )

    x = Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = model(x)

    assert y.data.shape == (2, 1)


def test_sequential_parameters():
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1),
    )

    params = list(model.parameters())
    assert len(params) == 4


def test_sequential_indexing():
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1),
    )

    assert isinstance(model[0], Linear)
    assert isinstance(model[1], ReLU)
    assert isinstance(model[2], Linear)
    assert len(model) == 3