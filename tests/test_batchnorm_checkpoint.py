import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn import BatchNorm2d, ReLU, Sequential
from kernel.nn.layers.conv import Conv2d
from kernel.nn.layers.flatten import Flatten
from kernel.nn.layers.linear import Linear


def build_model():
    return Sequential(
        Conv2d(1, 4, 3, padding=1),
        BatchNorm2d(4),
        ReLU(),
        Flatten(),
        Linear(4 * 8 * 8, 10),
    )


def test_batchnorm_buffers_are_saved_and_loaded_in_state_dict():
    np.random.seed(42)

    model1 = build_model()
    model1.train()

    x = Tensor(np.random.randn(16, 1, 8, 8), requires_grad=False)

    # Несколько проходов, чтобы running stats точно обновились
    for _ in range(3):
        _ = model1(x)

    state = model1.state_dict()

    model2 = build_model()
    model2.load_state_dict(state)

    state2 = model2.state_dict()

    assert set(state.keys()) == set(state2.keys())

    for key in state:
        assert np.allclose(state[key], state2[key]), f"Mismatch for key: {key}"