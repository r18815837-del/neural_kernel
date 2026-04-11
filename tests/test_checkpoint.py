import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn import BatchNorm2d, ReLU, Sequential
from kernel.nn.layers.conv import Conv2d
from kernel.nn.layers.flatten import Flatten
from kernel.nn.layers.linear import Linear
from kernel.utils.checkpoint import save_checkpoint, load_checkpoint


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

    for _ in range(3):
        _ = model1(x)

    state1 = model1.state_dict()

    model2 = build_model()
    model2.load_state_dict(state1)

    state2 = model2.state_dict()

    assert set(state1.keys()) == set(state2.keys())

    for key in state1:
        assert np.allclose(state1[key], state2[key]), f"Mismatch for key: {key}"


def test_checkpoint_save_load_with_batchnorm_buffers(tmp_path):
    np.random.seed(42)

    model1 = build_model()
    model1.train()

    x_train = Tensor(np.random.randn(16, 1, 8, 8), requires_grad=False)

    for _ in range(3):
        _ = model1(x_train)

    model1.eval()

    x_test = Tensor(np.random.randn(5, 1, 8, 8), requires_grad=False)
    y1 = model1(x_test).data.copy()

    ckpt_path = tmp_path / "bn_model.pkl"
    save_checkpoint(model1, ckpt_path)

    model2 = build_model()
    load_checkpoint(model2, ckpt_path)
    model2.eval()

    y2 = model2(x_test).data.copy()

    state1 = model1.state_dict()
    state2 = model2.state_dict()

    assert set(state1.keys()) == set(state2.keys())

    for key in state1:
        assert np.allclose(state1[key], state2[key]), f"Mismatch for key: {key}"

    assert np.allclose(y1, y2)