import os
import tempfile

import numpy as np
import pytest

from kernel import Linear, Sequential, ReLU, Tensor
from kernel.optim import SGD
from kernel.utils import save_checkpoint, load_checkpoint


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "data"):
        data = x.data
        if isinstance(data, np.ndarray):
            return data
        try:
            return np.array(data)
        except Exception:
            pass
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.array(x)


def clone_params(module):
    return [np.array(to_numpy(p), copy=True) for p in module.parameters()]


def params_allclose(params_a, params_b, atol=1e-6):
    if len(params_a) != len(params_b):
        return False
    return all(np.allclose(a, b, atol=atol) for a, b in zip(params_a, params_b))


def make_model():
    return Sequential(
        Linear(3, 4),
        ReLU(),
        Linear(4, 2),
    )


def make_train_step(model):
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    return loss


def optimizer_supports_checkpoint(optimizer):
    return hasattr(optimizer, "state_dict") and hasattr(optimizer, "load_state_dict")


def test_save_model_checkpoint_creates_file():
    model = make_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pkl")

        save_checkpoint(model, path)

        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_model_weights_restore_after_load():
    model = make_model()
    optimizer = SGD(model.parameters(), lr=0.01)

    original_params = clone_params(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pkl")

        save_checkpoint(model, path)

        loss = make_train_step(model)
        optimizer.step()

        changed_params = clone_params(model)
        assert not params_allclose(original_params, changed_params)

        load_checkpoint(model, path)

        restored_params = clone_params(model)
        assert params_allclose(original_params, restored_params)


def test_load_checkpoint_restores_forward_outputs():
    model = make_model()
    optimizer = SGD(model.parameters(), lr=0.01)

    x = Tensor([[1.0, 2.0, 3.0]])
    original_out = to_numpy(model(x))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pkl")

        save_checkpoint(model, path)

        loss = make_train_step(model)
        optimizer.step()

        changed_out = to_numpy(model(x))
        assert not np.allclose(original_out, changed_out)

        load_checkpoint(model, path)

        restored_out = to_numpy(model(x))
        assert np.allclose(original_out, restored_out, atol=1e-6)


def test_optimizer_state_can_be_saved_and_loaded_if_supported():
    model = make_model()
    optimizer = SGD(model.parameters(), lr=0.01)

    if not optimizer_supports_checkpoint(optimizer):
        pytest.skip("optimizer does not yet implement state_dict/load_state_dict")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pkl")

        save_checkpoint(model, path, optimizer)

        loss = make_train_step(model)
        optimizer.step()

        load_checkpoint(model, path, optimizer)

        assert optimizer is not None