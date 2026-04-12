import numpy as np

from kernel import Linear, ReLU, Sequential, Tensor
from kernel.nn.modules import ModuleDict, ModuleList


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


def count_params(module):
    params = list(module.parameters())
    return len(params)


def test_sequential_forward_shape():
    model = Sequential(
        Linear(3, 4),
        ReLU(),
        Linear(4, 2),
    )

    x = Tensor([[1.0, 2.0, 3.0]])
    y = model(x)
    arr = to_numpy(y)

    assert arr.shape == (1, 2)
    assert np.isfinite(arr).all()


def test_sequential_parameters_are_registered():
    model = Sequential(
        Linear(3, 4),
        ReLU(),
        Linear(4, 2),
    )

    params = list(model.parameters())

    assert len(params) > 0


def test_modulelist_stores_modules():
    modules = ModuleList([
        Linear(3, 4),
        ReLU(),
        Linear(4, 2),
    ])

    assert len(modules) == 3
    assert modules[0] is not None
    assert modules[1] is not None
    assert modules[2] is not None


def test_modulelist_parameters_are_registered():
    modules = ModuleList([
        Linear(3, 4),
        ReLU(),
        Linear(4, 2),
    ])

    params = list(modules.parameters())

    assert len(params) > 0


def test_moduledict_stores_named_modules():
    modules = ModuleDict({
        "fc1": Linear(3, 4),
        "act": ReLU(),
        "fc2": Linear(4, 2),
    })

    assert "fc1" in modules
    assert "act" in modules
    assert "fc2" in modules

    assert modules["fc1"] is not None
    assert modules["act"] is not None
    assert modules["fc2"] is not None


def test_moduledict_parameters_are_registered():
    modules = ModuleDict({
        "fc1": Linear(3, 4),
        "act": ReLU(),
        "fc2": Linear(4, 2),
    })

    params = list(modules.parameters())

    assert len(params) > 0


def test_sequential_has_more_than_zero_parameters():
    model = Sequential(
        Linear(3, 4),
        ReLU(),
        Linear(4, 2),
    )

    assert count_params(model) > 0


def test_modulelist_len_matches_inserted_modules():
    modules = ModuleList([
        Linear(2, 2),
        ReLU(),
        Linear(2, 1),
    ])

    assert len(modules) == 3


def test_moduledict_len_matches_inserted_modules():
    modules = ModuleDict({
        "a": Linear(2, 2),
        "b": ReLU(),
        "c": Linear(2, 1),
    })

    assert len(modules) == 3