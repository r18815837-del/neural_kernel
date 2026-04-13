from __future__ import annotations

import pytest

from kernel.core.tensor import Tensor
from kernel.optim.sgd import SGD


def make_param(value):
    return Tensor(value, requires_grad=True)


def make_param(value):
    return Tensor(value, requires_grad=True)


def test_sgd_state_dict_has_expected_top_level_keys():
    p1 = make_param([1.0, 2.0])
    p2 = make_param([3.0, 4.0])

    optimizer = SGD([p1, p2], lr=0.1)
    state = optimizer.state_dict()

    assert isinstance(state, dict)
    assert set(state.keys()) == {"state", "param_groups", "defaults", "meta"}


def test_sgd_state_dict_contains_lr_and_class_metadata():
    p1 = make_param([1.0, 2.0])

    optimizer = SGD([p1], lr=0.05)
    state = optimizer.state_dict()

    assert state["defaults"]["lr"] == 0.05
    assert state["meta"]["optimizer_class"] == "SGD"
    assert state["meta"]["format_version"] == 1


def test_sgd_state_dict_contains_single_param_group():
    p1 = make_param([1.0, 2.0])
    p2 = make_param([3.0, 4.0])

    optimizer = SGD([p1, p2], lr=0.01)
    state = optimizer.state_dict()

    assert isinstance(state["param_groups"], list)
    assert len(state["param_groups"]) == 1

    group = state["param_groups"][0]
    assert group["params"] == [0, 1]
    assert group["lr"] == 0.01


def test_sgd_state_dict_contains_empty_per_parameter_state():
    p1 = make_param([1.0, 2.0])
    p2 = make_param([3.0, 4.0])

    optimizer = SGD([p1, p2], lr=0.01)
    state = optimizer.state_dict()

    assert state["state"] == {
        0: {},
        1: {},
    }


def test_sgd_load_state_dict_restores_lr():
    p1 = make_param([1.0, 2.0])

    source_optimizer = SGD([p1], lr=0.123)
    saved_state = source_optimizer.state_dict()

    target_param = make_param([1.0, 2.0])
    target_optimizer = SGD([target_param], lr=0.999)

    assert target_optimizer.lr == 0.999

    target_optimizer.load_state_dict(saved_state)

    assert target_optimizer.lr == 0.123
    assert target_optimizer.defaults["lr"] == 0.123


def test_sgd_state_dict_round_trip_preserves_structure():
    p1 = make_param([1.0, 2.0])
    p2 = make_param([3.0, 4.0])

    optimizer = SGD([p1, p2], lr=0.1)
    saved_state = optimizer.state_dict()

    new_p1 = make_param([1.0, 2.0])
    new_p2 = make_param([3.0, 4.0])
    restored_optimizer = SGD([new_p1, new_p2], lr=0.5)

    restored_optimizer.load_state_dict(saved_state)
    restored_state = restored_optimizer.state_dict()

    assert restored_state["defaults"] == saved_state["defaults"]
    assert restored_state["meta"] == saved_state["meta"]
    assert restored_state["param_groups"] == saved_state["param_groups"]
    assert restored_state["state"] == saved_state["state"]


def test_sgd_load_state_dict_rejects_non_dict():
    p1 = make_param([1.0, 2.0])
    optimizer = SGD([p1], lr=0.1)

    with pytest.raises(TypeError, match="must be a dict"):
        optimizer.load_state_dict(None)


def test_sgd_load_state_dict_rejects_missing_keys():
    p1 = make_param([1.0, 2.0])
    optimizer = SGD([p1], lr=0.1)

    invalid_state = {
        "state": {},
        "defaults": {"lr": 0.1},
        "meta": {"optimizer_class": "SGD", "format_version": 1},
    }

    with pytest.raises(ValueError, match="Missing keys"):
        optimizer.load_state_dict(invalid_state)


def test_sgd_load_state_dict_rejects_class_mismatch():
    p1 = make_param([1.0, 2.0])
    optimizer = SGD([p1], lr=0.1)

    invalid_state = optimizer.state_dict()
    invalid_state["meta"]["optimizer_class"] = "Adam"

    with pytest.raises(ValueError, match="Optimizer class mismatch"):
        optimizer.load_state_dict(invalid_state)


def test_sgd_load_state_dict_rejects_parameter_count_mismatch():
    p1 = make_param([1.0, 2.0])
    p2 = make_param([3.0, 4.0])

    source_optimizer = SGD([p1, p2], lr=0.1)
    saved_state = source_optimizer.state_dict()

    target_param = make_param([1.0, 2.0])
    target_optimizer = SGD([target_param], lr=0.1)

    with pytest.raises(ValueError, match="Parameter count mismatch"):
        target_optimizer.load_state_dict(saved_state)


def test_sgd_load_state_dict_rejects_missing_optimizer_class_in_meta():
    p1 = make_param([1.0, 2.0])
    optimizer = SGD([p1], lr=0.1)

    invalid_state = optimizer.state_dict()
    del invalid_state["meta"]["optimizer_class"]

    with pytest.raises(ValueError, match="optimizer_class"):
        optimizer.load_state_dict(invalid_state)


def test_sgd_load_state_dict_rejects_missing_format_version_in_meta():
    p1 = make_param([1.0, 2.0])
    optimizer = SGD([p1], lr=0.1)

    invalid_state = optimizer.state_dict()
    del invalid_state["meta"]["format_version"]

    with pytest.raises(ValueError, match="format_version"):
        optimizer.load_state_dict(invalid_state)


def test_sgd_load_state_dict_rejects_unsupported_format_version():
    p1 = make_param([1.0, 2.0])
    optimizer = SGD([p1], lr=0.1)

    invalid_state = optimizer.state_dict()
    invalid_state["meta"]["format_version"] = 999

    with pytest.raises(ValueError, match="Unsupported optimizer state format version"):
        optimizer.load_state_dict(invalid_state)