import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.init import (
    xavier_uniform_,
    xavier_normal_,
    kaiming_uniform_,
    kaiming_normal_,
)


def test_xavier_uniform_changes_tensor():
    t = Tensor(np.zeros((4, 8)), requires_grad=True)
    xavier_uniform_(t)

    assert t.data.shape == (4, 8)
    assert not np.allclose(t.data, 0.0)


def test_xavier_normal_changes_tensor():
    t = Tensor(np.zeros((4, 8)), requires_grad=True)
    xavier_normal_(t)

    assert t.data.shape == (4, 8)
    assert not np.allclose(t.data, 0.0)


def test_kaiming_uniform_changes_tensor():
    t = Tensor(np.zeros((4, 8)), requires_grad=True)
    kaiming_uniform_(t)

    assert t.data.shape == (4, 8)
    assert not np.allclose(t.data, 0.0)


def test_kaiming_normal_changes_tensor():
    t = Tensor(np.zeros((4, 8)), requires_grad=True)
    kaiming_normal_(t)

    assert t.data.shape == (4, 8)
    assert not np.allclose(t.data, 0.0)


def test_init_conv_tensor_shape():
    t = Tensor(np.zeros((16, 3, 3, 3)), requires_grad=True)
    kaiming_normal_(t)

    assert t.data.shape == (16, 3, 3, 3)