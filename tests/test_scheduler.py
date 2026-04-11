import numpy as np

from kernel.core.tensor import Tensor
from kernel.optim import SGD, Adam, StepLR


def test_steplr_keeps_lr_before_step_boundary():
    p = Tensor(np.array([1.0]), requires_grad=True)
    opt = SGD([p], lr=0.1)
    scheduler = StepLR(opt, step_size=3, gamma=0.5)

    scheduler.step()
    scheduler.step()

    assert opt.lr == 0.1


def test_steplr_reduces_lr_on_step_boundary():
    p = Tensor(np.array([1.0]), requires_grad=True)
    opt = SGD([p], lr=0.1)
    scheduler = StepLR(opt, step_size=3, gamma=0.5)

    scheduler.step()
    scheduler.step()
    scheduler.step()

    assert np.isclose(opt.lr, 0.05)


def test_steplr_multiple_reductions():
    p = Tensor(np.array([1.0]), requires_grad=True)
    opt = SGD([p], lr=0.1)
    scheduler = StepLR(opt, step_size=2, gamma=0.1)

    scheduler.step()  # epoch 1
    assert np.isclose(opt.lr, 0.1)

    scheduler.step()  # epoch 2
    assert np.isclose(opt.lr, 0.01)

    scheduler.step()  # epoch 3
    assert np.isclose(opt.lr, 0.01)

    scheduler.step()  # epoch 4
    assert np.isclose(opt.lr, 0.001)


def test_steplr_works_with_adam():
    p = Tensor(np.array([1.0]), requires_grad=True)
    opt = Adam([p], lr=0.001)
    scheduler = StepLR(opt, step_size=2, gamma=0.5)

    scheduler.step()
    assert np.isclose(opt.lr, 0.001)

    scheduler.step()
    assert np.isclose(opt.lr, 0.0005)