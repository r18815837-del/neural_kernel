import numpy as np


from kernel.nn import LayerNorm
from kernel.core.tensor import Tensor
from kernel.autograd.ops import softmax

def numerical_grad_scalar_fn(f, x, eps=1e-6):
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index

        old_val = x[idx]

        x[idx] = old_val + eps
        fx1 = f(x.copy())

        x[idx] = old_val - eps
        fx2 = f(x.copy())

        x[idx] = old_val
        grad[idx] = (fx1 - fx2) / (2.0 * eps)

        it.iternext()

    return grad


def test_gradcheck_layernorm_input():
    np.random.seed(42)

    x_data = np.array(
        [[1.5, -2.0, 0.7, -0.3],
         [0.2,  1.1, -1.4, 2.0]],
        dtype=np.float64,
    )

    ln = LayerNorm(4, affine=False)

    def f_numpy(x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        y = (x - mean) / np.sqrt(var + 1e-5)
        return y.sum()

    x = Tensor(x_data.copy(), requires_grad=True)
    y = ln(x).sum()
    y.backward()

    autograd_grad = x.grad
    numeric_grad = numerical_grad_scalar_fn(f_numpy, x_data.copy())

    assert np.allclose(autograd_grad, numeric_grad, atol=1e-4, rtol=1e-4)


def numerical_grad_scalar_fn(f, x, eps=1e-6):
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index

        old_val = x[idx]

        x[idx] = old_val + eps
        fx1 = f(x.copy())

        x[idx] = old_val - eps
        fx2 = f(x.copy())

        x[idx] = old_val
        grad[idx] = (fx1 - fx2) / (2.0 * eps)

        it.iternext()

    return grad


def test_gradcheck_softmax():
    x_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)

    def f_numpy(x):
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shifted)
        y = exp / np.sum(exp, axis=1, keepdims=True)
        return y.sum()

    x = Tensor(x_data.copy(), requires_grad=True)
    y = softmax(x, axis=1).sum()
    y.backward()

    autograd_grad = x.grad
    numeric_grad = numerical_grad_scalar_fn(f_numpy, x_data.copy())

    assert np.allclose(autograd_grad, numeric_grad, atol=1e-5, rtol=1e-5)