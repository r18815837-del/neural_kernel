import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.modules.transformer import PositionalEncoding


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_pe_forward_shape():
    B, T, D = 2, 5, 8

    x = Tensor(np.zeros((B, T, D), dtype=np.float64), requires_grad=True)
    pe = PositionalEncoding(d_model=D, max_len=32)

    y = pe(x)

    assert y.shape == (B, T, D)


def smoke_pe_changes_values():
    B, T, D = 1, 4, 8

    x = Tensor(np.zeros((B, T, D), dtype=np.float64), requires_grad=True)
    pe = PositionalEncoding(d_model=D, max_len=32)

    y = pe(x)
    arr = y.numpy()

    assert not np.allclose(arr, np.zeros((B, T, D)))
    assert np.allclose(arr[0, 0, :], pe.pe.numpy()[0, 0, :])


def smoke_pe_backward():
    B, T, D = 2, 5, 8

    x = Tensor(np.ones((B, T, D), dtype=np.float64), requires_grad=True)
    pe = PositionalEncoding(d_model=D, max_len=32)

    y = pe(x)
    loss = y.mean()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


def smoke_pe_buffer():
    pe = PositionalEncoding(d_model=8, max_len=16)

    assert pe.pe is not None
    assert pe.pe.requires_grad is False
    assert pe.pe.shape == (1, 16, 8)


def smoke_pe_to_cpu():
    pe = PositionalEncoding(d_model=8, max_len=16)
    pe.to("cpu")

    assert pe.pe.device == "cpu"


def main():
    check("pe forward shape", smoke_pe_forward_shape)
    check("pe changes values", smoke_pe_changes_values)
    check("pe backward", smoke_pe_backward)
    check("pe buffer", smoke_pe_buffer)
    check("pe to cpu", smoke_pe_to_cpu)


if __name__ == "__main__":
    main()