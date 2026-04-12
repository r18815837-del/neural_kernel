import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.functional.masks import make_causal_mask
from kernel.nn.functional.attention import scaled_dot_product_attention


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_mask_shape():
    mask = make_causal_mask(4)

    assert mask.shape == (1, 1, 4, 4)
    arr = mask.numpy()

    assert arr[0, 0, 0, 0] == 0.0
    assert arr[0, 0, 0, 1] < -1e8
    assert arr[0, 0, 1, 0] == 0.0
    assert arr[0, 0, 3, 3] == 0.0


def smoke_mask_device_cpu():
    mask = make_causal_mask(4, device="cpu")
    assert mask.device == "cpu"


def smoke_causal_attention():
    B, H, T, D = 1, 2, 4, 3

    q = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    k = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    v = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)

    mask = make_causal_mask(T, device=q.device)
    out, attn = scaled_dot_product_attention(q, k, v, mask=mask)

    assert out.shape == (B, H, T, D)
    assert attn.shape == (B, H, T, T)

    arr = attn.numpy()[0, 0]

    # будущие токены должны быть занулены
    assert np.allclose(arr[0, 1:], 0.0, atol=1e-6)
    assert np.allclose(arr[1, 2:], 0.0, atol=1e-6)
    assert np.allclose(arr[2, 3:], 0.0, atol=1e-6)


def smoke_causal_attention_backward():
    B, H, T, D = 1, 2, 4, 3

    q = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    k = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    v = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)

    mask = make_causal_mask(T, device=q.device)
    out, attn = scaled_dot_product_attention(q, k, v, mask=mask)

    loss = out.mean()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def main():
    check("mask shape", smoke_mask_shape)
    check("mask device cpu", smoke_mask_device_cpu)
    check("causal attention", smoke_causal_attention)
    check("causal attention backward", smoke_causal_attention_backward)


if __name__ == "__main__":
    main()