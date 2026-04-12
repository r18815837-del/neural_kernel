import numpy as np

from kernel.core.tensor import Tensor
from kernel.nn.functional.masks import make_padding_mask
from kernel.nn.functional.attention import scaled_dot_product_attention


def check(name, fn):
    try:
        fn()
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def smoke_padding_mask_shape():
    mask = make_padding_mask([5, 3, 4], max_len=5)

    assert mask.shape == (3, 1, 1, 5)
    arr = mask.numpy()

    assert np.allclose(arr[0, 0, 0], np.array([0, 0, 0, 0, 0], dtype=np.float64))
    assert np.allclose(arr[1, 0, 0], np.array([0, 0, 0, -1e9, -1e9], dtype=np.float64))
    assert np.allclose(arr[2, 0, 0], np.array([0, 0, 0, 0, -1e9], dtype=np.float64))


def smoke_padding_mask_infer_max_len():
    mask = make_padding_mask([2, 4, 3])

    assert mask.shape == (3, 1, 1, 4)


def smoke_padding_mask_device_cpu():
    mask = make_padding_mask([3, 2], device="cpu")
    assert mask.device == "cpu"


def smoke_padding_attention():
    B, H, T, D = 2, 2, 5, 3

    q = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    k = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    v = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)

    mask = make_padding_mask([5, 3], max_len=5, device=q.device)
    out, attn = scaled_dot_product_attention(q, k, v, mask=mask)

    assert out.shape == (B, H, T, D)
    assert attn.shape == (B, H, T, T)

    arr = attn.numpy()
    # во втором примере последние 2 позиции должны быть занулены
    assert np.allclose(arr[1, :, :, 3:], 0.0, atol=1e-6)


def smoke_padding_attention_backward():
    B, H, T, D = 2, 2, 5, 3

    q = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    k = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)
    v = Tensor(np.arange(B * H * T * D, dtype=np.float64).reshape(B, H, T, D), requires_grad=True)

    mask = make_padding_mask([5, 3], max_len=5, device=q.device)
    out, attn = scaled_dot_product_attention(q, k, v, mask=mask)

    loss = out.mean()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def main():
    check("padding mask shape", smoke_padding_mask_shape)
    check("padding mask infer max len", smoke_padding_mask_infer_max_len)
    check("padding mask device cpu", smoke_padding_mask_device_cpu)
    check("padding attention", smoke_padding_attention)
    check("padding attention backward", smoke_padding_attention_backward)


if __name__ == "__main__":
    main()