from __future__ import annotations

import numpy as np

from kernel.autograd.function import Function


def _pair(v: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(v, int):
        return (v, v)
    if isinstance(v, tuple) and len(v) == 2:
        return v
    raise ValueError(f"Expected int or tuple of length 2, got {v}")


def _conv2d_forward(x, weight, bias=None, stride=1, padding=0):
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)

    n, c_in, h, w = x.shape
    c_out, c_in_w, kh, kw = weight.shape

    if c_in != c_in_w:
        raise ValueError(f"Input channels mismatch: got {c_in}, expected {c_in_w}")

    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (ph, ph), (pw, pw)),
        mode="constant",
    )

    h_padded = h + 2 * ph
    w_padded = w + 2 * pw

    h_out = (h_padded - kh) // sh + 1
    w_out = (w_padded - kw) // sw + 1

    if h_out <= 0 or w_out <= 0:
        raise ValueError(
            f"Invalid output shape: got ({h_out}, {w_out}) from input ({h}, {w}), "
            f"kernel ({kh}, {kw}), stride ({sh}, {sw}), padding ({ph}, {pw})"
        )

    out = np.zeros((n, c_out, h_out, w_out), dtype=x.dtype)

    for b in range(n):
        for oc in range(c_out):
            for i in range(h_out):
                for j in range(w_out):
                    h_start = i * sh
                    w_start = j * sw
                    h_end = h_start + kh
                    w_end = w_start + kw

                    region = x_padded[b, :, h_start:h_end, w_start:w_end]
                    out[b, oc, i, j] = np.sum(region * weight[oc])

                    if bias is not None:
                        out[b, oc, i, j] += bias[oc]

    return out


class Conv2D(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding):
        sh, sw = _pair(stride)
        ph, pw = _pair(padding)

        out = _conv2d_forward(x, weight, bias, stride=(sh, sw), padding=(ph, pw))

        ctx.save_for_backward(x, weight, bias)
        ctx.meta["stride"] = (sh, sw)
        ctx.meta["padding"] = (ph, pw)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        sh, sw = ctx.meta["stride"]
        ph, pw = ctx.meta["padding"]

        n, c_in, h, w = x.shape
        c_out, _, kh, kw = weight.shape
        _, _, h_out, w_out = grad_output.shape

        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (ph, ph), (pw, pw)),
            mode="constant",
        )

        grad_x_padded = np.zeros_like(x_padded)
        grad_w = np.zeros_like(weight)
        grad_b = np.zeros_like(bias) if bias is not None else None

        for b in range(n):
            for oc in range(c_out):
                for i in range(h_out):
                    for j in range(w_out):
                        go = grad_output[b, oc, i, j]

                        h_start = i * sh
                        w_start = j * sw
                        h_end = h_start + kh
                        w_end = w_start + kw

                        region = x_padded[b, :, h_start:h_end, w_start:w_end]

                        grad_x_padded[b, :, h_start:h_end, w_start:w_end] += go * weight[oc]
                        grad_w[oc] += go * region

                        if grad_b is not None:
                            grad_b[oc] += go

        if ph == 0 and pw == 0:
            grad_x = grad_x_padded
        else:
            grad_x = grad_x_padded[:, :, ph:ph + h, pw:pw + w]

        return grad_x, grad_w, grad_b, None, None


def conv2d(x, weight, bias=None, stride=1, padding=0):
    return Conv2D.apply(x, weight, bias, stride, padding)