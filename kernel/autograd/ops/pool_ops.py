from __future__ import annotations

import math
import numpy as np

from kernel.autograd.function import Function


class MaxPool2D(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride):
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size

        if stride is None:
            sh, sw = kh, kw
        elif isinstance(stride, int):
            sh = sw = stride
        else:
            sh, sw = stride

        n, c, h, w = x.shape
        h_out = (h - kh) // sh + 1
        w_out = (w - kw) // sw + 1

        out = np.zeros((n, c, h_out, w_out), dtype=x.dtype)
        max_idx = np.zeros((n, c, h_out, w_out, 2), dtype=np.int64)

        for b in range(n):
            for ch in range(c):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * sh
                        w_start = j * sw
                        region = x[b, ch, h_start:h_start + kh, w_start:w_start + kw]

                        flat_idx = np.argmax(region)
                        rr, cc = np.unravel_index(flat_idx, region.shape)

                        out[b, ch, i, j] = region[rr, cc]
                        max_idx[b, ch, i, j] = (h_start + rr, w_start + cc)

        ctx.save_for_backward(max_idx)
        ctx.meta["x_shape"] = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (max_idx,) = ctx.saved_tensors
        grad_x = np.zeros(ctx.meta["x_shape"], dtype=grad_output.dtype)

        n, c, h_out, w_out = grad_output.shape
        for b in range(n):
            for ch in range(c):
                for i in range(h_out):
                    for j in range(w_out):
                        rr, cc = max_idx[b, ch, i, j]
                        grad_x[b, ch, rr, cc] += grad_output[b, ch, i, j]

        return grad_x, None, None


class AvgPool2D(Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride):
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size

        if stride is None:
            sh, sw = kh, kw
        elif isinstance(stride, int):
            sh = sw = stride
        else:
            sh, sw = stride

        n, c, h, w = x.shape
        h_out = (h - kh) // sh + 1
        w_out = (w - kw) // sw + 1

        out = np.zeros((n, c, h_out, w_out), dtype=x.dtype)

        for b in range(n):
            for ch in range(c):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * sh
                        w_start = j * sw
                        region = x[b, ch, h_start:h_start + kh, w_start:w_start + kw]
                        out[b, ch, i, j] = region.mean()

        ctx.meta["x_shape"] = x.shape
        ctx.meta["kernel_size"] = (kh, kw)
        ctx.meta["stride"] = (sh, sw)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        n, c, h, w = ctx.meta["x_shape"]
        kh, kw = ctx.meta["kernel_size"]
        sh, sw = ctx.meta["stride"]

        grad_x = np.zeros((n, c, h, w), dtype=grad_output.dtype)
        scale = 1.0 / (kh * kw)

        _, _, h_out, w_out = grad_output.shape
        for b in range(n):
            for ch in range(c):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * sh
                        w_start = j * sw
                        grad_x[b, ch, h_start:h_start + kh, w_start:w_start + kw] += (
                            grad_output[b, ch, i, j] * scale
                        )

        return grad_x, None, None


class AdaptiveAvgPool2D(Function):
    @staticmethod
    def forward(ctx, x, output_size):
        if isinstance(output_size, int):
            out_h = out_w = output_size
        else:
            out_h, out_w = output_size

        n, c, h, w = x.shape
        out = np.zeros((n, c, out_h, out_w), dtype=x.dtype)
        regions = []

        for oh in range(out_h):
            h_start = math.floor(oh * h / out_h)
            h_end = math.ceil((oh + 1) * h / out_h)

            for ow in range(out_w):
                w_start = math.floor(ow * w / out_w)
                w_end = math.ceil((ow + 1) * w / out_w)

                region = x[:, :, h_start:h_end, w_start:w_end]
                out[:, :, oh, ow] = region.mean(axis=(2, 3))
                regions.append((h_start, h_end, w_start, w_end))

        ctx.meta["x_shape"] = x.shape
        ctx.meta["output_size"] = (out_h, out_w)
        ctx.meta["regions"] = regions
        return out

    @staticmethod
    def backward(ctx, grad_output):
        n, c, h, w = ctx.meta["x_shape"]
        out_h, out_w = ctx.meta["output_size"]
        regions = ctx.meta["regions"]

        grad_x = np.zeros((n, c, h, w), dtype=grad_output.dtype)

        k = 0
        for oh in range(out_h):
            for ow in range(out_w):
                h_start, h_end, w_start, w_end = regions[k]
                area = (h_end - h_start) * (w_end - w_start)
                grad = grad_output[:, :, oh, ow][:, :, None, None] / area
                grad_x[:, :, h_start:h_end, w_start:w_end] += grad
                k += 1

        return grad_x, None

class AdaptiveMaxPool2D(Function):
    @staticmethod
    def forward(ctx, x, output_size):
        if isinstance(output_size, int):
            out_h = out_w = output_size
        else:
            out_h, out_w = output_size

        n, c, h, w = x.shape
        out = np.zeros((n, c, out_h, out_w), dtype=x.dtype)
        max_indices = np.zeros((n, c, out_h, out_w, 2), dtype=np.int64)

        for oh in range(out_h):
            h_start = math.floor(oh * h / out_h)
            h_end = math.ceil((oh + 1) * h / out_h)

            for ow in range(out_w):
                w_start = math.floor(ow * w / out_w)
                w_end = math.ceil((ow + 1) * w / out_w)

                region = x[:, :, h_start:h_end, w_start:w_end]

                flat_idx = region.reshape(n, c, -1).argmax(axis=2)
                out[:, :, oh, ow] = region.reshape(n, c, -1).max(axis=2)

                region_w = w_end - w_start
                rr = flat_idx // region_w
                cc = flat_idx % region_w

                max_indices[:, :, oh, ow, 0] = h_start + rr
                max_indices[:, :, oh, ow, 1] = w_start + cc

        ctx.save_for_backward(max_indices)
        ctx.meta["x_shape"] = x.shape
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (max_indices,) = ctx.saved_tensors
        n, c, h, w = ctx.meta["x_shape"]

        grad_x = np.zeros((n, c, h, w), dtype=grad_output.dtype)
        _, _, out_h, out_w = grad_output.shape

        for b in range(n):
            for ch in range(c):
                for oh in range(out_h):
                    for ow in range(out_w):
                        rr = max_indices[b, ch, oh, ow, 0]
                        cc = max_indices[b, ch, oh, ow, 1]
                        grad_x[b, ch, rr, cc] += grad_output[b, ch, oh, ow]

        return grad_x, None


def maxpool2d(x, kernel_size, stride=None):
    return MaxPool2D.apply(x, kernel_size, stride)


def avgpool2d(x, kernel_size, stride=None):
    return AvgPool2D.apply(x, kernel_size, stride)


def adaptive_avgpool2d(x, output_size):
    return AdaptiveAvgPool2D.apply(x, output_size)


def adaptive_maxpool2d(x, output_size):
    return AdaptiveMaxPool2D.apply(x, output_size)
