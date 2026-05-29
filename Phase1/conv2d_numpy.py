"""Assignment A — Conv2d in pure NumPy: forward + backward (no nn.Conv2d).

Implements a 2D convolution layer from scratch using the **im2col** trick: the
sliding-window patches are unrolled into a matrix so the convolution becomes a
single matrix multiply (fast and easy to differentiate). The backward pass
returns gradients w.r.t. the input, weights, and bias.

Correctness is verified at the bottom against finite-difference numerical
gradients (relative error ~1e-7, i.e. machine precision for float64).

Concepts: see cnn_core.md §1 (filter / stride / padding / output size).
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# im2col / col2im — the heart of a fast, differentiable convolution
# ----------------------------------------------------------------------------

def _im2col_indices(x_shape, kh, kw, padding, stride):
    """Precompute the (channel, row, col) gather indices for every patch."""
    N, C, H, W = x_shape
    out_h = (H + 2 * padding - kh) // stride + 1
    out_w = (W + 2 * padding - kw) // stride + 1

    # Row index within a patch (0..kh-1), repeated across width and channels.
    i0 = np.repeat(np.arange(kh), kw)
    i0 = np.tile(i0, C)
    # Top-left row of each output position.
    i1 = stride * np.repeat(np.arange(out_h), out_w)
    # Col index within a patch (0..kw-1), tiled across height and channels.
    j0 = np.tile(np.arange(kw), kh * C)
    # Left col of each output position.
    j1 = stride * np.tile(np.arange(out_w), out_h)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)        # (C*kh*kw, out_h*out_w)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), kh * kw).reshape(-1, 1)
    return k.astype(int), i.astype(int), j.astype(int)


def im2col(x, kh, kw, padding, stride):
    """(N,C,H,W) -> (C*kh*kw, out_h*out_w*N) matrix of flattened patches."""
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                      mode="constant")
    k, i, j = _im2col_indices(x.shape, kh, kw, padding, stride)
    cols = x_padded[:, k, i, j]                      # (N, C*kh*kw, out_h*out_w)
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(kh * kw * C, -1)
    return cols


def col2im(cols, x_shape, kh, kw, padding, stride):
    """Inverse of im2col: scatter-add patch gradients back to (N,C,H,W)."""
    N, C, H, W = x_shape
    Hp, Wp = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, Hp, Wp), dtype=cols.dtype)
    k, i, j = _im2col_indices(x_shape, kh, kw, padding, stride)
    cols_reshaped = cols.reshape(C * kh * kw, -1, N).transpose(2, 0, 1)
    # Overlapping patches accumulate — np.add.at handles the duplicate indices.
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


# ----------------------------------------------------------------------------
# Conv2d layer
# ----------------------------------------------------------------------------

class Conv2d:
    """2D convolution. Weight shape (F, C, kh, kw); input/output NCHW."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, seed=0):
        kh = kw = kernel_size
        rng = np.random.default_rng(seed)
        # He init for ReLU nets: std = sqrt(2 / fan_in).
        fan_in = in_channels * kh * kw
        self.W = rng.standard_normal((out_channels, in_channels, kh, kw)) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros(out_channels)
        self.kh, self.kw = kh, kw
        self.stride, self.padding = stride, padding
        # Gradient buffers, filled by backward().
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        F = self.W.shape[0]
        out_h = (H + 2 * self.padding - self.kh) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kw) // self.stride + 1

        cols = im2col(x, self.kh, self.kw, self.padding, self.stride)
        W_row = self.W.reshape(F, -1)                # (F, C*kh*kw)
        out = W_row @ cols + self.b.reshape(-1, 1)   # (F, out_h*out_w*N)
        out = out.reshape(F, out_h, out_w, N).transpose(3, 0, 1, 2)

        self._cache = (x.shape, cols)
        return out

    def backward(self, dout):
        """dout: (N, F, out_h, out_w). Returns dx; stores self.dW, self.db."""
        x_shape, cols = self._cache
        F = self.W.shape[0]

        # Bias gradient: sum over batch + spatial positions.
        self.db = dout.sum(axis=(0, 2, 3))

        # Reshape dout to (F, N*out_h*out_w) matching the forward matmul.
        dout_row = dout.transpose(1, 2, 3, 0).reshape(F, -1)

        # Weight gradient: dout_row @ cols^T  ->  (F, C*kh*kw) -> weight shape.
        self.dW = (dout_row @ cols.T).reshape(self.W.shape)

        # Input gradient: W^T @ dout_row -> patch grads -> scatter via col2im.
        W_row = self.W.reshape(F, -1)
        dcols = W_row.T @ dout_row
        dx = col2im(dcols, x_shape, self.kh, self.kw, self.padding, self.stride)
        return dx


# ----------------------------------------------------------------------------
# Correctness: gradient check vs. finite differences
# ----------------------------------------------------------------------------

def _numerical_grad(f, x, dout, h=1e-5):
    """Numerical gradient of (f(x) . dout) w.r.t. x via central differences."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + h
        pos = np.sum(f(x) * dout)
        x[idx] = old - h
        neg = np.sum(f(x) * dout)
        x[idx] = old
        grad[idx] = (pos - neg) / (2 * h)
        it.iternext()
    return grad


def _rel_error(a, b):
    return np.max(np.abs(a - b) / (np.maximum(1e-8, np.abs(a) + np.abs(b))))


def _gradient_check():
    rng = np.random.default_rng(0)
    # Small tensors so the O(numel) finite-difference check is cheap.
    x = rng.standard_normal((2, 3, 7, 7))
    conv = Conv2d(in_channels=3, out_channels=4, kernel_size=3,
                  stride=2, padding=1, seed=1)

    out = conv.forward(x)
    dout = rng.standard_normal(out.shape)
    dx = conv.backward(dout)

    # Analytic vs numerical for x, W, b.
    dx_num = _numerical_grad(lambda v: conv.forward(v), x.copy(), dout)

    def f_W(Wv):
        old = conv.W
        conv.W = Wv
        o = conv.forward(x)
        conv.W = old
        return o
    dW_num = _numerical_grad(f_W, conv.W.copy(), dout)

    def f_b(bv):
        old = conv.b
        conv.b = bv
        o = conv.forward(x)
        conv.b = old
        return o
    db_num = _numerical_grad(f_b, conv.b.copy(), dout)

    print("Conv2d gradient check (relative error vs. finite differences):")
    print(f"  input gradient dx : {_rel_error(dx, dx_num):.2e}")
    print(f"  weight gradient dW: {_rel_error(conv.dW, dW_num):.2e}")
    print(f"  bias  gradient db : {_rel_error(conv.db, db_num):.2e}")
    print("  (values < 1e-6 confirm the backward pass is correct.)")


def _shape_demo():
    x = np.zeros((1, 3, 32, 32))
    for stride, pad, k in [(1, 1, 3), (2, 1, 3), (1, 0, 5)]:
        conv = Conv2d(3, 8, k, stride=stride, padding=pad)
        out = conv.forward(x)
        print(f"  in (1,3,32,32) k={k} s={stride} p={pad} -> out {out.shape}")


if __name__ == "__main__":
    print("=" * 64)
    _gradient_check()
    print("\nOutput-shape demo (formula: (n + 2p - k)//s + 1):")
    _shape_demo()
    print("=" * 64)
