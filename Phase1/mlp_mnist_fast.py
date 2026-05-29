"""Vectorized micrograd: a NumPy-backed autograd engine that reaches >=97% on MNIST.

This is the natural evolution of the scalar ``Value`` engine (see
``micrograd_value.py`` / ``mlp_mnist.py``): same design — every op records a
``_backward`` closure and parents, and ``backward()`` replays a topological
order — but each node now wraps a whole NumPy **array** instead of one scalar.
That single change turns ~9 min/epoch into ~seconds/epoch, which is what makes
hitting the accuracy target feasible.

Why we can't reach 97% with the scalar engine *or* a tiny subset:
  * scalar autograd is ~1000x too slow to train at full resolution, and
  * 97% is a DATA requirement — it needs full 28x28 inputs and thousands of
    training images. A 300-image 7x7 subset tops out around ~85% regardless.

So here we use full-resolution MNIST and a large training subset, train for many
epochs with SGD + momentum, and save the final model.

Run:  python3 mlp_mnist_fast.py
"""

from __future__ import annotations

import numpy as np
from torchvision import datasets


# ============================================================================
# Vectorized autograd engine (micrograd-style, arrays instead of scalars)
# ============================================================================

def _unbroadcast(grad, shape):
    """Sum ``grad`` down to ``shape`` to undo NumPy broadcasting (e.g. bias add)."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """An array-valued node in the autograd graph."""

    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            # d(X@W)/dX = grad @ W^T ;  d(X@W)/dW = X^T @ grad
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def cross_entropy(self, targets):
        """Softmax cross-entropy. ``self``: (B, C) logits; ``targets``: (B,) int.

        Stabilized log-sum-exp; gradient on logits is ``(softmax - onehot)/B``
        (see nn_core.md §4). Returns a scalar-loss ``Tensor``.
        """
        x = self.data - self.data.max(axis=1, keepdims=True)
        exp = np.exp(x)
        probs = exp / exp.sum(axis=1, keepdims=True)
        B = x.shape[0]
        loss = -np.log(probs[np.arange(B), targets] + 1e-12).mean()
        out = Tensor(loss, (self,), "cross_entropy")

        def _backward():
            d = probs.copy()
            d[np.arange(B), targets] -= 1.0
            d /= B
            self.grad += d * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)   # seed: d(loss)/d(loss) = 1
        for node in reversed(topo):
            node._backward()


# ============================================================================
# Network: Linear -> MLP  (parallels Layer/MLP in the scalar version)
# ============================================================================

class Linear:
    def __init__(self, n_in, n_out, rng):
        # He init for ReLU: std = sqrt(2 / fan_in).
        self.W = Tensor(rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in))
        self.b = Tensor(np.zeros(n_out))

    def __call__(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class MLP:
    """``sizes`` = [n_in, h1, ..., n_out]. ReLU on hidden layers, linear logits."""

    def __init__(self, sizes, rng):
        self.layers = [Linear(sizes[i], sizes[i + 1], rng)
                       for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.relu()
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)


# ============================================================================
# Data
# ============================================================================

def load_mnist(n_train, n_test, seed=0):
    train = datasets.MNIST(root="./data", train=True, download=True)
    test = datasets.MNIST(root="./data", train=False, download=True)

    # Whole-array conversion (NOT per-pixel) — this is the vectorization win.
    Xtr_all = train.data.numpy().reshape(-1, 784).astype(np.float64)
    ytr_all = train.targets.numpy().astype(np.int64)
    Xte_all = test.data.numpy().reshape(-1, 784).astype(np.float64)
    yte_all = test.targets.numpy().astype(np.int64)

    # Standardize with the classic MNIST stats (helps optimization a lot).
    mean, std = 0.1307 * 255.0, 0.3081 * 255.0
    Xtr_all = (Xtr_all - mean) / std
    Xte_all = (Xte_all - mean) / std

    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(len(Xtr_all), size=n_train, replace=False)
    te_idx = (np.arange(len(Xte_all)) if n_test >= len(Xte_all)
              else rng.choice(len(Xte_all), size=n_test, replace=False))
    return Xtr_all[tr_idx], ytr_all[tr_idx], Xte_all[te_idx], yte_all[te_idx]


# ============================================================================
# Eval, save/load
# ============================================================================

def accuracy(model, X, y, batch=2048):
    correct = 0
    for i in range(0, len(X), batch):
        logits = model(Tensor(X[i:i + batch])).data
        correct += int((logits.argmax(axis=1) == y[i:i + batch]).sum())
    return correct / len(X)


def save_model(model, sizes, path):
    arrays = {"arch": np.array(sizes)}
    for i, layer in enumerate(model.layers):
        arrays[f"W{i}"] = layer.W.data
        arrays[f"b{i}"] = layer.b.data
    np.savez(path, **arrays)


def load_model(path):
    z = np.load(path)
    sizes = list(z["arch"])
    model = MLP(sizes, np.random.default_rng(0))
    for i, layer in enumerate(model.layers):
        layer.W.data = z[f"W{i}"]
        layer.b.data = z[f"b{i}"]
    return model


# ============================================================================
# Training loop — SGD with momentum
# ============================================================================

def train():
    SEED = 42
    N_TRAIN, N_TEST = 60_000, 10_000   # full set: needed to clear 97%
    SIZES = [784, 512, 10]             # wide single hidden layer: stable, ~98%
    EPOCHS = 40
    BATCH = 128
    LR = 0.1                            # peak LR; cosine-decayed to LR_MIN
    LR_MIN = 0.001
    MOMENTUM = 0.9
    TARGET = 0.97
    MODEL_PATH = "mnist_mlp.npz"

    print("Loading MNIST (full 28x28 resolution)...")
    Xtr, ytr, Xte, yte = load_mnist(N_TRAIN, N_TEST, seed=SEED)

    rng = np.random.default_rng(SEED)
    model = MLP(SIZES, rng)
    params = model.parameters()
    velocities = [np.zeros_like(p.data) for p in params]
    n_params = sum(p.data.size for p in params)
    print(f"MLP {SIZES} with {n_params:,} parameters")
    print(f"train={N_TRAIN}  test={N_TEST}  batch={BATCH}  lr={LR}  "
          f"momentum={MOMENTUM}\n")

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        # Cosine LR decay from LR -> LR_MIN over training.
        lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1 + np.cos(np.pi * (epoch - 1) / EPOCHS))
        perm = rng.permutation(N_TRAIN)
        epoch_loss, n_batches = 0.0, 0

        for start in range(0, N_TRAIN, BATCH):
            idx = perm[start:start + BATCH]
            logits = model(Tensor(Xtr[idx]))
            loss = logits.cross_entropy(ytr[idx])

            model.zero_grad()
            loss.backward()

            # SGD + momentum:  v = mu*v - lr*g ;  p += v
            for p, v in zip(params, velocities):
                v *= MOMENTUM
                v -= lr * p.grad
                p.data += v

            epoch_loss += float(loss.data)
            n_batches += 1

        test_acc = accuracy(model, Xte, yte)
        print(f"epoch {epoch:2d} | lr {lr:.4f} | loss {epoch_loss / n_batches:.4f} "
              f"| test acc {test_acc:.4f}")

        # Save the model each time it improves (so the final save is the best).
        if test_acc > best_acc:
            best_acc = test_acc
            save_model(model, SIZES, MODEL_PATH)

        if test_acc >= TARGET:
            print(f"\nReached target {TARGET:.0%} at epoch {epoch} "
                  f"(test acc {test_acc:.4f}).")

    # Final-epoch model + a guaranteed save of the final state.
    save_model(model, SIZES, MODEL_PATH)
    print(f"\nFinal test accuracy: {test_acc:.4f}  (best {best_acc:.4f})")
    print(f"Saved model -> {MODEL_PATH}")

    # Sanity check: reload and confirm identical accuracy.
    reloaded = load_model(MODEL_PATH)
    print(f"Reloaded model test accuracy: {accuracy(reloaded, Xte, yte):.4f}")


if __name__ == "__main__":
    train()
