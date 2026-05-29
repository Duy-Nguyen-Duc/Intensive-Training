"""Train a from-scratch MLP on MNIST using the scalar ``Value`` autograd engine.

This builds ``Neuron`` -> ``Layer`` -> ``MLP`` on top of ``micrograd_value.Value``
(see ``nn_core.md`` §7 for the PyTorch equivalent) and trains with softmax
cross-entropy + mini-batch SGD.

A BIG caveat on speed
---------------------
``Value`` is a *scalar* engine: every multiply/add allocates a graph node in
pure Python. A single forward pass of one neuron over a 784-pixel image builds
~1.5k nodes, so full MNIST ([784, ...] over 60k images) is utterly impractical
here. To keep this runnable as a learning exercise we:

  * 4x4 average-pool each image  28x28 -> 7x7  (784 -> 49 inputs),
  * train on a small SUBSET (a few hundred images),
  * use a small network and a handful of epochs.

(For reference: at [196, 32, 10] over 500 images, one epoch takes ~9 min on the
scalar engine. The shrunk config below trades resolution for a demo that
finishes in a few minutes while still climbing well above the 10% baseline.)

This is enough to watch the loss fall and accuracy rise — the point is the
mechanics, not a competitive model. For real training, use the PyTorch path.

Run:  python3 mlp_mnist.py
"""

from __future__ import annotations

import random

import numpy as np
from torchvision import datasets

from micrograd_value import Value


# ----------------------------------------------------------------------------
# Network: Neuron -> Layer -> MLP  (mirrors Karpathy's micrograd.nn)
# ----------------------------------------------------------------------------

class Module:
    """Base class providing ``parameters()`` and ``zero_grad()``."""

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, n_in, nonlin=True):
        # Small random init; scale ~ 1/sqrt(fan_in) keeps pre-activations sane.
        scale = n_in ** -0.5
        self.w = [Value(random.uniform(-1, 1) * scale) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        # w . x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, n_in, n_out, nonlin=True):
        self.neurons = [Neuron(n_in, nonlin=nonlin) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """A multi-layer perceptron. ``sizes`` = [n_in, h1, ..., n_out]."""

    def __init__(self, sizes):
        self.layers = []
        for i in range(len(sizes) - 1):
            # Last layer is linear (raw logits); hidden layers use ReLU.
            last = i == len(sizes) - 2
            self.layers.append(Layer(sizes[i], sizes[i + 1], nonlin=not last))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ----------------------------------------------------------------------------
# Loss: softmax cross-entropy over a list of logit Values
# ----------------------------------------------------------------------------

def softmax_cross_entropy(logits, target):
    """CE loss for one example. ``logits``: list[Value]; ``target``: int class.

    Numerically stabilized by subtracting max logit before exp (log-sum-exp).
    Gradient on the logits collapses to ``softmax - onehot`` (see nn_core.md §4).
    """
    max_logit = max(l.data for l in logits)
    exps = [(l - max_logit).exp() for l in logits]
    denom = sum(exps, Value(0.0))
    # -log p_target  =  -log( exp(z_t)/sum exp )
    log_prob = (exps[target] / denom).log()
    return -log_prob


# ----------------------------------------------------------------------------
# Data: load MNIST, average-pool to 14x14, normalize, subsample
# ----------------------------------------------------------------------------

def avg_pool(img, k=4):
    """28x28 uint8 -> (28/k)x(28/k) float in [0,1] via kxk mean pooling."""
    s = 28 // k
    x = np.asarray(img, dtype=np.float32) / 255.0          # (28, 28)
    x = x.reshape(s, k, s, k).mean(axis=(1, 3))            # (s, s)
    return x.reshape(-1).tolist()                          # s*s vector


def load_mnist(n_train, n_test, pool=4, seed=0):
    train = datasets.MNIST(root="./data", train=True, download=True)
    test = datasets.MNIST(root="./data", train=False, download=True)

    rng = random.Random(seed)
    tr_idx = rng.sample(range(len(train)), n_train)
    te_idx = rng.sample(range(len(test)), n_test)

    Xtr = [avg_pool(train[i][0], pool) for i in tr_idx]
    ytr = [train[i][1] for i in tr_idx]
    Xte = [avg_pool(test[i][0], pool) for i in te_idx]
    yte = [test[i][1] for i in te_idx]
    return Xtr, ytr, Xte, yte


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def accuracy(model, X, y):
    correct = 0
    for xi, yi in zip(X, y):
        logits = model([Value(v) for v in xi])
        pred = max(range(len(logits)), key=lambda k: logits[k].data)
        correct += int(pred == yi)
    return correct / len(X)


# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------

def train():
    random.seed(1337)

    # Keep these small — scalar autograd is slow (see module docstring).
    N_TRAIN, N_TEST = 300, 200
    POOL = 4                            # 28x28 -> 7x7 = 49 inputs
    EPOCHS = 15
    BATCH = 16
    LR = 0.2

    print("Loading MNIST (downsampling 28x28 -> 7x7)...")
    Xtr, ytr, Xte, yte = load_mnist(N_TRAIN, N_TEST, pool=POOL)
    n_in = len(Xtr[0])

    model = MLP([n_in, 32, 10])         # 49 -> 32 (ReLU) -> 10 logits
    n_params = len(model.parameters())
    print(f"MLP [{n_in}, 32, 10] with {n_params} parameters")
    print(f"train={N_TRAIN}  test={N_TEST}  batch={BATCH}  lr={LR}\n")

    idx = list(range(N_TRAIN))
    for epoch in range(1, EPOCHS + 1):
        random.shuffle(idx)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N_TRAIN, BATCH):
            batch = idx[start:start + BATCH]

            # --- forward: accumulate loss over the mini-batch ---
            losses = []
            for i in batch:
                logits = model([Value(v) for v in Xtr[i]])
                losses.append(softmax_cross_entropy(logits, ytr[i]))
            loss = sum(losses, Value(0.0)) / len(batch)

            # --- backward ---
            model.zero_grad()           # reset .grad (it accumulates!)
            loss.backward()

            # --- SGD update ---
            for p in model.parameters():
                p.data -= LR * p.grad

            epoch_loss += loss.data
            n_batches += 1

        train_acc = accuracy(model, Xtr, ytr)
        test_acc = accuracy(model, Xte, yte)
        print(f"epoch {epoch:2d} | loss {epoch_loss / n_batches:.4f} "
              f"| train acc {train_acc:.3f} | test acc {test_acc:.3f}")

    print("\nDone. (Subset + scalar engine — accuracy is illustrative only.)")


if __name__ == "__main__":
    train()
