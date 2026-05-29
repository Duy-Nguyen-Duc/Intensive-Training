from __future__ import annotations

import math


class Value:
    """A scalar node in the autograd graph."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        # Internal autograd bookkeeping.
        self._backward = lambda: None   # local VJP closure; no-op for leaves
        self._prev = set(_children)     # the Values this one was built from
        self._op = _op                  # the op label, for debugging/repr

    # ----- core ops -------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1; chain in out.grad.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # Product rule: d(out)/d(self) = other, and vice versa.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supports int/float powers"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            # d(x**n)/dx = n * x**(n-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d(tanh)/dx = 1 - tanh(x)^2
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            # d(exp)/dx = exp(x) = out.data
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            # d(log x)/dx = 1/x
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            # d(relu)/dx = 1 if x > 0 else 0
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    # ----- derived ops (built on the core) --------------------------------

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    # Reflected operators so ``2 + x`` / ``2 * x`` also work.
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return other * self ** -1

    # ----- automated backprop --------------------------------------------

    def backward(self):
        """Reverse-mode autodiff: fill ``.grad`` for every node feeding ``self``."""
        # Build a topological ordering of the graph (children before parents).
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Seed: d(self)/d(self) = 1, then replay closures in reverse.
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # ----- niceties -------------------------------------------------------

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


if __name__ == "__main__":
    # A single neuron: out = tanh(w1*x1 + w2*x2 + b)
    x1, x2 = Value(2.0), Value(0.0)
    w1, w2 = Value(-3.0), Value(1.0)
    b = Value(6.881373587019543)

    n = x1 * w1 + x2 * w2 + b
    o = n.tanh()
    o.backward()

    print(f"output      o = {o.data:.4f}")
    print(f"dO/dx1        = {x1.grad:.4f}")
    print(f"dO/dw1        = {w1.grad:.4f}")
    print(f"dO/dx2        = {x2.grad:.4f}")
    print(f"dO/dw2        = {w2.grad:.4f}")
    print(f"dO/db         = {b.grad:.4f}")

    # Sanity check that exp + the derived ops compose and differentiate.
    a = Value(2.0)
    y = (a.exp() + 1) / 2        # uses exp, __radd__, __truediv__
    y.backward()
    # dy/da = exp(a)/2 = e^2 / 2 ~= 3.6945
    print(f"\ny = (exp(a)+1)/2 = {y.data:.4f},  dy/da = {a.grad:.4f}")
