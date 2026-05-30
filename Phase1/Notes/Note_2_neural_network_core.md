# Note 2 — Neural Network Core

This note covering the building blocks of neural networks: automatic differentiation, activations, loss functions, optimizers, regularization, and the PyTorch primitives that implement them.

---

## 1. Backpropagation — Chain Rule on a DAG

A neural network is a **computational graph**: a directed acyclic graph (DAG) whose nodes are operations and whose edges carry tensors. The forward pass evaluates nodes in topological order; the backward pass walks the *reverse* topological order, accumulating gradients via the chain rule.

### The core identity

For a scalar loss $L$ and any intermediate value $v$ feeding into successors $u_1, \dots, u_k$, gradients **sum over all outgoing paths**:

$$\frac{\partial L}{\partial v} = \sum_{j} \frac{\partial L}{\partial u_j}\,\frac{\partial u_j}{\partial v}.$$

This multivariate sum (vs. a single product) is the only thing that distinguishes graph backprop from the scalar chain rule. The local factor $\partial u_j / \partial v$ is the **local Jacobian** of each op; the upstream factor $\partial L/\partial u_j$ arrives from downstream.

### Vector-Jacobian products (VJPs)

In practice we never materialize a full Jacobian. Each op exposes a **vector-Jacobian product**: given the upstream gradient $\bar{\mathbf{u}} = \partial L/\partial \mathbf{u}$, it returns $\bar{\mathbf{v}} = J^\top \bar{\mathbf{u}}$. For $\mathbf{u} = W\mathbf{v}$ the VJP is $\bar{\mathbf{v}} = W^\top \bar{\mathbf{u}}$, and the weight gradient is the outer product $\bar{W} = \bar{\mathbf{u}}\,\mathbf{v}^\top$.

### Worked matrix example

A scalar chain hides the only interesting part of a VJP — the transpose. Here is the same idea where every local Jacobian is an actual matrix, so $\bar{\mathbf v} = J^\top \bar{\mathbf u}$ is a genuine matrix–vector product. The graph is a linear layer, an elementwise nonlinearity, then a scalar loss:

$$\mathbf{z} = W\mathbf{x}, \qquad \mathbf{a} = \mathbf{z}\odot\mathbf{z}\;\;(\text{elementwise square}), \qquad L = \tfrac{1}{2}\,\mathbf{a}^\top\mathbf{a}.$$

**Forward** with $W = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix},\; \mathbf{x} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$:

$$\mathbf{z} = W\mathbf{x} = \begin{pmatrix} 3 \\ 1 \end{pmatrix}, \qquad \mathbf{a} = \begin{pmatrix} 9 \\ 1 \end{pmatrix}, \qquad L = \tfrac{1}{2}(81 + 1) = 41.$$

**Local Jacobians.** Each op's Jacobian is a matrix; the elementwise op is *diagonal* (output $i$ depends only on input $i$), the linear op is the *full* weight matrix:

$$J_{L\leftarrow\mathbf{a}} = \mathbf{a}^\top, \qquad J_{\mathbf{a}\leftarrow\mathbf{z}} = \operatorname{diag}(2\mathbf{z}) = \begin{pmatrix} 6 & 0 \\ 0 & 2 \end{pmatrix}, \qquad J_{\mathbf{z}\leftarrow\mathbf{x}} = W.$$

**Backward** — seed $\bar L = 1$ and push it leftward, each op applying its *transposed* Jacobian:

$$\bar{\mathbf a} = \mathbf{a} = \begin{pmatrix} 9 \\ 1 \end{pmatrix}, \qquad \bar{\mathbf z} = \operatorname{diag}(2\mathbf{z})^\top \bar{\mathbf a} = \begin{pmatrix} 6 & 0 \\ 0 & 2 \end{pmatrix}\begin{pmatrix} 9 \\ 1 \end{pmatrix} = \begin{pmatrix} 54 \\ 2 \end{pmatrix},$$

$$\bar{\mathbf x} = W^\top \bar{\mathbf z} = \begin{pmatrix} 1 & 0 \\ 2 & 1 \end{pmatrix}\begin{pmatrix} 54 \\ 2 \end{pmatrix} = \begin{pmatrix} 54 \\ 110 \end{pmatrix}, \qquad \bar W = \bar{\mathbf z}\,\mathbf{x}^\top = \begin{pmatrix} 54 \\ 2 \end{pmatrix}\begin{pmatrix} 1 & 1 \end{pmatrix} = \begin{pmatrix} 54 & 54 \\ 2 & 2 \end{pmatrix}.$$

Note the two patterns that show up in *every* layer: the input gradient uses $W^\top$ (transpose, never the inverse), and the weight gradient is the **outer product** of the upstream gradient with the layer's input.

*Check by substitution:* with $z_1 = x_1 + 2x_2,\; z_2 = x_2$ we have $L = \tfrac{1}{2}(z_1^4 + z_2^4)$, so $\partial L/\partial x_1 = 2z_1^3 = 54$ and $\partial L/\partial x_2 = 2z_1^3\cdot 2 + 2z_2^3 = 108 + 2 = 110$. ✓

Each op needed only its local Jacobian and the gradient handed down from its parent — no global formula, and no full Jacobian ever materialized (the diagonal op stores just $2\mathbf{z}$). That locality is exactly what makes autograd composable.

---

## 2. Autograd Engine — Forward & Backward Passes

An autograd engine records operations during the forward pass into a graph, then replays it backward to compute gradients. Two design axes:

- **Forward pass**: compute output values *and* remember, per node, the inputs (or intermediate results) needed for its backward function.
- **Backward pass**: traverse nodes in reverse topological order, calling each node's VJP and **accumulating** into `grad` (accumulation matters when a tensor is reused along multiple paths).

### Reverse-mode vs. forward-mode

- **Reverse-mode** (backprop): one backward sweep yields the gradient of one scalar w.r.t. *all* inputs. Cost ≈ one extra forward pass. Ideal when outputs (1 scalar loss) ≪ inputs (millions of params). This is what deep learning uses.
- **Forward-mode**: propagates derivatives alongside values; one sweep gives the derivative of *all* outputs w.r.t. *one* input. Efficient when inputs ≪ outputs.

### A minimal autograd `Value` (Karpathy-style micrograd)

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None      # local VJP closure
        self._prev = set(_children)

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad  += out.grad          # d(out)/d(self)  = 1
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad  += other.data * out.grad   # product rule
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0                     # seed
        for v in reversed(topo):            # reverse topological order
            v._backward()
```

The two ideas that make this work: (1) every op stores a `_backward` closure capturing its local gradient rule, and (2) `backward()` topologically sorts the graph so each node fires only after its consumers have accumulated into it.

---

## 3. Activations

Nonlinearities between linear layers are what let networks approximate non-linear functions. Pick by gradient behavior.

| Activation | $f(x)$ | $f'(x)$ | Notes |
| ---------- | ------ | ------- | ----- |
| **ReLU** | $\max(0, x)$ | $\mathbb{1}[x>0]$ | Cheap, sparse, no vanishing for $x>0$; "dying ReLU" for $x<0$. |
| **Sigmoid** | $\dfrac{1}{1+e^{-x}}$ | $f(x)(1-f(x))$ | Squashes to $(0,1)$; saturates → vanishing gradients. Use for binary output, gates. |
| **Tanh** | $\dfrac{e^x-e^{-x}}{e^x+e^{-x}}$ | $1-f(x)^2$ | Zero-centered sigmoid; still saturates. |
| **GELU** | $x\,\Phi(x)$ | smooth | $\Phi$ = standard normal CDF; smooth, non-monotone. Default in Transformers. |

### GELU detail

$$\text{GELU}(x) = x\,\Phi(x) = x \cdot \tfrac{1}{2}\Bigl[1 + \operatorname{erf}\!\bigl(x/\sqrt{2}\bigr)\Bigr].$$

A common fast approximation:

$$\text{GELU}(x) \approx 0.5\,x\Bigl(1 + \tanh\bigl[\sqrt{2/\pi}\,(x + 0.044715\,x^3)\bigr]\Bigr).$$

### Numerical example

At $x = 1$: $\;\text{ReLU}=1$, $\;\sigma(1)=\frac{1}{1+e^{-1}}\approx 0.731$, $\;\text{GELU}(1)=1\cdot\Phi(1)\approx 0.841$. At $x=-1$: $\;\text{ReLU}=0$, $\;\sigma(-1)\approx 0.269$, $\;\text{GELU}(-1)\approx -0.159$ (note: slightly negative — GELU passes a little signal where ReLU hard-zeros).

---

## 4. Loss Functions

A loss maps (prediction, target) → scalar to minimize. The gradient w.r.t. the *pre-activation logits* is what gets backpropagated — and for the canonical pairings below it simplifies beautifully.

### Mean Squared Error (MSE) — regression

$$\text{MSE} = \frac{1}{n}\sum_i (\hat{y}_i - y_i)^2, \qquad \frac{\partial \text{MSE}}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i).$$

### Binary Cross-Entropy (BCE) — single label / multi-label

With prediction $\hat{y} = \sigma(z)$ (sigmoid of logit $z$):

$$\text{BCE} = -\bigl[y\log\hat{y} + (1-y)\log(1-\hat{y})\bigr], \qquad \boxed{\frac{\partial \text{BCE}}{\partial z} = \hat{y} - y.}$$

The sigmoid and the log conveniently cancel — implement with `BCEWithLogitsLoss` for numerical stability (log-sum-exp), never `sigmoid` + `BCELoss`.

### Cross-Entropy (CE) — multi-class

With logits $\mathbf{z}$, softmax $\hat{y}_k = \dfrac{e^{z_k}}{\sum_j e^{z_j}}$, and one-hot true class $y$:

$$\text{CE} = -\sum_k y_k \log \hat{y}_k = -\log \hat{y}_{\text{true}}, \qquad \boxed{\frac{\partial \text{CE}}{\partial z_k} = \hat{y}_k - y_k.}$$

Same elegant `softmax − onehot` form as BCE. PyTorch's `nn.CrossEntropyLoss` takes **raw logits** and applies log-softmax internally.

### Numerical example (CE)

Logits $\mathbf{z} = (2, 1, 0)$, true class $= 0$. Softmax: $e^2{:}e^1{:}e^0 = 7.39{:}2.72{:}1.00$, sum $= 11.11$, so $\hat{y} = (0.665, 0.245, 0.090)$. Loss $= -\log 0.665 = 0.408$. Gradient on logits $= \hat{y} - (1,0,0) = (-0.335,\; 0.245,\; 0.090)$ — pushes the true logit up, the others down.

---

## 5. Optimizers

Given gradient $g_t = \nabla_\theta L$, each optimizer defines the update rule. The progression below is roughly "add one fix at a time."

### SGD

$$\theta_{t+1} = \theta_t - \eta\, g_t.$$

Simple, but oscillates in ravines and crawls on plateaus.

### Momentum

Accumulate an exponentially-decayed velocity to smooth the trajectory:

$$v_t = \beta v_{t-1} + g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_t \quad (\beta \approx 0.9).$$

Damps oscillation across steep directions, accelerates along consistent ones.

### Adam (Adaptive per-parameter steps)

Track first moment (mean) and second moment (uncentered variance) of gradients:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,$$

bias-correct (they start at 0):

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t},$$

then step:

$$\theta_{t+1} = \theta_t - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.$$

Defaults: $\beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$. Each parameter gets its own effective learning rate scaled by its gradient history.

### AdamW — decoupled weight decay

Adam folds L2 regularization into the gradient, which interacts badly with the adaptive denominator. **AdamW** applies weight decay *directly to the weights*, decoupled from the moment estimates:

$$\theta_{t+1} = \theta_t - \eta\Bigl(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\,\theta_t\Bigr).$$

This is the de-facto standard for training Transformers. The $\lambda\theta_t$ term decays weights at a rate independent of their gradient magnitude.

---

## 6. Regularization

Techniques that reduce overfitting or stabilize training.

### Dropout

During training, zero each activation independently with probability $p$, then scale survivors by $1/(1-p)$ (**inverted dropout**) so the expected activation is unchanged and inference needs no rescaling.

$$\tilde{a}_i = \frac{m_i}{1-p}\,a_i, \qquad m_i \sim \text{Bernoulli}(1-p).$$

Acts like training an ensemble of subnetworks. Disabled at eval time (`model.eval()`).

### Batch Normalization

Normalize each feature across the **batch** dimension, then learn a scale $\gamma$ and shift $\beta$:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y_i = \gamma\,\hat{x}_i + \beta.$$

- $\mu_B, \sigma_B^2$ are batch statistics during training; running averages are kept for inference.
- Smooths the loss landscape, allows higher learning rates. **Batch-size dependent** and awkward for sequences/RNNs.

### Layer Normalization

Normalize across the **feature** dimension, *per sample* — independent of batch size:

$$\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad \mu, \sigma^2 \text{ over features of one sample.}$$

Same train/eval behavior, no running stats. The default normalization in Transformers and RNNs.

| | Normalize over | Batch-dependent | Train≠Eval |
| --- | --- | --- | --- |
| **BatchNorm** | batch (per feature) | yes | yes (running stats) |
| **LayerNorm** | features (per sample) | no | no |

---

## 7. PyTorch Core

### Tensor

The n-dimensional array + autograd metadata.

```python
import torch
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()      # builds the graph
y.backward()            # reverse-mode autodiff
print(x.grad)           # tensor([2., 4., 6.])  ==  d(sum x^2)/dx = 2x
```

- `requires_grad=True` flags a leaf tensor to track gradients.
- Operations on tracked tensors build the graph dynamically (define-by-run).
- `.grad` **accumulates** — call `optimizer.zero_grad()` each step.
- `with torch.no_grad():` disables graph building (inference, manual updates).
- `.detach()` returns a tensor sharing storage but cut from the graph.

### autograd

- `y.backward()` requires `y` scalar (or pass a `gradient=` vector for the VJP seed). It populates `.grad` on every leaf with `requires_grad=True`.
- `torch.autograd.grad(outputs, inputs)` computes gradients functionally without mutating `.grad` — useful for higher-order derivatives.
- The graph is freed after `backward()` unless `retain_graph=True`.

### nn.Module

The container for parameters + forward logic. Subclass it, register submodules in `__init__`, define `forward`.

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, p=0.1):
        super().__init__()
        self.fc1  = nn.Linear(d_in, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)
        self.drop = nn.Dropout(p)
        self.fc2  = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = self.drop(F.gelu(self.norm(self.fc1(x))))
        return self.fc2(x)               # raw logits → CrossEntropyLoss
```

The standard training loop ties every section above together:

```python
model = MLP(784, 256, 10)
opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()          # expects raw logits

model.train()
for x, y in loader:
    opt.zero_grad()                      # clear accumulated grads
    logits = model(x)                    # forward pass (builds graph)
    loss   = loss_fn(logits, y)
    loss.backward()                      # backward pass (autograd)
    opt.step()                           # optimizer update
```

- `model.parameters()` yields every registered tensor with `requires_grad`.
- `model.train()` / `model.eval()` toggle Dropout and BatchNorm behavior.
- `state_dict()` serializes parameters + buffers (running stats) for checkpointing.
