# Phase 1 — Convolutional & Vision Core

Notes on the building blocks of convolutional neural networks and the training tricks that make modern vision models work: convolution, pooling, residual connections, batch normalization, transfer learning, and data augmentation.

> Builds on [`nn_core.md`](./nn_core.md) (backprop, activations, normalization, optimizers). A conv layer is just a weight-shared linear layer; everything about gradients and training carries over.

---

## 1. Convolution

A convolution slides a small learnable **filter** (kernel) across a spatially-structured input, computing a dot product at each location. Unlike a dense layer, weights are **shared** across positions — this gives *translation equivariance* (a feature detector works anywhere in the image) and slashes the parameter count.

### Filter (kernel)

- A filter is a small weight tensor of shape $(C_{\text{out}}, C_{\text{in}}, k_h, k_w)$ — e.g. a $3\times3$ kernel over 3 input channels producing 64 output channels is $(64, 3, 3, 3)$.
- Each output channel is one filter convolved over all input channels, summed, plus a bias. The output is a stack of $C_{\text{out}}$ **feature maps**.
- Early layers learn edges/colors/textures; deeper layers compose these into parts and objects.

### Stride

- **Stride** $s$ is the step size as the filter slides. $s=1$ keeps resolution; $s=2$ downsamples by ~2× (a cheap alternative to pooling).
- Larger stride → smaller output, less compute, coarser spatial detail.

### Padding

- **Padding** adds a border (usually zeros) around the input so the filter can sit at the edges.
- **"valid"** (no padding) shrinks the output; **"same"** padding ($p = (k-1)/2$ for odd $k$, stride 1) keeps output size equal to input.
- Without padding, spatial size shrinks every layer and corner pixels get undersampled.

### Output size & receptive field

The output spatial size along one dimension:

$$o = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1,$$

where $n$ = input size, $k$ = kernel size, $p$ = padding, $s$ = stride.

**Numerical example.** Input $32\times32$, kernel $k=3$, padding $p=1$, stride $s=1$: $o = \lfloor(32 + 2 - 3)/1\rfloor + 1 = 32$ ("same"). With $s=2$: $o = \lfloor(32+2-3)/2\rfloor + 1 = 16$ (halved).

The **receptive field** is the region of the *input* that influences one output unit. It grows as you stack layers — each $3\times3$ conv adds $(k-1)=2$ to the receptive field per layer (more with stride/pooling). Two stacked $3\times3$ convs see a $5\times5$ patch; three see $7\times7$ — with fewer parameters and more nonlinearity than a single $7\times7$ conv. This is why deep stacks of small kernels (VGG-style) beat shallow large ones.

---

## 2. Pooling

Pooling downsamples each feature map, reducing spatial resolution (and compute) while adding a small amount of translation invariance. It has **no learnable parameters**.

- **Max pooling**: take the maximum in each window — keeps the strongest activation, robust to small shifts. The classic default.
- **Average pooling**: take the mean — smoother, retains background context. **Global average pooling** (average each whole feature map to one number) is the standard head before the classifier in modern CNNs, replacing large dense layers.

**Spatial reduction.** A $2\times2$ pool with stride 2 turns $H\times W \to \tfrac{H}{2}\times\tfrac{W}{2}$ (¼ the activations), leaving channel count unchanged. Backprop: max-pool routes the gradient only to the arg-max position; avg-pool spreads it equally over the window.

Modern architectures increasingly replace pooling with **strided convolutions** (learnable downsampling), but pooling remains common and cheap.

---

## 3. ResNet — Skip Connections (tại sao cần thiết / why they're necessary)

**The problem.** Naively stacking more layers makes deep plain networks *worse* — not from overfitting but from an **optimization** failure: gradients degrade flowing back through many layers (vanishing/exploding), and the network struggles to even fit the training set. A 56-layer plain net underperformed a 20-layer one.

**The fix — residual learning.** A ResNet block learns a *residual* $F(x)$ and adds the input back via a **skip (identity) connection**:

$$y = F(x) + x.$$

Why this is the key idea:

- **Easy identity.** If the optimal transform is "do nothing," the block only needs to drive $F(x)\to 0$ — far easier than forcing a stack of layers to learn the identity from scratch. Adding layers can no longer hurt.
- **Gradient highway.** The backward path through the $+x$ term has gradient 1, so $\partial L/\partial x = \partial L/\partial y \,(1 + \partial F/\partial x)$. That additive 1 lets gradients flow directly to early layers, defeating vanishing gradients and enabling 100+ layer networks.
- **Ensemble-like behavior.** The network behaves like an ensemble of many shorter paths, smoothing the loss landscape.

When $F(x)$ changes the channel count or spatial size, the skip uses a $1\times1$ conv (with matching stride) to project $x$ so the shapes add. Skip connections are now ubiquitous — Transformers use the same residual + norm pattern.

---

## 4. Batch Normalization — Covariate Shift & Training Dynamics

(Mechanics and the BatchNorm vs. LayerNorm comparison are in [`nn_core.md`](./nn_core.md) §6; here we focus on *why* it helps.)

**Internal covariate shift (the original motivation).** As earlier layers update, the distribution of inputs to later layers keeps shifting, so each layer chases a moving target. BatchNorm renormalizes each feature to zero mean / unit variance per mini-batch, stabilizing those distributions. The learnable $\gamma, \beta$ let the network undo the normalization if needed.

**What it actually does (the modern view).** Later analysis argued the bigger effect is a **smoother loss landscape** — BatchNorm reduces the Lipschitz constant of the loss and its gradients, so the optimizer can take larger, more stable steps. Either way, the empirical payoff is real.

**Training dynamics & consequences:**

- **Higher learning rates** become safe → faster convergence.
- **Mild regularization** — each sample's normalization depends on the random batch it landed in, injecting noise (so you often need less dropout).
- **Train ≠ eval.** During training it uses *batch* statistics and updates running averages; at inference it uses those frozen running stats. Forgetting `model.eval()` is a classic bug.
- **Batch-size sensitive.** Small batches give noisy statistics and hurt — a key reason LayerNorm/GroupNorm are preferred for sequences and tiny-batch regimes.

---

## 5. Transfer Learning — Freeze / Unfreeze

Reuse a model pretrained on a large dataset (e.g. ImageNet) as a starting point for a new, usually smaller task. Early layers learn generic features (edges, textures) that transfer broadly; later layers are task-specific.

**The workflow:**

1. **Replace the head.** Swap the final classifier for one matching your number of classes.
2. **Freeze the backbone** (set `requires_grad=False`) and train only the new head. Fast, needs little data, low overfitting risk. This is *feature extraction*.
3. **Fine-tune (unfreeze).** Optionally unfreeze some or all backbone layers and continue training with a **small learning rate** (e.g. 10× lower) so pretrained weights are nudged, not destroyed. A common recipe unfreezes top layers first, then progressively deeper ones (*gradual unfreezing*).

**Rules of thumb (data size × similarity to source):**

| Your data | Similar to source | Different from source |
| --------- | ----------------- | --------------------- |
| **Small** | freeze backbone, train head only | freeze early layers, train head (risky) |
| **Large** | fine-tune the whole network | fine-tune all, even from scratch is viable |

Always use a low LR when unfreezing, and remember frozen BatchNorm layers should keep their pretrained running stats (set them to eval mode).

---

## 6. Data Augmentation

Artificially enlarge the training set by applying **label-preserving** transforms, exposing the model to more variation and improving generalization. Applied on-the-fly each epoch (train only — never at eval).

- **RandomCrop** (often with padding): crop a random sub-region, teaching position invariance and discouraging reliance on exact framing. Pad-then-crop keeps the output size fixed.
- **Flip** — `RandomHorizontalFlip` is standard for natural images (a mirrored cat is still a cat). Avoid flips that change meaning (vertical flips of digits, or horizontal flips of text/'6' vs '9').
- **Color/geometry** — jitter brightness/contrast/saturation, small rotations, scaling — simulate lighting and viewpoint changes.
- **Mixup** — train on *convex combinations* of two examples and their labels:

$$\tilde{x} = \lambda x_i + (1-\lambda) x_j, \qquad \tilde{y} = \lambda y_i + (1-\lambda) y_j, \qquad \lambda \sim \text{Beta}(\alpha, \alpha).$$

Mixup smooths decision boundaries, calibrates confidence, and regularizes strongly. **CutMix** is a related variant that pastes a patch of one image onto another and mixes labels by area.

The right augmentations encode your **prior about valid invariances** — pick transforms under which the label genuinely shouldn't change.

---

## Quick Mental-Model Map

| Concept | What it really is |
| ------- | ----------------- |
| Convolution | Weight-shared linear layer → translation equivariance, few params |
| Stride / padding | Control output size; padding preserves edges, stride downsamples |
| Receptive field | Input region one output sees; grows by stacking small kernels |
| Pooling | Param-free spatial downsampling + small shift invariance |
| Skip connection | Additive identity path → gradient highway, enables very deep nets |
| BatchNorm | Per-batch renorm → smoother landscape, higher LR, mild regularizer |
| Transfer learning | Reuse pretrained features; freeze generic, fine-tune specific |
| Data augmentation | Encode label-preserving invariances to fight overfitting |
