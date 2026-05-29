# Note 1 — Math & Python Foundations

This note will covering the mathematical prerequisites for ML: linear algebra, calculus, probability, and optimization basics.

---

## 1. Linear Algebra

### Vectors

- A vector $\mathbf{v} \in \mathbb{R}^n$ is an ordered tuple of $n$ real numbers; geometrically a point or arrow from the origin.
- **Norm** ($L_2$): $\mathbf{v}_2 = \sqrt{\sum_i v_i^2}$. Other norms: $L^1$ (sum of absolute values), $L^\infty$ (max).
- **Dot product**: $\mathbf{u} \cdot \mathbf{v} = \sum_i u_i v_i = \mathbf{u} \mathbf{v} \cos\theta$. Zero ⇒ orthogonal.
- **Linear independence**: $\mathbf{v}_1, \dots, \mathbf{v}_k$ are independent iff no non-trivial linear combination equals $\mathbf{0}$.
- **Basis / span**: a basis is a maximal independent set; its span is the whole space.

### Matrices

- $A \in \mathbb{R}^{m \times n}$ maps $\mathbb{R}^n \to \mathbb{R}^m$ linearly: $\mathbf{y} = A\mathbf{x}$.
- **Rank**: dimension of the column space. Full rank ⇒ injective (if tall) or surjective (if wide).
- **Inverse**: $A^{-1}$ exists iff $A$ is square and full-rank. $A A^{-1} = I$.
- **Transpose**: $(AB)^\top = B^\top A^\top$. **Symmetric** if $A = A^\top$; **orthogonal** if $A^\top A = I$.
- **Trace**: $\text{tr}(A) = \sum_i A_{ii}$. Cyclic: $\text{tr}(ABC) = \text{tr}(BCA)$.
- **Determinant**: signed volume scaling factor; $\det(AB) = \det(A)\det(B)$.

### Eigenvalues & Eigenvectors

- $A \mathbf{v} = \lambda \mathbf{v}$, with $\mathbf{v} \ne \mathbf{0}$. $\lambda$ is the eigenvalue, $\mathbf{v}$ the eigenvector.
- Found by solving $\det(A - \lambda I) = 0$ (characteristic polynomial).
- For symmetric $A$: eigenvalues are real, eigenvectors orthogonal — gives spectral decomposition $A = Q \Lambda Q^\top$.
- $\text{tr}(A) = \sum_i \lambda_i$, $\det(A) = \prod_i \lambda_i$.
- **Positive (semi-)definite**: all $\lambda_i > 0$ (resp. $\ge 0$). Critical for covariance matrices, Hessians, kernels.

### Singular Value Decomposition (SVD)

- Any $A \in \mathbb{R}^{m \times n}$ factors as $A = U \Sigma V^\top$ where:
  - $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$ are orthogonal.
  - $\Sigma$ is diagonal with non-negative **singular values** $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$.
- $\sigma_i^2$ are eigenvalues of $A^\top A$ (and of $A A^\top$).
- **Low-rank approximation**: truncate to top-$k$ singular values — Eckart–Young says this minimizes Frobenius error. Used in PCA, image compression, recommender systems, latent semantic analysis.
- **Condition number**: $\sigma_{\max} / \sigma_{\min}$ — measures numerical sensitivity.

---

## 2. Calculus

### Partial Derivatives

- For $f: \mathbb{R}^n \to \mathbb{R}$, the partial $\partial f / \partial x_i$ holds all other variables fixed.
- **Gradient**: $\nabla f = (\partial f/\partial x_1, \dots, \partial f/\partial x_n)^\top$. Points in the direction of steepest ascent; magnitude is the rate.
- **Directional derivative** along unit $\mathbf{u}$: $\nabla f \cdot \mathbf{u}$.

**Numerical example (scalar field).** Let $f(x_1, x_2) = x_1^2 + 3x_1 x_2 + x_2^3$.

$$\frac{\partial f}{\partial x_1} = 2x_1 + 3x_2, \qquad \frac{\partial f}{\partial x_2} = 3x_1 + 3x_2^2.$$

At $\mathbf{x} = (1, 2)^\top$:

$$\nabla f = \begin{pmatrix} 2(1) + 3(2) \\ 3(1) + 3(2)^2 \end{pmatrix} = \begin{pmatrix} 8 \\ 15 \end{pmatrix}.$$

The directional derivative along the unit vector $\mathbf{u} = \tfrac{1}{\sqrt{2}}(1, 1)^\top$ is

$$\nabla f \cdot \mathbf{u} = \frac{1}{\sqrt{2}}(8 + 15) = \frac{23}{\sqrt{2}} \approx 16.26.$$

**Numerical example (quadratic form).** For $f(\mathbf{x}) = \tfrac{1}{2}\mathbf{x}^{\top} A \mathbf{x} - \mathbf{b}^{\top} \mathbf{x}$ with symmetric $A$, the gradient is $\nabla f = A\mathbf{x} - \mathbf{b}$. With

$$A = \begin{pmatrix} 4 & 1 \\ 1 & 3 \end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}, \quad \mathbf{x} = \begin{pmatrix} 1 \\ 1 \end{pmatrix},$$

$$\nabla f = \begin{pmatrix} 4 & 1 \\ 1 & 3 \end{pmatrix}\begin{pmatrix} 1 \\ 1 \end{pmatrix} - \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} 5 \\ 4 \end{pmatrix} - \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} 4 \\ 2 \end{pmatrix}.$$

The minimum is at $\mathbf{x}^{*} = A^{-1}\mathbf{b}$, where the gradient vanishes.

### Chain Rule

- Single-variable: $(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$.
- Multivariate / vector form: if $\mathbf{y} = g(\mathbf{x})$ and $z = f(\mathbf{y})$, then
$$\frac{\partial z}{\partial x_i} = \sum_j \frac{\partial z}{\partial y_j} \frac{\partial y_j}{\partial x_i}.$$
- In matrix form: $\nabla_{\mathbf{x}} z = J_g(\mathbf{x})^\top  \nabla_{\mathbf{y}} z$.
- This is the engine of **backpropagation**: gradients flow backwards through a composition of layers.

**Numerical example (vector chain rule).** Let $\mathbf{y} = g(\mathbf{x})$ map $\mathbb{R}^2 \to \mathbb{R}^2$ and $z = f(\mathbf{y})$ map $\mathbb{R}^2 \to \mathbb{R}$:

$$g(\mathbf{x}) = \begin{pmatrix} x_1^2 \\ x_1 x_2 \end{pmatrix}, \qquad z = y_1 + y_2^2.$$

The Jacobian of $g$ and the gradient of $f$ are

$$J_g = \begin{pmatrix} 2x_1 & 0 \\ x_2 & x_1 \end{pmatrix}, \qquad \nabla_{\mathbf{y}} z = \begin{pmatrix} 1 \\ 2y_2 \end{pmatrix}.$$

At $\mathbf{x} = (2, 3)^\top$ we get $\mathbf{y} = (4, 6)^\top$, so

$$J_g = \begin{pmatrix} 4 & 0 \\ 3 & 2 \end{pmatrix}, \qquad \nabla_{\mathbf{y}} z = \begin{pmatrix} 1 \\ 12 \end{pmatrix}.$$

Applying $\nabla_{\mathbf{x}} z = J_g^\top \nabla_{\mathbf{y}} z$:

$$\nabla_{\mathbf{x}} z = \begin{pmatrix} 4 & 3 \\ 0 & 2 \end{pmatrix}\begin{pmatrix} 1 \\ 12 \end{pmatrix} = \begin{pmatrix} 40 \\ 24 \end{pmatrix}.$$

*Check by substitution:* $z = x_1^2 + (x_1 x_2)^2$, so $\partial z/\partial x_1 = 2x_1 + 2x_1 x_2^2 = 4 + 2(2)(9) = 40$ and $\partial z/\partial x_2 = 2x_1^2 x_2 = 2(4)(3) = 24$. ✓

**Numerical example (one linear layer).** A layer computes $\mathbf{y} = W\mathbf{x}$ and a scalar loss $L = \tfrac{1}{2}\mathbf{y}^\top \mathbf{y}$, so $\nabla_{\mathbf{y}} L = \mathbf{y}$. With

$$W = \begin{pmatrix} 1 & 2 \\ 0 & 1 \end{pmatrix}, \quad \mathbf{x} = \begin{pmatrix} 1 \\ 1 \end{pmatrix} \;\Rightarrow\; \mathbf{y} = \begin{pmatrix} 3 \\ 1 \end{pmatrix}.$$

Since $J_g = W$, the gradient w.r.t. the input is

$$\nabla_{\mathbf{x}} L = W^\top \mathbf{y} = \begin{pmatrix} 1 & 0 \\ 2 & 1 \end{pmatrix}\begin{pmatrix} 3 \\ 1 \end{pmatrix} = \begin{pmatrix} 3 \\ 7 \end{pmatrix},$$

and the gradient w.r.t. the weights is the outer product $\nabla_W L = \nabla_{\mathbf{y}} L \, \mathbf{x}^\top = \begin{pmatrix} 3 \\ 1 \end{pmatrix}\begin{pmatrix} 1 & 1 \end{pmatrix} = \begin{pmatrix} 3 & 3 \\ 1 & 1 \end{pmatrix}$ — exactly the local-gradient × upstream-gradient pattern used in backprop.

### Jacobian

- For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is the $m \times n$ matrix:
$$J_{ij} = \partial f_i / \partial x_j.$$
- Generalizes the derivative to vector-valued functions. Determines local linear behavior: $\mathbf{f}(\mathbf{x} + \delta) \approx \mathbf{f}(\mathbf{x}) + J\delta$.
- **Hessian** $H_{ij} = \partial^2 f / \partial x_i \partial x_j$ — the Jacobian of the gradient; encodes curvature. Symmetric for $C^2$ functions (Schwarz).
- Hessian positive-definite ⇒ local minimum; indefinite ⇒ saddle point.

**Numerical example (Jacobian).** For $\mathbf{f}(\mathbf{x}) = (x_1^2 x_2,\; \sin x_1 + x_2)^\top$,

$$J = \begin{pmatrix} 2x_1 x_2 & x_1^2 \\ \cos x_1 & 1 \end{pmatrix}.$$

At $\mathbf{x} = (0, 5)^\top$: $\;J = \begin{pmatrix} 0 & 0 \\ 1 & 1 \end{pmatrix}$, so a small step $\delta = (0.1, 0)^\top$ gives $\mathbf{f}(\mathbf{x} + \delta) \approx \mathbf{f}(\mathbf{x}) + J\delta = \mathbf{f}(\mathbf{x}) + (0,\; 0.1)^\top$.

**Numerical example (Hessian).** For $f(\mathbf{x}) = x_1^2 + x_1 x_2 + 2x_2^2$,

$$H = \begin{pmatrix} \partial^2 f/\partial x_1^2 & \partial^2 f/\partial x_1 \partial x_2 \\ \partial^2 f/\partial x_2 \partial x_1 & \partial^2 f/\partial x_2^2 \end{pmatrix} = \begin{pmatrix} 2 & 1 \\ 1 & 4 \end{pmatrix}.$$

Leading minors $2 > 0$ and $\det H = 8 - 1 = 7 > 0$, so $H$ is positive-definite — the unique critical point is a minimum.

---

## 3. Probability

### Distributions

- **Discrete**: PMF $p(x)$ with $\sum_x p(x) = 1$. Common: Bernoulli, Binomial, Categorical, Poisson.
- **Continuous**: PDF $p(x)$ with $\int p(x)dx = 1$. Common: Uniform, Gaussian, Exponential, Beta, Dirichlet.
- **Gaussian**: $\mathcal{N}(\mu, \sigma^2)$, density $\frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$. Multivariate: $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$.
- **Expectation**: $\mathbb{E}[X] = \sum_x xp(x)$ or $\int xp(x)dx$. Linear: $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$.
- **Variance**: $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$.
- **Bayes' rule**: $p(A \mid B) = \frac{p(B \mid A)  p(A)}{p(B)}$ — the backbone of probabilistic inference.

### Entropy

- Measures uncertainty of a distribution $p$:
$$H(p) = -\sum_x p(x) \log p(x) \quad \text{(or } -\int p(x) \log p(x)dx\text{)}.$$
- Units: bits (log base 2) or nats (log base $e$).
- Maximum at uniform distribution; zero when $p$ is a delta.
- **Cross-entropy**: $H(p, q) = -\sum_x p(x) \log q(x)$. Equal to $H(p)$ only when $q = p$. Standard loss for classification.

### KL Divergence

- $$D_{\text{KL}}(p  q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p).$$
- **Properties**: $D_{\text{KL}} \ge 0$, equals 0 iff $p = q$. **Asymmetric** — $D_{\text{KL}}(p  q) \ne D_{\text{KL}}(q  p)$ in general. Not a true metric.
- Interpretation: extra bits needed to encode samples from $p$ using a code optimized for $q$.
- Appears in: variational inference (ELBO), VAEs, policy-gradient regularization (TRPO/PPO), distillation.

---

## 4. Optimization

### Gradient Descent

- Iterate: $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta  \nabla L(\boldsymbol{\theta}_t)$, where $\eta$ is the **learning rate**.
- Moves opposite to the gradient — locally steepest descent.
- **Variants**:
  - *Batch GD*: full dataset per step — accurate, slow.
  - *SGD*: one (or a few) samples — noisy, fast, often generalizes better.
  - *Mini-batch SGD*: the standard middle ground.
  - *Momentum*: accumulate a velocity term to damp oscillations and accelerate along flat directions.
  - *Adam / RMSprop*: per-parameter adaptive step sizes via running estimates of gradient moments.
- **Learning rate** is the most important hyperparameter — too large diverges, too small stalls. Warmup + decay (cosine, step) is standard.

### Convexity

- A set $C$ is **convex** if $\lambda x + (1-\lambda) y \in C$ for all $x, y \in C$, $\lambda \in [0, 1]$.
- A function $f$ is **convex** if its epigraph is convex; equivalently:
$$f(\lambda x + (1-\lambda) y) \le \lambda f(x) + (1-\lambda) f(y).$$
- Twice-differentiable $f$ is convex iff Hessian is positive semi-definite everywhere.
- **Strict** convexity ⇒ at most one minimum. **Strong** convexity ($f - \tfrac{\mu}{2}x^2$ is convex) ⇒ linear convergence under GD.
- **Why it matters**: for convex objectives, any local minimum is global, and gradient descent provably converges. Examples: linear/logistic regression, SVMs (with hinge loss), LASSO.
- **Non-convex** (most deep learning losses): no global guarantees, but in practice SGD finds good minima — empirical phenomenon driven by overparameterization, implicit regularization, and benign loss landscapes.

---

## Quick Mental-Model Map


| Concept          | Why it matters in ML                                 |
| ---------------- | ---------------------------------------------------- |
| Eigen / SVD      | PCA, low-rank methods, stability analysis            |
| Jacobian         | Backprop, normalizing flows (log-det), sensitivity   |
| KL divergence    | Variational methods, distillation, RL regularization |
| Gradient descent | Every neural network ever                            |
| Convexity        | When optimization is "safe"; baseline guarantees     |


