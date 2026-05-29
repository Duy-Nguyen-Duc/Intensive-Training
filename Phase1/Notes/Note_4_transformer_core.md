# Phase 1 — Transformer Core

Notes on the attention mechanism and the Transformer architecture: scaled dot-product attention, multi-head attention, positional encoding, the block structure, the three architecture families, tokenization, and causal masking.


---

## 1. Scaled Dot-Product Attention — Q / K / V from scratch

Attention lets every token **look at** every other token and pull in a weighted mix of their information. Each token produces three vectors via learned projections of its embedding $\mathbf{x}$:

- **Query** $\mathbf{q} = \mathbf{x} W_Q$ — "what am I looking for?"
- **Key** $\mathbf{k} = \mathbf{x} W_K$ — "what do I offer?"
- **Value** $\mathbf{v} = \mathbf{x} W_V$ — "what I actually pass on if attended to."

Stacking the per-token vectors into matrices $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{n \times d_k}$, $V \in \mathbb{R}^{n \times d_v}$ for a sequence of $n$ tokens:

$$\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.$$

Step by step:

1. **Scores** $S = QK^\top$ — $S_{ij}$ = dot product of query $i$ with key $j$ = how relevant token $j$ is to token $i$. Shape $n \times n$.
2. **Scale** by $1/\sqrt{d_k}$. Without it, for large $d_k$ the dot products grow $\sim\sqrt{d_k}$ in magnitude, pushing softmax into saturated regions where gradients vanish.
3. **Softmax** over each row → attention weights that sum to 1 (a probability distribution over the keys).
4. **Weighted sum** of values: output row $i = \sum_j A_{ij}\mathbf{v}_j$.

### Numerical example (1 query, 2 keys, $d_k = 2$)

$\mathbf{q} = (1, 0)$, $K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $V = \begin{pmatrix} 10 \\ 20 \end{pmatrix}$.

Scores $\mathbf{q}K^\top = (1, 0)$; scaled by $1/\sqrt{2}$: $(0.707, 0)$. Softmax: $\frac{(e^{0.707}, e^0)}{e^{0.707}+e^0} = (0.670, 0.330)$. Output $= 0.670(10) + 0.330(20) = 13.3$ — the query leans toward the first value because its key aligned with the query.

---

## 2. Multi-Head Attention — parallel heads

A single attention computes one kind of relationship. **Multi-head attention** runs $h$ attention operations in parallel, each with its own learned $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$ projecting into a smaller subspace of size $d_k = d_{\text{model}} / h$. Each head can specialize (one tracks syntax, another coreference, etc.).

$$\text{head}_i = \text{Attention}(X W_Q^{(i)}, X W_K^{(i)}, X W_V^{(i)}),$$
$$\text{MultiHead}(X) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O.$$

- Concatenating $h$ heads of width $d_v = d_{\text{model}}/h$ restores width $d_{\text{model}}$; the output projection $W_O$ mixes information across heads.
- **Cost is roughly the same** as one full-width head — the per-head dimension shrinks by $h$, so total compute is preserved while gaining representational diversity.

**Example.** $d_{\text{model}} = 512$, $h = 8$ → each head works in $d_k = d_v = 64$. Eight $64$-dim outputs concatenate back to $512$.

---

## 3. Positional Encoding — sinusoidal vs. learned

Attention is **permutation-invariant**: shuffle the input tokens and the outputs shuffle identically — it has no inherent notion of order. We must inject position information by adding a position vector to each token embedding.

### Sinusoidal (fixed)

The original Transformer uses fixed sinusoids of geometrically-spaced frequencies:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right).$$

- No parameters; **extrapolates** to sequence lengths longer than seen in training.
- Relative positions are recoverable by a linear transform: $PE_{pos+k}$ is a fixed linear function of $PE_{pos}$, which helps the model attend by relative offset.

### Learned

A trainable embedding table indexed by position (used by BERT, GPT-2). Flexible and often slightly better in-distribution, but **capped at the max trained length** and uses parameters.

### Modern variants

Relative position biases and **RoPE** (rotary embeddings — rotate Q/K by a position-dependent angle) now dominate LLMs; RoPE encodes relative position directly in the dot product and extrapolates better.

| | Params | Extrapolates | Used by |
| --- | --- | --- | --- |
| Sinusoidal | none | yes | original Transformer |
| Learned | yes | no (fixed max len) | BERT, GPT-2 |
| RoPE / relative | small/none | yes (better) | LLaMA, GPT-NeoX, most modern LLMs |

---

## 4. Feed-Forward + Residual + LayerNorm — the block

A Transformer block wraps attention and a position-wise feed-forward network, each inside a **residual connection** followed by **LayerNorm**.

**Position-wise FFN** — applied independently to each token, expanding then contracting:

$$\text{FFN}(\mathbf{x}) = \sigma(\mathbf{x} W_1 + \mathbf{b}_1)\, W_2 + \mathbf{b}_2,$$

with hidden width typically $4\times d_{\text{model}}$ and $\sigma$ = GELU. Attention mixes information *across* tokens; the FFN processes each token's mixed representation *individually* (most parameters live here).

**Residual + Norm.** Each sublayer (attention, FFN) is wrapped:

- **Post-LN** (original): $\;\text{out} = \text{LayerNorm}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))$.
- **Pre-LN** (modern, more stable for deep stacks): $\;\text{out} = \mathbf{x} + \text{Sublayer}(\text{LayerNorm}(\mathbf{x}))$.

Residuals give the gradient highway (see [`cnn_core.md`](./cnn_core.md) §3); LayerNorm (not BatchNorm — normalization is per-token over features, batch-independent; see [`nn_core.md`](./nn_core.md) §6) stabilizes the scale of activations. A full block:

```
x = x + MultiHeadAttention(LayerNorm(x))   # mix across tokens
x = x + FFN(LayerNorm(x))                   # process each token
```

---

## 5. Architecture Families — Encoder-only, Decoder-only, Seq2Seq

The same block composes into three families differing in **masking** and **direction**:

- **Encoder-only (BERT)** — bidirectional self-attention; every token sees every other token. Great for *understanding* (classification, NER, retrieval). Pretrained with **masked language modeling** (predict randomly masked tokens). Not generative.
- **Decoder-only (GPT)** — **causal** self-attention; each token sees only itself and earlier tokens (§7). Trained as a next-token predictor; generates autoregressively. The dominant LLM design (GPT, LLaMA, Mistral).
- **Encoder–Decoder / Seq2Seq (T5, original Transformer, BART)** — a bidirectional encoder reads the input; a causal decoder generates the output while **cross-attending** to the encoder's representations (decoder queries, encoder keys/values). Natural for translation/summarization. T5 frames *every* task as text-to-text.

| Family | Attention | Trained on | Best at | Example |
| --- | --- | --- | --- | --- |
| Encoder-only | bidirectional | masked LM | understanding | BERT |
| Decoder-only | causal | next-token | generation | GPT |
| Encoder–decoder | bi + causal + cross | span/denoise | transduction | T5 |

---

## 6. Tokenization — BPE & WordPiece from scratch

Models operate on a fixed vocabulary of integer IDs, not raw text. Subword tokenizers strike a balance: common words become single tokens, rare words split into pieces — handling any input without a giant vocabulary or out-of-vocabulary failures.

### Byte-Pair Encoding (BPE)

Start with characters (or bytes), then **greedily merge the most frequent adjacent pair** repeatedly until the vocab reaches the target size.

```python
from collections import Counter

def get_pair_counts(corpus):
    """corpus: dict {tuple_of_symbols: frequency}. Count adjacent pairs."""
    pairs = Counter()
    for symbols, freq in corpus.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs

def merge_pair(corpus, pair):
    """Merge every occurrence of `pair` into one symbol."""
    a, b = pair
    out = {}
    for symbols, freq in corpus.items():
        merged, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        out[tuple(merged)] = freq
    return out

def train_bpe(words, num_merges):
    # Each word -> tuple of chars + end-of-word marker.
    corpus = {tuple(w) + ("</w>",): f for w, f in words.items()}
    merges = []
    for _ in range(num_merges):
        pairs = get_pair_counts(corpus)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)     # most frequent adjacent pair
        corpus = merge_pair(corpus, best)
        merges.append(best)
    return merges                            # ordered merge rules = the model
```

Given `{"low": 5, "lower": 2, "newest": 6, "widest": 3}`, BPE quickly merges `e`+`s`→`es`, then `es`+`t`→`est`, etc., discovering the frequent suffix `est` as a reusable token. Encoding a new word replays the learned merges in order. GPT-2/GPT-4 use **byte-level BPE** (merging over raw bytes → never out-of-vocab).

### WordPiece (BERT)

Same merge loop, but instead of picking the most *frequent* pair it picks the pair that most increases corpus likelihood — score $\dfrac{\text{freq}(a,b)}{\text{freq}(a)\,\text{freq}(b)}$ — favoring pairs that co-occur more than chance. Continuation pieces are marked `##` (e.g. `playing → play, ##ing`).

| | Merge criterion | Marker | Used by |
| --- | --- | --- | --- |
| BPE | most frequent pair | `</w>` end-marker | GPT, RoBERTa |
| WordPiece | highest likelihood gain | `##` prefix on continuations | BERT |

---

## 7. Causal Masking — autoregressive generation

A decoder must not "cheat" by looking at future tokens it's supposed to predict. **Causal (look-ahead) masking** enforces this: before softmax, set scores for future positions to $-\infty$ so their attention weight becomes 0.

$$\text{scores}_{ij} \leftarrow \begin{cases} \text{scores}_{ij} & j \le i \quad (\text{past or self}) \\ -\infty & j > i \quad (\text{future}) \end{cases}$$

The mask is a lower-triangular pattern; after softmax each token's weights are nonzero only over positions $\le$ itself.

```python
import numpy as np
n = 4
mask = np.triu(np.ones((n, n)), k=1).astype(bool)   # True above the diagonal = future
scores = np.where(mask, -np.inf, scores)            # block future before softmax
```

For a 4-token sequence the allowed-attention pattern (1 = visible):

```
        k0 k1 k2 k3
q0  [ 1  0  0  0 ]
q1  [ 1  1  0  0 ]
q2  [ 1  1  1  0 ]
q3  [ 1  1  1  1 ]
```

**Why it matters for training & generation:**

- **Training**: one forward pass computes the next-token loss at *every* position in parallel, yet each position only ever saw its past — teacher forcing without leakage.
- **Generation**: tokens are produced one at a time, each conditioned on all previously generated tokens (autoregressive). A **KV-cache** stores past keys/values so each new step is $O(n)$ instead of recomputing the whole sequence.

(Distinguish from the **padding mask**, which hides padding tokens in batched variable-length inputs — often combined with the causal mask.)

---

## Quick Mental-Model Map

| Concept | What it really is |
| ------- | ----------------- |
| Q/K/V attention | Content-based weighted average; softmax of scaled key-query matches |
| Scaling by √dₖ | Keep softmax out of saturation so gradients survive |
| Multi-head | Parallel attention subspaces → diverse relations, same cost |
| Positional encoding | Inject order into a permutation-invariant operation |
| FFN | Per-token processing; holds most of the parameters |
| Residual + LayerNorm | Gradient highway + activation-scale stability |
| Encoder/Decoder/Seq2Seq | Bidirectional vs. causal vs. both + cross-attention |
| BPE / WordPiece | Greedy subword merges → open-vocabulary token IDs |
| Causal mask | −∞ on future scores → autoregressive, no leakage |
