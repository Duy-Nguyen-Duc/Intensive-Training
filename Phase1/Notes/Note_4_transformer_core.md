# Note 4 — Transformer Core

This note focus on the attention mechanism and the Transformer architecture: scaled dot-product attention, multi-head attention, the attention variants (self / cross / causal / bidirectional, plus multi-query and grouped-query), positional encoding, the block structure, the three architecture families, tokenization, and causal masking.


---

## 1. Scaled Dot-Product Attention — Q / K / V from scratch

Attention lets every token **look at** every other token and pull in a weighted mix of their information. Each token produces three vectors via learned projections of its embedding $\mathbf{x}$:

- **Query** $\mathbf{q} = \mathbf{x} * W_Q$, represent "what am I looking for?"
- **Key** $\mathbf{k} =  \mathbf{x} * W_K$, represent "what do I have?"
- **Value** $\mathbf{v} = \mathbf{x} * W_V$, represent "what I actually pass on if attended to."

Stacking the per-token vectors into matrices $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{n \times d_k}$, $V \in \mathbb{R}^{n \times d_v}$ for a sequence of $n$ tokens:

$$\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.$$

Step by step:

1. **Scores** $S = QK^\top$, where $S_{ij}$ = dot product of query $i$ with key $j$ = how relevant token $j$ is to token $i$. Shape $n \times n$.
2. **Scale** by $1/\sqrt{d_k}$. Without it, for large $d_k$ the dot products grow $\sim\sqrt{d_k}$ in magnitude, pushing softmax into saturated regions where gradients vanish.
3. **Softmax** over each row → attention weights that sum to 1 (a probability distribution over the keys).
4. **Weighted sum** of values: output row $i = \sum_j A_{ij}\mathbf{v}_j$.

### Numerical example (1 query, 2 keys, $d_k = 2$)

$\mathbf{q} = (1, 0)$, $K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$, $V = \begin{pmatrix} 10 \\ 20 \end{pmatrix}$.

Scores $\mathbf{q}K^\top = (1, 0)$; scaled by $1/\sqrt{2}$: $(0.707, 0)$. Softmax: $\frac{(e^{0.707}, e^0)}{e^{0.707}+e^0} = (0.670, 0.330)$. Output $= 0.670(10) + 0.330(20) = 13.3$ — the query leans toward the first value because its key aligned with the query.

---

## 2. Multi-Head Attention — parallel heads

A single attention computes one kind of relationship, based on the initialized state. **Multi-head attention** runs $h$ attention operations in parallel, each with its own learned $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}$ projecting into a smaller subspace of size $d_k = d_{\text{model}} / h$. Each head can specialize (one tracks syntax, another coreference, etc.).

$$\text{head}_i = \text{Attention}(X W_Q^{(i)}, X W_K^{(i)}, X W_V^{(i)}),$$
$$\text{MultiHead}(X) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O.$$

- Concatenating $h$ heads of width $d_v = d_{\text{model}}/h$ restores width $d_{\text{model}}$; the output projection $W_O$ mixes information across heads. The split happens in the embedding space, not the token space, so that each head will still see the full sentence, just a different feature space. 
- **Cost is roughly the same** as one full-width head — the per-head dimension shrinks by $h$, so total compute is preserved while gaining representational diversity.

**Example.** $d_{\text{model}} = 512$, $h = 8$ → each head works in $d_k = d_v = 64$. Eight $64$-dim outputs concatenate back to $512$.

---

## 3. Attention Variants — Self, Cross, Causal, Bidirectional, MQA/GQA

Scaled dot-product attention (§1) and multi-head attention (§2) are the *core mechanism*. The variants of attention blocks come from the answer of these choices: 

1. **Where do Q, K, V come from?** which creates self-attention and cross-attention.
2. **Which keys is each query allowed to see?** which creats bidirectional and causal (masked).

Any combination is valid; the named architectures (§6) are simply specific combinations.

### Self-attention — one sequence attends to itself

Q, K, and V are all projections of the **same** input sequence $X$:

$$\text{SelfAttn}(X) = \text{Attention}(X W_Q,\; X W_K,\; X W_V).$$

Every token builds its query, key, and value from itself, so the sequence *mixes information internally* — "the" looks at "cat", "it" looks at its antecedent. This is the workhorse inside both encoder and decoder blocks.

### Cross-attention — one sequence attends to another

Queries come from one sequence, keys/values from **another**:

$$\text{CrossAttn}(Y, X) = \text{Attention}(Y W_Q,\; X W_K,\; X W_V).$$

In a seq2seq decoder (§6), $Y$ = decoder states (what I'm generating) and $X$ = encoder outputs (the source sentence). The decoder *queries* the encoded input — "given what I've written so far, which source tokens matter now?" — which is how translation conditions output on input. The same shape powers retrieval-augmented and multimodal models, where K/V come from documents or image patches.

### Bidirectional vs. causal — the mask sets the direction

Independent of where Q/K/V come from, a mask controls visibility:

- **Bidirectional (unmasked):** every query sees every key. Full context, but the model can "see the future" — good for *understanding*, unusable for generation. Encoders (BERT).
- **Causal (masked):** query $i$ sees only keys $j \le i$; future positions are scored $-\infty$ (mechanism in §8). Required for autoregressive generation, so position $i$ can be trained to predict $i{+}1$ without peeking. Decoders (GPT).

Cross-attention is normally **unmasked** (the decoder may look at the whole source), bounded only by a padding mask.

### Multi-head self-attention (MHSA) — the practical unit

In real models the self-attention above is *always* multi-head (§2): $h$ heads in parallel subspaces, concatenated. So the three sublayers you actually stack are:

| Sublayer | Q from | K, V from | Mask | Appears in |
| --- | --- | --- | --- | --- |
| Bidirectional MHSA | $X$ | $X$ | none (+pad) | encoder (BERT) |
| Causal / masked MHSA | $X$ | $X$ | causal | decoder (GPT) |
| Multi-head cross-attention | decoder $Y$ | encoder $X$ | none (+pad) | seq2seq decoder (T5) |

Concretely, with the `MultiHeadAttention(query, key, value, mask)` from `Ass_4.ipynb` — note that self-attention just passes the **same tensor three times**:

```python
mha(x, x, x)                       # bidirectional self-attention (encoder)
mha(x, x, x, mask=causal_mask(n))  # causal self-attention       (decoder)
mha(dec, enc, enc)                 # cross-attention (decoder queries encoder)
```

A full decoder block therefore has **two** attention sublayers — causal self-attention, then cross-attention — plus the FFN. A decoder-only LM (GPT, `Ass_5.ipynb`) drops the cross-attention and keeps only causal self-attention.

---

## 4. Positional Encoding — sinusoidal vs. learned

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

## 5. Feed-Forward + Residual + LayerNorm — the block

A Transformer block wraps attention and a position-wise feed-forward network, each inside a **residual connection** followed by **LayerNorm**.

**Position-wise FFN** — applied independently to each token, expanding then contracting:

$$\text{FFN}(\mathbf{x}) = \sigma(\mathbf{x} W_1 + \mathbf{b}_1)\, W_2 + \mathbf{b}_2,$$

with hidden width typically $4\times d_{\text{model}}$ and $\sigma$ = GELU. Attention mixes information *across* tokens; the FFN processes each token's mixed representation *individually* (most parameters live here).

**Residual + Norm.** Each sublayer (attention, FFN) is wrapped:

- **Post-LN** (original): $\;\text{out} = \text{LayerNorm}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))$.
- **Pre-LN** (modern, more stable for deep stacks): $\;\text{out} = \mathbf{x} + \text{Sublayer}(\text{LayerNorm}(\mathbf{x}))$.

Residuals give the gradient highway, while LayerNorm stabilizes the scale of activations. A full block:

```
x = x + MultiHeadAttention(LayerNorm(x))   # mix across tokens
x = x + FFN(LayerNorm(x))                   # process each token
```

---

## 6. Architecture Families — Encoder-only, Decoder-only, Seq2Seq

The same block composes into three families differing in **masking** and **direction**:

- **Encoder-only (BERT)** — bidirectional self-attention; every token sees every other token. Great for *understanding* (classification, NER, retrieval). Pretrained with **masked language modeling** (predict randomly masked tokens). Not generative.
- **Decoder-only (GPT)** — **causal** self-attention; each token sees only itself and earlier tokens (§8). Trained as a next-token predictor; generates autoregressively. The dominant LLM design (GPT, LLaMA, Mistral).
- **Encoder–Decoder / Seq2Seq (T5, original Transformer, BART)** — a bidirectional encoder reads the input; a causal decoder generates the output while **cross-attending** to the encoder's representations (decoder queries, encoder keys/values). Natural for translation/summarization. T5 frames *every* task as text-to-text.

| Family | Attention | Trained on | Best at | Example |
| --- | --- | --- | --- | --- |
| Encoder-only | bidirectional | masked LM | understanding | BERT |
| Decoder-only | causal | next-token | generation | GPT |
| Encoder–decoder | bi + causal + cross | span/denoise | transduction | T5 |

---

## 7. Tokenization — BPE & WordPiece from scratch

Firstly, *tokenizer* is the function that splits raw text into a sequence of tokens (the atomic units) and maps each token to an integer ID from a fixed vocabulary. It's the boundary between messy human text and the model's input:
```
  "the cat sat"  --tokenize-->  ["the", "cat", "sat"]  --lookup-->  [791, 9059, 7731]
```
Those IDs then index an embedding table to become vectors, such as transform the interger ID 791 into a 64-dimension vector. The tokenizer decides what the model's atoms are. 

The history of tokenizer come all the way to the most traditional **Bag-of-Word method**, where each word is tokenized into the number of its appearance in the document. While the method is simple and fast, losing its order and its similarity between words are the limitations of using this method in modern approaches. **Word-level tokenization** is the higher stages, where each word is a token, preserves order and pairs nicely with embeddings. However, the method stucks in its training vocabulary that it cannot generalize or find similar words when deploy in real world, or it cannot understand "run", "runs" and "running" refer to the same action. 
**Character-level** approachs the tokenizing problem at the opposite extreme. Instead of tokennize each word, it focuses on a sublevel of each character, where each word is a sequence of characters, then it can generalize "running" is the same meaning as "run", only with something as suffix. Despite the effort, the sequences get very long, where it becomes hundreds of tokens, and a tokens carry little meaning. This inspire the introduce of subword tokenization. With tokenizer, we would want something that is OOV-immunity of characters with the meaning-density and short sequences of words. The insight: let the data decide the units. Frequent strings should become single tokens; rare strings should break into smaller, reusable pieces. That's exactly what **BPE** does, start from the smallest units (bytes/characters), then greedily merge the most frequent adjacent pair, repeatedly, until it hits a target vocab size. The learned merge rules are the tokenizer. 


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

---

## 8. Causal Masking — autoregressive generation

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

