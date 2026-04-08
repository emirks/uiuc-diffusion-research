# Transformer Study Notes

## From Token Embeddings to Next-Token Prediction

---

## 1. Purpose of These Notes

These notes summarize our current understanding of the internal process of a transformer-based language model, especially the parts needed to build a precise conceptual foundation before moving on to models such as LTX and other multimodal or latent-token transformers.

The goal is not to cover every transformer detail, but to build a technically sound mental model of:

- tokenization and one-hot token space
- embedding and unembedding
- how training updates embeddings
- variable sequence length
- self-attention mechanics and shapes
- output projection in multi-head attention
- the difference between attention and MLP blocks
- training versus inference
- the overall transformer pipeline

---

## 2. Notation Table

| Symbol | Meaning |
|---|---|
| \(V\) | Vocabulary size |
| \(d\) | Model / embedding dimension |
| \(T\) | Sequence length |
| \(h\) | Number of attention heads |
| \(d_k\) | Per-head feature dimension, usually \(d/h\) |
| \(E \in \mathbb{R}^{V \times d}\) | Token embedding matrix |
| \(x \in \mathbb{R}^{1 \times V}\) | One-hot token vector |
| \(H \in \mathbb{R}^{T \times d}\) | Hidden-state matrix across the sequence |
| \(W_Q, W_K, W_V \in \mathbb{R}^{d \times d}\) | Query, key, value projection matrices |
| \(W_O \in \mathbb{R}^{d \times d}\) | Output projection matrix inside attention |
| \(Q, K, V\) | Query, key, value tensors |
| \(A\) | Raw attention score matrix |
| \(P\) | Attention weight matrix after softmax |
| \(Z\) | Vocabulary logits |
| \(L\) | Training loss |

---

## 3. Overview of the Full Pipeline

A transformer language model processes text as follows:

\[
\text{text}
\rightarrow
\text{token IDs}
\rightarrow
\text{token embeddings}
\rightarrow
\text{positional information}
\rightarrow
\text{transformer stack}
\rightarrow
\text{vocabulary logits}
\rightarrow
\text{softmax}
\rightarrow
\text{next-token probabilities}
\]

During training, all trainable parts are optimized end-to-end using next-token prediction loss.

---

## 4. Tokenization and the Canonical Token Space

A tokenizer converts text into token IDs.

Example:

```text
"The cat sat" -> [17, 842, 1932]
```

If the vocabulary size is \(V\), then each token can be represented as a basis vector in:

\[
\mathbb{R}^V
\]

For token \(i\), the one-hot vector is:

\[
x = e_i
\]

where \(e_i\) has a 1 at index \(i\) and 0 elsewhere.

### Interpretation

This one-hot space is the canonical symbolic token space.
Each axis corresponds to one vocabulary item.
A one-hot vector is an exact symbolic identity representation of a token.

### Important clarification

The values in a one-hot vector are not probabilities in the usual modeling sense.
They are indicator coordinates:

- 1 means: this token index is active
- 0 means: this token index is not active

### Compact notes from our discussion

- Correct insight: one-hot space can be viewed as the model's formal discrete token basis space.
- Important correction: one-hot entries are indicators, not a probability distribution over tokens.

---

## 5. Embedding Matrix: Mapping Token Space to Dense Feature Space

The model stores a learned embedding matrix:

\[
E \in \mathbb{R}^{V \times d}
\]

where:

- \(V\) is vocabulary size
- \(d\) is embedding dimension

Each row of \(E\) is the embedding of one token.

If a token is represented by one-hot vector \(x \in \mathbb{R}^{1 \times V}\), then the embedding is:

\[
h = xE
\]

with shape:

\[
[1 \times V] \cdot [V \times d] = [1 \times d]
\]

So the one-hot input multiplies the embedding matrix from the left.
This is equivalent to selecting the row corresponding to the token ID.

### Toy example

Let:

\[
E =
\begin{bmatrix}
a & b \\
c & d \\
e & f
\end{bmatrix}
\in \mathbb{R}^{3 \times 2}
\]

and let token index 1 be:

\[
x = [0\;1\;0]
\]

Then:

\[
xE = [c\;d]
\]

### Compact notes from our discussion

- Correct insight: the embedding matrix is a learned projection from discrete token space into dense feature space.
- Correct insight: the matrix orientation matters; the standard view is \(E \in \mathbb{R}^{V \times d}\), and we select a row.

---

## 6. Positional Information

Token identity alone does not encode order.
A sequence of embeddings therefore needs positional information.

After embedding lookup, the model forms:

\[
H_0 = \text{token embeddings} + \text{positional encoding}
\]

with:

\[
H_0 \in \mathbb{R}^{T \times d}
\]

where each row corresponds to one token position in the sequence.

### Compact notes from our discussion

- Correct insight: the one-hot token representation itself does not capture order.
- Position must be injected separately.

---

## 7. How the Embedding Matrix Is Trained

The embedding matrix is not trained in a separate stage.
It is learned jointly with the transformer through the next-token prediction objective.

If:

\[
h = xE
\]

then the gradient from the language-model loss flows backward through \(h\) into \(E\).

Training objective:

\[
L = -\log P(\text{correct next token})
\]

Gradient flow:

```text
loss
↓
softmax + logits
↓
output projection / unembedding
↓
transformer layers
↓
input embeddings
```

The gradient with respect to the embedding matrix is:

\[
\frac{\partial L}{\partial E} = x^\top \frac{\partial L}{\partial h}
\]

Since \(x\) is one-hot, only the active row of \(E\) is directly updated for that token occurrence.

### Compact notes from our discussion

- Correct insight: embeddings are not trained externally or separately.
- Correct insight: they are optimized end-to-end via the language-model loss.

---

## 8. Unembedding and Weight Tying

After the final transformer layer, the model must score all vocabulary tokens.
If the final hidden state at a position is:

\[
h \in \mathbb{R}^{1 \times d}
\]

then logits are computed as:

\[
z = hE^\top
\]

with shape:

\[
[1 \times d] \cdot [d \times V] = [1 \times V]
\]

Each logit is:

\[
z_i = \langle h, e_i \rangle
\]

where \(e_i\) is the embedding vector for token \(i\).

Then softmax converts logits into probabilities over vocabulary items.

### Weight tying

Many models use the same matrix for:

- input token embeddings
- output token scoring

This is called weight tying.

---

## 9. Why the Transpose Is Not an Inverse

A major conceptual correction we made is this:

\[
E^\top \neq E^{-1}
\]

in general.

The output step is not trying to invert the embedding map.
It is not reconstructing the original one-hot token.
Instead, it is scoring which token embedding best matches the final contextual hidden state.

So:

\[
\text{embedding} = xE
\]

is a projection from token basis space into feature space, while:

\[
\text{logits} = hE^\top
\]

means: compare \(h\) against all token embeddings by dot product.

This is a classifier/scoring interpretation, not an autoencoder reconstruction interpretation.

### Why inverse cannot even be expected

Usually:

- \(E\) is rectangular
- \(V \gg d\)

Example:

\[
E \in \mathbb{R}^{50000 \times 4096}
\]

So a true two-sided inverse is impossible.

### Compact notes from our discussion

- Correct insight: the output side maps back toward token coordinates.
- Crucial correction: transpose is used for scoring, not for exact inversion.
- Crucial correction: weight tying does not enforce orthogonality or identity reconstruction.

---

## 10. Context Length and Padding

If a model has context length 1024, that means this is the maximum allowed number of tokens, not the required number.

The model can process:

- 1 token
- 200 tokens
- 700 tokens
- 1024 tokens

but not more than 1024.

### Pretraining case

In standard GPT-style pretraining, text is often chunked into exact-length blocks, so padding may not be needed.

### Fine-tuning / batching case

When variable-length sequences are batched together, padding may be used.
If so, the PAD token typically has a learned embedding, but attention masking prevents the model from treating padding as actual content.

### Compact notes from our discussion

- Correct insight: asking whether PAD has an embedding is exactly the right question.
- Correct answer: yes, if PAD exists it is usually learned, but masked.

---

## 11. Variable Sequence Length: Why the Model Can Handle It

A transformer does not require a fixed sequence length because its weights depend on the feature dimension \(d\), not on the number of tokens \(T\).

The hidden state is always of the form:

\[
H \in \mathbb{R}^{T \times d}
\]

Possible examples:

\[
[1 \times d],\quad [200 \times d],\quad [1024 \times d]
\]

If a linear map uses:

\[
W \in \mathbb{R}^{d \times d}
\]

then:

\[
[T \times d] \cdot [d \times d] = [T \times d]
\]

for any valid \(T\).

So the same parameters can process different sequence lengths.

### Practical note

In implementation, batches are often padded to a common length for efficient GPU execution.
That is an engineering convenience, not a mathematical requirement of the transformer itself.

### Compact notes from our discussion

- Correct insight: the fixed-input assumption from classical neural networks does not directly apply here.
- Key clarification: transformers process a sequence of vectors, not one giant fixed-size vector.

---

## 12. The Transformer State Space

At every layer, the model state has shape:

\[
H \in \mathbb{R}^{T \times d}
\]

This should be interpreted as:

- \(T\) token positions
- each token carrying a \(d\)-dimensional feature vector

Although one could flatten this into a length-\(Td\) vector, that is not the most useful conceptual view.
The state is structured along two axes:

1. token axis
2. feature axis

A helpful interpretation is that there are \(T\) token-level subspaces of dimension \(d\), with attention allowing information flow across token positions and MLPs processing each token locally.

### Compact notes from our discussion

- Correct insight: this is not best thought of as an arbitrary flat \(Td\)-dimensional space.
- Correct insight: your “\(T\) subspaces of dimension \(d\)” picture is a useful mental model.

---

## 13. Transformer Block Structure

A standard transformer block has the following high-level structure:

```text
Input H
↓
LayerNorm
↓
Multi-Head Attention
↓
Residual Add
↓
LayerNorm
↓
MLP / Feed-Forward
↓
Residual Add
```

So each block maps:

\[
[T \times d] \rightarrow [T \times d]
\]

Stacking many such blocks creates the transformer stack.

### Compact notes from our discussion

- Correct insight: each transformer block preserves the overall \(T \times d\) shape.
- Correct insight: the transformer is built by cascading such blocks.

---

## 14. Self-Attention: Formula and Shapes

Given hidden states:

\[
H \in \mathbb{R}^{T \times d}
\]

we compute:

\[
Q = HW_Q, \qquad K = HW_K, \qquad V = HW_V
\]

with:

\[
W_Q, W_K, W_V \in \mathbb{R}^{d \times d}
\]

Therefore:

\[
Q, K, V \in \mathbb{R}^{T \times d}
\]

For multi-head attention, the feature dimension is split into heads, so per head we use:

\[
Q, K, V \in \mathbb{R}^{T \times d_k}
\]

where typically \(d_k = d/h\).

### Attention scores

For one head:

\[
A = \frac{QK^\top}{\sqrt{d_k}}
\]

Shape:

\[
[T \times d_k] \cdot [d_k \times T] = [T \times T]
\]

Each entry \(A_{ij}\) measures how strongly token \(i\) attends to token \(j\).

### Attention weights

\[
P = \text{softmax}(A)
\]

This is still \([T \times T]\), with each row summing to 1.

### Weighted value mixing

\[
O = PV
\]

Shape:

\[
[T \times T] \cdot [T \times d_k] = [T \times d_k]
\]

This means each token becomes a weighted mixture of value vectors from other tokens.

### Causal masking in language models

Future positions are masked, so token \(i\) can only attend to tokens \(1,\dots,i\).

### Compact notes from our discussion

- Correct insight: self-attention introduces inter-token interaction.
- Important refinement: the actual inter-token information flow happens at the \(PV\) stage.
- Important refinement: the earlier \(Q/K/V\) projections are still token-local linear projections.

---

## 15. Multi-Head Attention and the Output Projection

In multi-head attention, several heads compute attention in parallel.
Their outputs are concatenated:

\[
O = \text{concat}(head_1, \dots, head_h)
\]

After concatenation, we are back in a \([T \times d]\) representation.
Then the model applies an output projection:

\[
H_{\text{attn}} = OW_O
\]

with:

\[
W_O \in \mathbb{R}^{d \times d}
\]

### Why \(W_O\) is needed

Even though \(PV\) already gives meaningful weighted value combinations, the output projection is still useful because it:

1. mixes information across heads
2. recombines head-specific subspaces into a unified representation
3. maps the result back into the shared model feature space
4. supports compatibility with the residual path

Without \(W_O\), the head outputs would remain more isolated from one another.

### Compact notes from our discussion

- Correct insight: \(PV\) already contains semantically meaningful mixed values.
- Crucial addition: \(W_O\) is needed mainly to reintegrate and mix multi-head outputs.

---

## 16. MLP / Feed-Forward Layer

The MLP block acts independently on each token position.
A typical form is:

\[
\text{MLP}(x) = W_2\,\sigma(W_1x + b_1) + b_2
\]

Applied row-wise to \(H\), this means:

- each token is processed independently
- no token-to-token communication occurs inside the MLP
- feature mixing happens within the \(d\)-dimensional token representation

### Compact notes from our discussion

- Correct insight: the MLP captures within-token feature transformations.
- Correct insight: it does not capture inter-token relations.

---

## 17. Communication vs Computation

A very useful summary of the transformer block is:

- **Attention = communication**
- **MLP = computation**

More precisely:

### In the attention part

- \(Q/K/V\) projections perform token-local feature transformations
- \(QK^\top\) measures inter-token relations
- \(PV\) performs actual cross-token information exchange

### In the MLP part

- the model processes each token representation independently
- information is refined within each token's feature vector

So the transformer alternates between:

1. communication across tokens
2. local computation within tokens

### Compact notes from our discussion

- Correct insight: your intra-relation / inter-relation framing is a strong mental model.
- Clean summary:

| Component | Intra-token | Inter-token |
|---|---:|---:|
| Q/K/V projections | yes | no |
| Attention mixing \(PV\) | yes indirectly | yes |
| MLP | yes | no |

---

## 18. Training Versus Inference

### During training

Every position predicts the next token.

Example:

Input:

\[
[t_1, t_2, t_3, t_4]
\]

Targets:

\[
[t_2, t_3, t_4, t_5]
\]

So the model computes logits for every position and applies loss at every position.

### During inference

Only the hidden state at the last actual token position is used to predict the next token.

If the input length is 200, we use the hidden state of token 200, not token 1024.

If the input contains one token, that one token's hidden state predicts the next token; the new token is appended and the process repeats autoregressively.

### Compact notes from our discussion

- Correct insight: the phrase “use the last token” needed clarification.
- Correct distinction:
  - training: all positions predict
  - inference: the last actual position predicts the next token

---

## 19. Final Language-Model Head

After the last transformer block, we have:

\[
H_L \in \mathbb{R}^{T \times d}
\]

Then unembedding gives logits:

\[
Z = H_LE^\top
\]

with:

\[
Z \in \mathbb{R}^{T \times V}
\]

Each row contains scores for all tokens in the vocabulary.
Softmax converts these into probabilities.
The training loss encourages the true next token to receive high probability.

At inference time:

1. take the last row of \(Z\)
2. apply softmax
3. sample or select the next token
4. append it to the sequence
5. repeat

---

## 20. Final Conceptual Summary

A transformer can be understood through the following layered abstraction:

```text
symbolic token space
↓
one-hot token identity
↓
embedding projection
↓
positionalized sequence state (T × d)
↓
repeated transformer blocks
   - attention: communication across tokens
   - MLP: computation within tokens
↓
contextual token representations
↓
unembedding / token scoring
↓
probability distribution over the next token
```

This is a strong and usable foundation for studying transformer-based systems more deeply.

---

## 21. Why This Foundation Is Enough to Move Toward LTX

This understanding is sufficient to begin studying LTX and other transformer-based generative models.

Why? Because the core transformer logic stays the same when we replace text tokens with other token types.

For example, in latent video or image models:

- text tokens may be replaced by image patches or latent video patches
- the sequence becomes a sequence of visual or latent tokens
- the transformer still processes a structured \((T \times d)\) state
- attention still performs cross-token communication
- MLP still performs per-token computation

What changes are things like:

- tokenization scheme
- positional encoding scheme
- conditioning setup
- training objective (e.g. diffusion / denoising instead of next-token prediction)

But the transformer core remains recognizably the same.

---

## 22. One-Page Cheat Sheet

### Core objects

- Vocabulary size: \(V\)
- Sequence length: \(T\)
- Model dimension: \(d\)
- Embedding matrix: \(E \in \mathbb{R}^{V \times d}\)
- Hidden state: \(H \in \mathbb{R}^{T \times d}\)

### Token embedding

\[
x \in \mathbb{R}^{1 \times V}, \qquad h = xE \in \mathbb{R}^{1 \times d}
\]

Interpretation: one-hot token basis vector mapped into dense feature space.

### Transformer input

\[
H_0 = \text{token embeddings} + \text{positional information}
\]

### Self-attention

\[
Q = HW_Q, \quad K = HW_K, \quad V = HW_V
\]

\[
A = \frac{QK^\top}{\sqrt{d_k}}, \qquad P = \text{softmax}(A), \qquad O = PV
\]

### Multi-head output

\[
H_{\text{attn}} = \text{concat}(head_1,\dots,head_h)W_O
\]

### MLP

Applied independently to each token row:

\[
\text{MLP}(x) = W_2\sigma(W_1x+b_1)+b_2
\]

### Block summary

```text
LayerNorm
→ Attention
→ Residual
→ LayerNorm
→ MLP
→ Residual
```

### Roles

- Attention = communication across tokens
- MLP = computation within each token

### Unembedding

\[
Z = H_LE^\top
\]

Each row of \(Z\) contains vocabulary logits.

### Training

- every position predicts the next token
- embeddings are learned end-to-end
- transpose in weight tying is not inverse

### Inference

- use the last actual token position
- softmax over vocabulary
- sample / choose next token
- append and repeat

### Key conceptual corrections

- one-hot entries are indicators, not probabilities
- \(E^\top\) is not an inverse decoder
- output scoring is classification, not reconstruction
- variable sequence length is allowed because weights depend on \(d\), not \(T\)

### Best compact mental model

```text
attention = communication
MLP = computation
```

---

## 23. Final Closing Note

This set of notes captures a coherent understanding of the transformer pipeline at a level strong enough to support deeper study.
In particular, it is a good foundation for moving into:

- RoPE and positional encodings
- patch tokenization
- latent token spaces
- multimodal transformers
- LTX and related video generation architectures

