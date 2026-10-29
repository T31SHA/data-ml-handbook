# Architectures: CNNs, RNNs, Transformers

tagline: The inductive biases that made deep learning work on images, sequences and language, with attention derived from first principles and the full Transformer block dissected.

## Inductive bias: why architecture matters

| Architecture | Assumption baked in | Data | Strength | Weakness |
|---|---|---|---|---|
| MLP | none (fully connected) | tabular, embeddings | universal | no structure sharing, parameter hungry |
| CNN | locality + translation equivariance | images, audio, 1-D signals | sample efficient, fast | limited global context |
| RNN/LSTM/GRU | sequential order, Markov state | time series, streaming | constant memory per step | slow (no parallelism), forgets long range |
| Transformer | all-to-all interactions, learned | text, code, images (ViT), audio, tabular | scales with data and compute | $O(n^2)$ attention, data hungry |
| GNN | relational structure | graphs, molecules, social | uses edges | over-smoothing in deep stacks |
| State space (Mamba) | linear recurrence, selective | very long sequences | $O(n)$ inference | younger ecosystem |

## Convolutional networks

![](figures/cnn.svg)

$$(\mathbf X * \mathbf K)_{i,j} = \sum_{u}\sum_{v} X_{i+u,\,j+v}\,K_{u,v},\qquad H_{out} = \left\lfloor\frac{H_{in} + 2p - k}{s}\right\rfloor + 1$$

**Intuition:** slide a small learned pattern detector over the image; the same weights everywhere (weight sharing) means a cat ear is detected wherever it appears. **Example:** 224×224 input, 7×7 kernel, stride 2, padding 3: $\lfloor(224 + 6 - 7)/2\rfloor + 1 = 112$. A 3×3 conv from 64 to 128 channels has $3\cdot3\cdot64\cdot128 + 128 = 73{,}856$ parameters, independent of image size.

| Concept | Meaning | Rule |
|---|---|---|
| Receptive field | input region influencing one output | grows by $(k-1)\times$ product of earlier strides per layer |
| Pooling | downsample (max/avg) | global average pooling before the classifier |
| 1×1 conv | mix channels cheaply | bottlenecks, channel projection |
| Depthwise-separable | per-channel spatial conv + 1×1 | ~8–9× fewer FLOPs (MobileNet) |
| Residual block | $\mathbf y = \mathbf x + F(\mathbf x)$ | enabled 100+ layers (ResNet) |
| Dilated conv | gaps in the kernel | larger receptive field, same params |

| Family | Year | Key idea |
|---|---|---|
| LeNet / AlexNet | 1998 / 2012 | conv + pool; ReLU, dropout, GPUs |
| VGG | 2014 | stacks of 3×3 |
| ResNet | 2015 | skip connections |
| EfficientNet | 2019 | compound scaling of depth/width/resolution |
| ViT | 2020 | image patches as tokens |
| ConvNeXt | 2022 | modernised CNN matching ViTs |

### Transfer learning (executed)

@@code code/s09_transfer.py@@
@@out code/out/s09_transfer.txt@@

!!! rule "Rule of thumb: how much to fine-tune"
    Small data, similar domain → freeze the backbone, train the head (0.02% of parameters above). Medium data → unfreeze the last block with a 10× smaller LR than the head (stage 2 above). Large or different domain (X-rays, satellite) → fine-tune everything with a low LR and warmup. Always reuse the pretrained model's input normalisation.

## Recurrent networks

![](figures/rnn.svg)

| Cell | Update | Note |
|---|---|---|
| Vanilla RNN | $\mathbf h_t = \tanh(W_h\mathbf h_{t-1} + W_x\mathbf x_t + \mathbf b)$ | gradient through $W_h^T$ vanishes/explodes |
| LSTM | forget/input/output gates; cell $\mathbf c_t = \mathbf f_t\odot\mathbf c_{t-1} + \mathbf i_t\odot\tilde{\mathbf c}_t$ | additive cell path keeps gradients alive |
| GRU | update + reset gates, no separate cell | fewer params, similar quality |
| Bidirectional | forward + backward passes | needs the full sequence (not streaming) |
| Seq2seq + attention | encoder states attended by decoder | the bridge to Transformers |

!!! intuition "Why LSTMs remember"
    The cell state is updated additively: $\partial\mathbf c_t/\partial\mathbf c_{t-1} = \mathbf f_t$. When the forget gate is near 1, the gradient passes through many steps almost unchanged, the same trick as a residual connection, but learned per step.

## Attention from first principles

![](figures/attention_mech.svg)

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

**Intuition:** each token asks a question (query), every token advertises what it contains (key) and what it will pass on (value). Similarity of query and key decides how much of each value flows in. $M$ is the mask ($-\infty$ on future positions for decoders).

!!! derivation "Derivation: why divide by √d_k"
    Let $q_i, k_i$ be independent with mean 0 and variance 1. Then $\text{Var}(\mathbf q\cdot\mathbf k) = \sum_{i=1}^{d_k}\text{Var}(q_ik_i) = \sum_i \mathbb E[q_i^2]\mathbb E[k_i^2] = d_k$. So dot products have standard deviation $\sqrt{d_k}$ (64 for $d_k = 4096$). Large logits push softmax into a near one-hot regime where gradients $p_k(\delta_{kj} - p_j)$ are ~0. Dividing by $\sqrt{d_k}$ restores unit variance. The snippet below measures exactly this: std 4.04, 15.96, 64.24 before scaling and 1.00 after.

@@code code/s10_attention.py@@
@@out code/out/s10_attention.txt@@

![](figures/attention.svg)

### Multi-head attention and positions

$$\text{MHA}(X) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W^O,\quad \text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$

Heads attend to different relations (syntax, coreference, position) in parallel at the same total cost as one full-width head ($d_k = d_{model}/h$).

| Positional scheme | How | Used by |
|---|---|---|
| Sinusoidal | fixed $\sin/\cos$ of position at many frequencies | original Transformer |
| Learned absolute | embedding per position | BERT, GPT-2 |
| RoPE | rotate $\mathbf q,\mathbf k$ by position-dependent angles: dot product depends on relative offset | LLaMA, Mistral, Qwen |
| ALiBi | linear distance penalty on logits | BLOOM, MPT; extrapolates |

## The Transformer block

![](figures/transformer_full.svg)

$$\mathbf x \leftarrow \mathbf x + \text{MHA}(\text{Norm}(\mathbf x)),\qquad \mathbf x \leftarrow \mathbf x + \text{FFN}(\text{Norm}(\mathbf x)),\qquad \text{FFN}(\mathbf x) = W_2\,\phi(W_1\mathbf x)$$

| Variant | Attention | Pretraining objective | Examples | Best for |
|---|---|---|---|---|
| Encoder-only | bidirectional | masked LM | BERT, RoBERTa, DeBERTa | classification, NER, embeddings |
| Decoder-only | causal | next-token prediction | GPT, LLaMA, Mistral, Qwen | generation, chat, reasoning |
| Encoder–decoder | bi + cross-attention | span corruption / denoising | T5, BART, Whisper | translation, summarisation, speech |

!!! example "Worked example: parameter count of a 7B-class model"
    Per layer: attention $4d^2$ ($Q,K,V,O$) + SwiGLU FFN $3\cdot d\cdot d_{ff}$. With $d = 4096$, $d_{ff} = 11008$, 32 layers: $32\,(4\cdot4096^2 + 3\cdot4096\cdot11008) = 32\,(67.1\text{M} + 135.3\text{M}) = 6.48$B, plus embeddings $32000\cdot4096\cdot2 = 0.26$B → about 6.7B (LLaMA-2-7B).

| Cost | Training | Inference |
|---|---|---|
| FLOPs | $\approx 6ND$ ($N$ params, $D$ tokens) | $\approx 2N$ per generated token |
| Attention | $O(n^2d)$ time, $O(n^2)$ memory (FlashAttention makes it IO-efficient) | KV cache: $O(n)$ per step |
| Memory | weights + grads + Adam states ≈ 16 bytes/param in mixed precision | weights + KV cache |

!!! interview "Interview: why did Transformers replace RNNs?"
    Parallelism: all positions are processed at once during training, so they scale with GPUs; RNNs are sequential. Path length: any two tokens are one attention hop apart instead of $O(n)$ steps, so long-range dependencies are learned easily. Cost: quadratic attention, mitigated by FlashAttention, sliding windows, and linear/state-space alternatives.

<div class="card" markdown="1">

## Cheat card: Architectures

<div class="cols" markdown="1">

#### Pick by data
- Images: pretrained ConvNeXt / EfficientNet / ViT; fine-tune
- Text: pretrained Transformer (encoder for classification, decoder for generation)
- Time series: GBM with lags first; then TFT, PatchTST, N-BEATS
- Audio: spectrogram + CNN, or wav2vec2 / Whisper
- Graphs: GNN (GCN, GraphSAGE, GAT)

#### Shapes and sizes
- Conv out: $\lfloor(H + 2p - k)/s\rfloor + 1$
- Conv params: $k^2 C_{in} C_{out} + C_{out}$
- Attention params: $4d^2$ per layer; FFN $8d^2$ (or $3 d\,d_{ff}$ SwiGLU)
- Train FLOPs $\approx 6ND$; inference $\approx 2N$/token

#### Attention
- $\text{softmax}(QK^\top/\sqrt{d_k})V$
- Scale keeps logits at unit variance
- Causal mask for decoders
- Heads $h$, $d_k = d/h$
- RoPE for relative positions

#### Training tips
- Pre-norm (norm before sublayer) is more stable
- Warmup + cosine, AdamW, grad clip 1.0
- Mixed precision bf16; gradient checkpointing for memory
- Transfer: freeze → partial → full fine-tune

</div>
</div>
