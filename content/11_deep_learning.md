# Deep Learning Foundations

tagline: Neurons, activations, losses, backpropagation derived and verified, initialisation, normalisation, optimisers and schedules, regularisation, plus PyTorch and TensorFlow/Keras side by side.

## The neuron, the layer, the network

$$\mathbf h^{(l)} = \phi\big(W^{(l)}\mathbf h^{(l-1)} + \mathbf b^{(l)}\big),\qquad \hat{\mathbf y} = \text{softmax}\big(W^{(L)}\mathbf h^{(L-1)} + \mathbf b^{(L)}\big)$$

**Intuition:** each layer is a learned change of coordinates followed by a bend ($\phi$). Stacking bends lets the network carve any shape (universal approximation); depth makes it do so with exponentially fewer units than width alone. **Example:** a 20 → 64 → 3 MLP has $20\cdot64 + 64 + 64\cdot3 + 3 = 1{,}539$ weights, plus 128 LayerNorm parameters = 1,667 (the count printed by both frameworks below).

![](figures/nn.svg)

### Activations

![](figures/activations.svg)

| Activation | Formula | Derivative | Use | Problem |
|---|---|---|---|---|
| Sigmoid | $\frac1{1+e^{-x}}$ | $\sigma(1-\sigma) \leq 0.25$ | binary output, gates | vanishing gradients, not zero-centred |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2$ | RNN states | saturates |
| ReLU | $\max(0,x)$ | $\mathbb 1[x>0]$ | CNN/MLP default | dead units for $x<0$ |
| Leaky ReLU | $\max(\alpha x, x)$ | $1$ or $\alpha$ | dead ReLU fix | extra hyperparameter |
| GELU | $x\,\Phi(x)$ | smooth | Transformers (BERT, GPT) | slightly costlier |
| SiLU / Swish | $x\,\sigma(x)$ | smooth | EfficientNet, LLaMA (SwiGLU) | |
| Softmax | $\frac{e^{z_k}}{\sum_j e^{z_j}}$ | $p_k(\delta_{kj} - p_j)$ | multiclass output | overflow → subtract max |

### Losses and the output layers they pair with

| Task | Output | Loss | PyTorch | Keras |
|---|---|---|---|---|
| Regression | linear | MSE / Huber | `MSELoss`, `HuberLoss` | `"mse"`, `Huber()` |
| Binary | 1 logit | BCE with logits | `BCEWithLogitsLoss` | `BinaryCrossentropy(from_logits=True)` |
| Multiclass | $K$ logits | cross-entropy | `CrossEntropyLoss` (int labels) | `SparseCategoricalCrossentropy(from_logits=True)` |
| Multilabel | $K$ logits | BCE per label | `BCEWithLogitsLoss` | `BinaryCrossentropy(from_logits=True)` |
| Imbalanced | logits | focal $-(1-p_t)^\gamma\log p_t$ | torchvision `sigmoid_focal_loss` | `BinaryFocalCrossentropy` |
| Embeddings | vectors | contrastive / triplet / InfoNCE | `TripletMarginLoss` | custom |
| Quantiles | linear | pinball | custom | custom |

!!! pitfall "Pitfall: double softmax"
    `CrossEntropyLoss` applies log-softmax internally. Adding `nn.Softmax()` at the end of the model still trains, but gradients shrink and learning slows. Return **logits**; apply softmax only at inference when you need probabilities. Same in Keras: `from_logits=True`.

## Backpropagation, derived and verified

![](figures/backprop_graph.svg)

!!! derivation "Derivation: 2-layer network backprop"
    Forward: $\mathbf z_1 = X W_1 + \mathbf b_1$, $H = \text{ReLU}(\mathbf z_1)$, $\mathbf z_2 = H W_2 + \mathbf b_2$, $P = \text{softmax}(\mathbf z_2)$, $\mathcal L = -\frac1n\sum\log P_{i,y_i}$. Backward, applying the chain rule node by node: $\frac{\partial\mathcal L}{\partial\mathbf z_2} = \frac1n(P - Y)$ (softmax + CE, from Calculus); $\frac{\partial\mathcal L}{\partial W_2} = H^\top\frac{\partial\mathcal L}{\partial\mathbf z_2}$; $\frac{\partial\mathcal L}{\partial\mathbf b_2} = \sum_i\frac{\partial\mathcal L}{\partial\mathbf z_2}$; $\frac{\partial\mathcal L}{\partial H} = \frac{\partial\mathcal L}{\partial\mathbf z_2}W_2^\top$; $\frac{\partial\mathcal L}{\partial\mathbf z_1} = \frac{\partial\mathcal L}{\partial H}\odot\mathbb 1[\mathbf z_1 > 0]$; $\frac{\partial\mathcal L}{\partial W_1} = X^\top\frac{\partial\mathcal L}{\partial\mathbf z_1}$. Pattern: **gradient of a weight = (input to the layer)ᵀ × (gradient arriving at its output)**. Shapes always match the parameter.

@@code code/s09_backprop.py@@
@@out code/out/s09_backprop.txt@@

!!! intuition "Why gradients vanish or explode"
    Backprop multiplies one Jacobian per layer. If their typical scale is below 1, the product shrinks geometrically with depth (sigmoid: ≤ 0.25 per layer, so $0.25^{10} \approx 10^{-6}$); above 1 it explodes. Remedies: ReLU-family activations, careful initialisation, normalisation layers, residual connections (gradient highway $\partial(\mathbf x + F(\mathbf x))/\partial\mathbf x = I + F'$), gradient clipping, LSTM/GRU gates.

## Initialisation and normalisation

| Technique | Rule | Why |
|---|---|---|
| Xavier/Glorot | $\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$ | keeps variance constant for tanh/sigmoid |
| He/Kaiming | $\text{Var}(W) = \frac{2}{n_{in}}$ | ReLU zeroes half the units, so double the variance |
| Zero init | never for weights | symmetry: all units learn the same thing |
| BatchNorm | normalise over the batch, learn $\gamma,\beta$ | CNNs; behaves differently in train/eval; small batches hurt |
| LayerNorm | normalise over features per sample | Transformers, RNNs; batch-size independent |
| RMSNorm | $\mathbf x/\text{RMS}(\mathbf x)\cdot\gamma$ | LLaMA-style LLMs; cheaper LayerNorm |
| GroupNorm | normalise over channel groups | detection/segmentation with tiny batches |

!!! example "Worked example: He initialisation"
    Layer with 512 inputs, ReLU: $\text{std}(W) = \sqrt{2/512} = 0.0625$. With $\text{std} = 1$ instead, pre-activations would have std $\approx\sqrt{512}\approx 22.6$, and after a few layers the activations overflow or saturate.

## Optimisers and learning rate

![](figures/lr_effects.svg)

| Optimiser | Update | Defaults | Notes |
|---|---|---|---|
| SGD + momentum | $\mathbf v \leftarrow \mu\mathbf v + \mathbf g$; $\theta \leftarrow \theta - \eta\mathbf v$ | $\eta$ 0.1 (CNN), $\mu$ 0.9 | often generalises best on vision with a schedule |
| Adam | $\mathbf m,\mathbf v$ EMAs of $\mathbf g,\mathbf g^2$; $\theta \leftarrow \theta - \eta\,\hat{\mathbf m}/(\sqrt{\hat{\mathbf v}}+\epsilon)$ | $\eta$ 1e-3, $\beta$ (0.9, 0.999) | fast, robust default |
| AdamW | Adam + decoupled weight decay $\theta \leftarrow \theta - \eta\lambda\theta$ | wd 0.01–0.1 | the Transformer default |
| Lion / Adafactor / Sophia | sign-based / factored / 2nd-order | | memory-efficient for huge models |

![](figures/lr_schedules.svg)

!!! rule "Rule of thumb: learning rate first"
    The learning rate matters more than any other hyperparameter. Use an LR range test (increase LR exponentially for one epoch; pick ~1/10 of where loss is lowest). Warmup (1–5% of steps) + cosine decay is the safe default for Transformers; one-cycle for CNNs. If you double the batch size, scale LR by ~2 (linear rule, SGD) or ~√2 (Adam).

![](figures/landscapes.png)

## Regularisation and generalisation

| Technique | Mechanism | Typical value |
|---|---|---|
| Weight decay | penalise $\lVert\theta\rVert^2$ | 1e-4 (CNN), 0.01–0.1 (AdamW) |
| Dropout | randomly zero units (scale by $\frac1{1-p}$ at train) | 0.1–0.5 |
| Early stopping | stop when validation stops improving | patience 5–10 epochs |
| Data augmentation | label-preserving transforms | flips, crops, RandAugment, mixup, CutMix |
| Label smoothing | target $1-\epsilon$ instead of 1 | $\epsilon$ 0.1 |
| Batch/Layer norm | smoother landscape, mild noise | |
| Smaller model / fewer epochs | less capacity | |
| Stochastic depth, EMA of weights | | large vision / diffusion models |

![](figures/learning_curve.svg)

## PyTorch vs TensorFlow/Keras side by side

<div class="sbs" markdown="1">
<div markdown="1">

**PyTorch: model and step**

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64), nn.LayerNorm(64),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 3))
    def forward(self, x):
        return self.net(x)          # logits

model = MLP().to(device)
opt = torch.optim.AdamW(model.parameters(),
                        lr=3e-3, weight_decay=1e-2)
loss_fn = nn.CrossEntropyLoss()
for xb, yb in train_dl:             # you own the loop
    model.train(); opt.zero_grad()
    loss = loss_fn(model(xb), yb)
    loss.backward(); opt.step()
```

</div>
<div markdown="1">

**Keras 3 (TensorFlow backend)**

```python
model = keras.Sequential([
    layers.Input((20,)),
    layers.Dense(64), layers.LayerNormalization(),
    layers.Activation("gelu"), layers.Dropout(0.2),
    layers.Dense(3)])                # logits

model.compile(
    optimizer=keras.optimizers.AdamW(
        3e-3, weight_decay=1e-2),
    loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
    metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds,
          epochs=30, callbacks=[
          keras.callbacks.EarlyStopping(
              patience=5, restore_best_weights=True)])
```

</div>
</div>

| Concern | PyTorch | TensorFlow / Keras |
|---|---|---|
| Style | imperative, explicit loop | declarative `compile`/`fit`; custom `train_step` or `GradientTape` for control |
| Data | `Dataset` + `DataLoader(num_workers, pin_memory)` | `tf.data` (`shuffle`, `batch`, `prefetch(AUTOTUNE)`) |
| Train/eval mode | `model.train()` / `model.eval()` + `torch.inference_mode()` | `training=True/False` handled by `fit`/`predict` |
| Gradients | `loss.backward()`; `opt.zero_grad()` | `with tf.GradientTape() as tape:` → `tape.gradient` |
| Device | `.to("cuda")` explicitly | automatic placement |
| Mixed precision | `torch.autocast("cuda", dtype=torch.bfloat16)` | `keras.mixed_precision.set_global_policy("mixed_bfloat16")` |
| Compile | `torch.compile(model)` | `jit_compile=True` (XLA) |
| Save | `torch.save(model.state_dict())`; export `torch.export` / ONNX | `model.save("m.keras")`; `SavedModel` for TF Serving |
| Distributed | DDP, FSDP, DeepSpeed | `tf.distribute.MirroredStrategy`, Keras 3 JAX backend |
| Serving | TorchServe, Triton, ONNX Runtime, vLLM | TF Serving, TFLite, TF.js |
| Ecosystem | research default, Hugging Face, Lightning | production mobile/edge, TFX |

### Executed: PyTorch training loop with early stopping

@@code code/s09_torch.py@@
@@out code/out/s09_torch.txt@@

### Executed: the same model in Keras

@@code code/s09_keras.py@@
@@out code/out/s09_keras.txt@@

!!! tip "Pro tip: sanity checks before a long run"
    1) Initial loss ≈ $\ln K$ for $K$ balanced classes (1.099 for 3: printed above). 2) Overfit a single batch to ~0 loss; if you cannot, there is a bug. 3) Check that the loss drops in the first 100 steps. 4) Plot a batch of inputs **after** the data pipeline. 5) Log gradient norms per layer.

<div class="card" markdown="1">

## Cheat card: Deep learning foundations

<div class="cols" markdown="1">

#### Architecture defaults
- Hidden activation: ReLU (CNN/MLP), GELU/SwiGLU (Transformers)
- Norm: BatchNorm (CNN), LayerNorm/RMSNorm (Transformers)
- Init: He for ReLU, Xavier for tanh (frameworks default sensibly)
- Residual connections for anything deep

#### Output ↔ loss
- Regression: linear + MSE/Huber
- Binary: 1 logit + BCEWithLogits
- Multiclass: K logits + CrossEntropy
- Multilabel: K logits + BCE per label

#### Optimisation
- AdamW lr 1e-3 to 3e-4, wd 0.01; SGD 0.1 + momentum 0.9 for CNNs
- Warmup + cosine; clip grad norm 1.0
- Batch ×2 → LR ×2 (SGD) / ×√2 (Adam)

#### Regularisation
- Weight decay, dropout 0.1–0.5, augmentation, early stopping, label smoothing 0.1

#### Backprop in one line
- $\partial\mathcal L/\partial W = (\text{layer input})^\top\,\delta_{\text{out}}$
- $\delta$ through ReLU: multiply by $\mathbb 1[z>0]$
- Softmax+CE: $\delta = p - y$

#### Debug order
- Loss ≈ ln K at init? Overfit one batch? LR range test?
- NaN → lower LR, clip, check log(0)/div 0, bf16
- Val ≫ train loss → overfit; both high → underfit

</div>
</div>
