# Applied AI: Vision, Generative, RL, RecSys, Time Series, Anomalies, Fairness

tagline: The specialised problem families you meet in production, each with the baseline to beat, the metric that matters, and the trap that catches teams.

## Computer vision tasks

| Task | Output | Models | Metric |
|---|---|---|---|
| Classification | label per image | ResNet, EfficientNet, ConvNeXt, ViT | accuracy, top-5, F1 |
| Detection | boxes + labels | YOLO (v8+), Faster R-CNN, DETR | mAP@[.5:.95] |
| Semantic segmentation | label per pixel | U-Net, DeepLab, SegFormer | mIoU, Dice |
| Instance segmentation | mask per object | Mask R-CNN, SAM (promptable) | mask mAP |
| OCR / documents | text + layout | Tesseract, PaddleOCR, Donut, LayoutLM, VLMs | CER, WER, field F1 |
| Embeddings / retrieval | vector | CLIP, DINOv2 | recall@k |
| Vision-language | text about images | GPT-4-class VLMs, LLaVA, Qwen-VL | task accuracy |

$$\text{IoU} = \frac{\lvert A\cap B\rvert}{\lvert A\cup B\rvert}\qquad \text{Dice} = \frac{2\lvert A\cap B\rvert}{\lvert A\rvert + \lvert B\rvert}$$

**Example:** predicted box 100 px², true box 100 px², overlap 50 px²: IoU $= 50/150 = 0.33$ (a miss at the usual 0.5 threshold), Dice $= 100/200 = 0.5$.

!!! tip "Pro tip: data beats architecture in vision"
    Label quality and coverage dominate. Audit the confusion matrix and the worst predictions, add augmentations that reflect deployment conditions (blur, lighting, compression), and use a pretrained backbone. Zero-shot CLIP or SAM often gives a strong baseline before any training.

## Generative models

![](figures/generative.svg)

| Family | Idea | Objective | Strength | Weakness |
|---|---|---|---|---|
| Autoregressive | $p(\mathbf x) = \prod_t p(x_t\mid x_{<t})$ | next-token log-likelihood | text, code, exact likelihood | sequential sampling |
| VAE | encoder → latent $\mathbf z$ → decoder | ELBO $= \mathbb E[\log p(\mathbf x\mid\mathbf z)] - \text{KL}(q(\mathbf z\mid\mathbf x)\Vert p(\mathbf z))$ | smooth latent space | blurry samples |
| GAN | generator vs discriminator game | $\min_G\max_D\ \mathbb E[\log D(\mathbf x)] + \mathbb E[\log(1 - D(G(\mathbf z)))]$ | sharp, fast sampling | unstable, mode collapse |
| Diffusion | learn to reverse gradual noising | $\mathbb E\lVert\boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\mathbf x_t, t)\rVert^2$ | quality, diversity, controllable | many sampling steps (distillation fixes) |
| Flow matching | learn a velocity field from noise to data | regression on velocities | simpler training, fewer steps | newer |

!!! intuition "Diffusion in one paragraph"
    Add Gaussian noise to an image in many small steps until it is pure noise: $\mathbf x_t = \sqrt{\bar\alpha_t}\,\mathbf x_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon$. Train a network to predict the noise $\boldsymbol\epsilon$ from $\mathbf x_t$ and $t$. To generate, start from noise and repeatedly subtract the predicted noise. Text conditioning plus classifier-free guidance steers the denoising toward the prompt. Latent diffusion does it in a compressed VAE space to save compute.

## Reinforcement learning

![](figures/rl.svg)

$$Q(s,a) \leftarrow Q(s,a) + \alpha\Big[r + \gamma\max_{a'}Q(s',a') - Q(s,a)\Big]$$

**Intuition:** nudge the value of what you did toward what you got plus the best you expect next (temporal difference). **Example:** $Q = 0.5$, $r = 1$, $\gamma = 0.9$, $\max Q(s') = 0.6$, $\alpha = 0.5$: $0.5 + 0.5(1 + 0.54 - 0.5) = 1.02$.

| Method | Learns | Use |
|---|---|---|
| Q-learning / DQN | action values | discrete actions, games |
| Policy gradient (REINFORCE) | $\nabla J = \mathbb E[\nabla\log\pi(a\mid s)\,G_t]$ | continuous/stochastic policies |
| Actor-critic, PPO | policy + value baseline | robotics, RLHF |
| Multi-armed bandits | best arm under exploration | ads, recommendations, pricing; Thompson sampling, UCB |
| Contextual bandits | arm given features | personalisation with less risk than full RL |
| Offline RL | from logged data | when exploration is costly |

## Recommender systems

| Stage | Purpose | Methods |
|---|---|---|
| Candidate generation | millions → hundreds, fast | two-tower embeddings + ANN, item-item co-occurrence, popularity |
| Ranking | hundreds → tens, accurate | GBM / deep ranking (DLRM, DCN) on rich features |
| Re-ranking | business rules | diversity, freshness, fairness, dedup |
| Feedback | learn from behaviour | implicit signals (clicks, dwell), bandits for exploration |

$$\hat r_{ui} = \mathbf p_u^\top\mathbf q_i + b_u + b_i + \mu,\qquad \min\sum_{(u,i)\in\text{obs}}(r_{ui} - \hat r_{ui})^2 + \lambda\big(\lVert\mathbf p_u\rVert^2 + \lVert\mathbf q_i\rVert^2\big)$$

!!! pitfall "Pitfalls in recommenders"
    Cold start (new users/items: use content features, popularity); popularity bias and feedback loops (the model only learns from what it showed); offline metrics (NDCG) that don't predict online lift; leakage from future interactions in random splits: always split by time.

## Time series forecasting

![](figures/timeseries.svg)

| Approach | When | Notes |
|---|---|---|
| Naive / seasonal naive | always, as the baseline | surprisingly hard to beat |
| ETS / Holt-Winters | few series, clear trend/seasonality | statsforecast, statsmodels |
| ARIMA / SARIMAX | stationary after differencing, exogenous regressors | ACF/PACF, AIC |
| Prophet | business series with holidays | easy, often not the most accurate |
| GBM with lag features | many series, rich covariates | the M5 winner pattern; lags ≥ horizon |
| Deep (N-BEATS, TFT, PatchTST, foundation models) | many long series | needs data; evaluate against GBM |

@@code code/s11_timeseries.py@@
@@out code/out/s11_timeseries.txt@@

!!! pitfall "Pitfall: trees can't extrapolate a trend"
    The executed run shows LightGBM on the raw target losing to a naive forecast: the series trends upward and tree predictions are bounded by the training range. Modelling the residual from a level (or differencing, or detrending) fixes it. Rules: lags must be ≥ the forecast horizon (or forecast recursively), validate with rolling-origin splits, and compare against seasonal naive with a scale-free metric (MASE).

## Anomaly detection

![](figures/anomaly.svg)

| Method | Idea | Use |
|---|---|---|
| Z-score / robust z (MAD) | distance from centre in std units | univariate metrics |
| IQR fences | outside $Q_1 - 1.5\,$IQR, $Q_3 + 1.5\,$IQR | quick univariate |
| Isolation Forest | anomalies are isolated in few random splits | tabular, default choice |
| LOF | local density compared to neighbours | varying densities |
| One-class SVM | boundary around normal data | small, clean data |
| Autoencoder | high reconstruction error | images, sequences, high-dim |
| Forecast residuals | actual vs predicted outside interval | time series monitoring |

@@code code/s11_anomaly_rec_rl.py@@
@@out code/out/s11_anomaly_rec_rl.txt@@

## Responsible AI and fairness

| Criterion | Definition | Meaning |
|---|---|---|
| Demographic parity | $P(\hat Y = 1\mid A=a)$ equal across groups | equal selection rates |
| Equal opportunity | equal TPR across groups | qualified people treated equally |
| Equalised odds | equal TPR and FPR | errors balanced |
| Predictive parity | equal precision | a positive means the same for all |
| Calibration within groups | $P(Y=1\mid p, A)$ = $p$ | scores mean the same |

@@code code/s11_fairness.py@@
@@out code/out/s11_fairness.txt@@

!!! interview "Interview: can a model satisfy all fairness criteria at once?"
    No. When base rates differ between groups, calibration and equal error rates (equalised odds) cannot all hold simultaneously except for a perfect classifier (Kleinberg et al.; Chouldechova). Choose the criterion with stakeholders based on the harm you want to prevent, document it in a model card, and monitor it.

| Practice | What |
|---|---|
| Model cards / datasheets | intended use, data, metrics per group, limitations |
| Slice evaluation | metrics per segment (region, device, language, group) |
| Mitigation | rebalancing, reweighting, group-specific thresholds (legal review), constraints (fairlearn) |
| Privacy | minimisation, anonymisation, differential privacy, access control |
| Human oversight | appeal paths, human review for high-stakes decisions |

<div class="card" markdown="1">

## Cheat card: Applied AI

<div class="cols" markdown="1">

#### Vision
- Pretrained backbone + fine-tune; augment for deployment conditions
- Detection: mAP; segmentation: mIoU/Dice
- Zero-shot baselines: CLIP, SAM, VLMs

#### Generative
- Text: autoregressive Transformers
- Images: latent diffusion / flow matching
- VAE: smooth latents; GAN: fast, unstable

#### RL & bandits
- Most business "RL" is a bandit: Thompson sampling
- TD update: $Q \mathrel{+}= \alpha(r + \gamma\max Q' - Q)$
- PPO for RLHF and robotics

#### RecSys
- Retrieve (two-tower + ANN) → rank (GBM/DNN) → re-rank (rules)
- Split by time; watch feedback loops; A/B test online

#### Time series
- Seasonal naive baseline; MASE
- GBM with lags ≥ horizon; model residual for trends
- Rolling-origin validation

#### Anomalies
- Isolation Forest default; robust z for single metrics
- Evaluate with PR-AUC on labelled incidents

#### Fairness
- Measure per group: selection rate, TPR, FPR, precision
- 80% rule for disparate impact
- Criteria conflict: choose, document, monitor

</div>
</div>
