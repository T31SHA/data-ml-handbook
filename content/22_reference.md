# Reference: Formula Sheet, Interview Bank & Glossary

tagline: Everything in one place for the last ten minutes before an interview or design review: the formulas that matter, the questions that get asked, and an expanded glossary.

## Master formula sheet

| Area | Formula | In words |
|---|---|---|
| Mean, variance | $\bar x = \frac1n\sum x_i$, $s^2 = \frac{1}{n-1}\sum(x_i - \bar x)^2$ | centre; average squared spread |
| Standard error | $\text{SE} = s/\sqrt n$ | uncertainty of a mean |
| z-score | $z = (x - \mu)/\sigma$ | how many SDs from the mean |
| Bayes | $P(H\mid D) = P(D\mid H)P(H)/P(D)$ | update belief with evidence |
| Expectation | $\mathbb E[g(X)] = \sum g(x)p(x)$ | probability-weighted average |
| Covariance | $\text{Cov}(X,Y) = \mathbb E[XY] - \mathbb E[X]\mathbb E[Y]$ | joint variability |
| Correlation | $\rho = \text{Cov}(X,Y)/(\sigma_X\sigma_Y)$ | scale-free linear association |
| Normal pdf | $\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$ | the bell curve |
| Binomial | $\binom nk p^k(1-p)^{n-k}$ | $k$ successes in $n$ trials |
| Poisson | $\lambda^k e^{-\lambda}/k!$ | counts at rate $\lambda$ |
| Entropy | $H = -\sum p\log p$ | average surprise |
| Cross-entropy | $H(p,q) = -\sum p\log q$ | cost of coding $p$ with $q$ |
| KL divergence | $\sum p\log(p/q)$ | extra cost; ≥ 0, asymmetric |
| OLS | $\hat{\boldsymbol\beta} = (X^\top X)^{-1}X^\top\mathbf y$ | least-squares solution |
| Ridge | $(X^\top X + \lambda I)^{-1}X^\top\mathbf y$ | shrunk, always invertible |
| Sigmoid | $\sigma(z) = 1/(1 + e^{-z})$ | logit → probability |
| Softmax | $e^{z_k}/\sum_j e^{z_j}$ | logits → distribution |
| Log-loss gradient | $\nabla_{\mathbf w} = X^\top(\mathbf p - \mathbf y)/n$ | prediction minus target |
| Gradient descent | $\theta \leftarrow \theta - \eta\nabla\mathcal L$ | step downhill |
| Adam | $\theta \leftarrow \theta - \eta\,\hat m/(\sqrt{\hat v} + \epsilon)$ | momentum + per-parameter scale |
| Bias–variance | $\text{MSE} = \text{bias}^2 + \text{var} + \sigma^2$ | three sources of error |
| Gini | $1 - \sum p_k^2$ | node impurity |
| Information gain | $H(\text{parent}) - \sum \frac{n_c}{n}H(c)$ | purity improvement |
| Precision, recall | $\frac{TP}{TP+FP}$, $\frac{TP}{TP+FN}$ | right when flagging; caught positives |
| F1 | $2PR/(P+R)$ | harmonic mean |
| AUC | $P(s_+ > s_-)$ | ranking quality |
| R² | $1 - SS_{res}/SS_{tot}$ | variance explained |
| Attention | $\text{softmax}(QK^\top/\sqrt{d_k})V$ | content-based weighted average |
| Conv output | $\lfloor(H + 2p - k)/s\rfloor + 1$ | spatial size after a conv |
| LoRA | $W_0 + \frac{\alpha}{r}BA$ | low-rank update |
| KV-cache | $2Lh_{kv}d_{head}\,n\,b$ bytes | memory per sequence |
| Train FLOPs | $\approx 6ND$ | params × tokens × 6 |
| Sample size | $n \approx 16\sigma^2/\delta^2$ per arm | α 0.05, power 0.8 |
| CUPED | $Y - \theta(X - \bar X)$, $\theta = \text{Cov}/\text{Var}$ | variance ↓ by $\rho^2$ |
| PSI | $\sum(a - e)\ln(a/e)$ | distribution shift |
| DiD | $(Y^T_1 - Y^T_0) - (Y^C_1 - Y^C_0)$ | effect net of shared trend |
| IPW | $\frac1n\sum\big[\frac{TY}{e} - \frac{(1-T)Y}{1-e}\big]$ | reweight to a pseudo-experiment |
| Q-learning | $Q \leftarrow Q + \alpha(r + \gamma\max Q' - Q)$ | temporal-difference update |
| Cosine similarity | $\frac{\mathbf a\cdot\mathbf b}{\lVert\mathbf a\rVert\lVert\mathbf b\rVert}$ | angle between embeddings |

## Loss functions at a glance

| Loss | Formula | Optimal prediction | Use |
|---|---|---|---|
| Squared (MSE) | $(y - \hat y)^2$ | conditional mean | regression default |
| Absolute (MAE) | $\lvert y - \hat y\rvert$ | conditional median | robust regression |
| Huber | quadratic if $\lvert e\rvert \leq \delta$, else linear | between mean and median | outliers |
| Pinball | $\max(qe, (q - 1)e)$ | $q$-quantile | prediction intervals |
| Log-loss | $-[y\log p + (1-y)\log(1-p)]$ | true probability | classification |
| Hinge | $\max(0, 1 - y\hat y)$ | sign | SVM |
| Focal | $-(1 - p_t)^\gamma\log p_t$ | | heavy imbalance, detection |
| Poisson | $\hat y - y\log\hat y$ | conditional rate | counts |
| Contrastive (InfoNCE) | $-\log\frac{e^{s(q,k^+)/\tau}}{\sum_k e^{s(q,k)/\tau}}$ | | embeddings, CLIP |
| KL | $\sum p\log(p/q)$ | | VAEs, distillation, RLHF penalty |

## Interview question bank

| Question | Crisp answer |
|---|---|
| Bias vs variance? | Bias: error from wrong assumptions (underfit). Variance: sensitivity to the training sample (overfit). Total error = bias² + variance + noise. |
| Why regularise? | Penalising complexity trades a little bias for a big variance reduction; equivalent to a prior on weights. |
| L1 vs L2? | L1 → sparse (feature selection), L2 → small, spread weights; elastic net for correlated groups. |
| How does a random forest reduce variance? | Bootstrap + random feature subsets decorrelate trees; averaging cuts variance by the uncorrelated part. |
| Gradient boosting in one sentence? | Sequentially fit trees to the negative gradient of the loss and add them with a small learning rate. |
| Precision vs recall trade-off? | Raising the threshold increases precision and lowers recall; choose by the cost of FP vs FN. |
| Why is accuracy misleading? | With 1% positives, predicting all negatives scores 99%; use PR-AUC, recall@precision, cost. |
| What is cross-validation for? | Estimating generalisation with less variance than a single split, and selecting hyperparameters. |
| What is data leakage? | Information unavailable at prediction time entering training (future data, target proxies, preprocessing on all data). |
| Explain p-value. | P(data this extreme or more ∣ null true); not the probability the null is true. |
| Type I vs II error? | False positive (α) vs false negative (β); power = 1 − β. |
| How long should an A/B test run? | Until the precomputed sample size is reached, in whole weeks; no peeking unless using sequential methods. |
| Correlation vs causation? | Correlation can come from confounders, reverse causation, or selection; causation needs experiments or causal assumptions. |
| Why divide attention by √d? | Dot products of random d-dim vectors have variance d; scaling keeps softmax out of saturation. |
| Vanishing gradients: causes and fixes? | Repeated multiplication by small Jacobians; ReLU, residuals, normalisation, good init, LSTM gates. |
| BatchNorm vs LayerNorm? | BN normalises across the batch (CNNs), LN across features per sample (Transformers, small batches). |
| What does dropout do? | Randomly zeros units in training: an ensemble of sub-networks, reduces co-adaptation. |
| Fine-tuning vs RAG? | RAG adds knowledge at query time with citations; fine-tuning changes behaviour/format. |
| What is LoRA? | Freeze the weights and learn a low-rank update BA; ~1% trainable parameters, mergeable. |
| How does the KV-cache work? | Store past keys/values so each new token only computes its own; memory grows with context. |
| How do you evaluate RAG? | Retrieval (recall@k, MRR) and generation (faithfulness, relevance, correctness) on a golden set. |
| Batch vs streaming? | Batch: simpler, cheaper, hours of latency; streaming when value decays in seconds. |
| What is idempotency and why does it matter? | Re-running gives the same result; makes retries and backfills safe. |
| Star schema? | Central fact table of events/measures linked to dimension tables describing who/what/where/when. |
| SCD Type 2? | Keep history by adding a new row per change with validity dates and a current flag. |
| Partitioning vs bucketing/clustering? | Partitioning splits files by a column for pruning; bucketing/clustering co-locates rows by key for joins/filters. |
| Parquet vs CSV? | Columnar, typed, compressed, with statistics for skipping; CSV is row text without types. |
| Spark shuffle? | Redistribution of data across executors for wide operations; the main cost; creates stage boundaries. |
| How to handle data skew? | Broadcast small side, salting, AQE skew join, isolate hot keys. |
| What is drift and how to detect it? | Change in P(X), P(Y) or P(Y∣X); PSI/KS on features, prediction distribution, delayed performance. |
| Training/serving skew? | Feature computation differs between training and serving; fix with shared definitions and feature stores. |
| Canary vs shadow? | Shadow: no user impact, compare outputs; canary: a small share of real traffic uses the new model. |

<div class="glossary" markdown="1">

## Glossary

A/B test
:   Randomised controlled experiment comparing variants on a metric.

Activation function
:   Non-linearity applied after a linear layer (ReLU, GELU, sigmoid).

Adam / AdamW
:   Adaptive optimiser with momentum and per-parameter scaling; AdamW decouples weight decay.

AQE
:   Spark Adaptive Query Execution: re-optimises joins and partitions at runtime.

Attention
:   Weighted average of values, with weights from query–key similarity.

AUC (ROC)
:   Probability a random positive scores higher than a random negative.

Autoencoder
:   Network trained to reconstruct its input through a bottleneck.

Avro
:   Row-based binary format with an embedded schema; common in Kafka.

Backfill
:   Re-running a pipeline for historical dates.

Backpropagation
:   Chain rule applied backwards through the computation graph to get gradients.

Bagging
:   Training models on bootstrap samples and averaging them.

Batch normalisation
:   Normalising activations using batch statistics.

Bayes' theorem
:   Posterior ∝ likelihood × prior.

Benjamini–Hochberg
:   Procedure controlling the false discovery rate across many tests.

Bias (statistical)
:   Systematic difference between an estimator's expectation and the truth.

Bias–variance trade-off
:   Simpler models err systematically; complex ones vary with the sample.

BM25
:   Lexical ranking function based on term frequency and document length.

Boosting
:   Sequentially adding weak learners that correct previous errors.

Bootstrap
:   Resampling with replacement to estimate uncertainty.

BPE
:   Byte-pair encoding: subword tokenizer built by merging frequent pairs.

Broadcast join
:   Sending a small table to every executor to avoid shuffling the big one.

Calibration
:   Agreement between predicted probabilities and observed frequencies.

Catalyst
:   Spark SQL's query optimiser.

CDC
:   Change data capture: streaming inserts, updates and deletes from a database log.

Central limit theorem
:   Means of many independent samples are approximately normal.

Chunking
:   Splitting documents into retrievable pieces for RAG.

Class imbalance
:   One class far rarer than others.

Collider
:   Variable caused by two others; conditioning on it creates spurious association.

Concept drift
:   Change in the relationship P(Y∣X) over time.

Confidence interval
:   Range built by a procedure that covers the true value at a stated rate.

Confounder
:   Variable affecting both treatment and outcome.

Confusion matrix
:   Table of TP, FP, FN, TN.

Consumer group
:   Kafka consumers sharing the partitions of a topic.

Contrastive learning
:   Learning embeddings by pulling positives together and pushing negatives apart.

Convolution
:   Sliding a learned kernel over input to detect local patterns.

Covariate shift
:   Change in the input distribution P(X).

Cross-entropy
:   Loss measuring the gap between true and predicted distributions.

Cross-validation
:   Repeated train/validation splits to estimate generalisation.

CUPED
:   Variance reduction for experiments using pre-period covariates.

Curse of dimensionality
:   Data becomes sparse and distances less informative as dimensions grow.

DAG
:   Directed acyclic graph: Airflow pipelines, Spark plans, causal diagrams.

Data contract
:   Versioned agreement on schema, semantics and quality between producer and consumer.

Data lake
:   Raw files in object storage, schema-on-read.

Data mesh
:   Domain-owned data products with federated governance.

Data warehouse
:   Managed analytical database, schema-on-write.

dbt
:   SQL-based transformation framework with testing, docs and lineage.

DBSCAN
:   Density-based clustering that labels sparse points as noise.

Delta Lake
:   Open table format adding ACID transactions to Parquet.

Diffusion model
:   Generative model that learns to reverse a gradual noising process.

Difference-in-differences
:   Causal estimate comparing changes over time between treated and control groups.

Dimension table
:   Descriptive attributes (customer, product) in a star schema.

Distillation
:   Training a small model to mimic a larger one.

Dropout
:   Randomly zeroing activations during training.

Early stopping
:   Stopping training when validation performance stops improving.

ELT
:   Extract, load, then transform inside the warehouse.

Embedding
:   Dense vector representation of an item, word or user.

Ensemble
:   Combination of multiple models.

Entropy
:   Expected information (surprise) of a distribution.

Epoch
:   One full pass over the training data.

ETL
:   Extract, transform, then load.

Event time
:   When an event happened, as opposed to when it was processed.

Explainability (XAI)
:   Methods to understand model predictions (SHAP, PDP, permutation importance).

F1 score
:   Harmonic mean of precision and recall.

Fact table
:   Events and measures at a defined grain.

Feature store
:   System serving consistent features for training and inference.

Few-shot prompting
:   Including examples in the prompt.

Fine-tuning
:   Continuing training of a pretrained model on task data.

FlashAttention
:   IO-aware fused attention kernel reducing memory traffic.

GBM
:   Gradient boosting machine.

GELU
:   Smooth activation $x\Phi(x)$ used in Transformers.

Gini impurity
:   Probability of misclassifying a random sample by random labelling.

GQA
:   Grouped-query attention: query heads share key/value heads.

Gradient
:   Vector of partial derivatives; direction of steepest ascent.

Gradient clipping
:   Limiting gradient norm to prevent explosions.

Grain
:   What one row of a table represents.

Guardrails
:   Checks on LLM inputs/outputs for safety, format and policy.

Hallucination
:   Fluent but unsupported model output.

HNSW
:   Graph-based approximate nearest-neighbour index.

Hyperparameter
:   Setting chosen before training (learning rate, depth).

Iceberg
:   Engine-neutral open table format for lakehouses.

Idempotent
:   Producing the same result when run multiple times.

Imputation
:   Filling missing values.

Instrumental variable
:   Variable affecting the outcome only through the treatment.

Isolation Forest
:   Anomaly detector isolating points with random splits.

k-means
:   Clustering by iteratively assigning to and recomputing centroids.

Kafka
:   Distributed append-only log for event streaming.

KL divergence
:   Asymmetric measure of how one distribution differs from another.

KV-cache
:   Stored keys/values of past tokens to speed autoregressive decoding.

L1 / L2 regularisation
:   Penalties on absolute / squared weights.

Lakehouse
:   Warehouse features on open table formats in object storage.

Layer normalisation
:   Normalising activations across features per example.

Learning rate
:   Step size of gradient descent.

Leakage
:   Information from outside the training-time context contaminating training.

Lineage
:   Record of where data came from and how it was transformed.

LLM-as-judge
:   Using a language model to grade outputs against a rubric.

Logit
:   Raw score before sigmoid/softmax; log-odds.

LoRA
:   Low-rank adaptation for parameter-efficient fine-tuning.

LSTM
:   Gated recurrent unit with a cell state for long memory.

MAP estimate
:   Mode of the posterior; MLE plus a prior.

MCC
:   Matthews correlation coefficient; balanced classification metric.

Medallion architecture
:   Bronze/silver/gold layering of data quality.

MERGE
:   SQL upsert: update matches, insert new rows.

Mixed precision
:   Training with 16-bit floats plus fp32 master weights.

MLE
:   Parameters maximising the likelihood of observed data.

MLflow
:   Open-source experiment tracking and model registry.

Model registry
:   Versioned store of models with stages and lineage.

Multicollinearity
:   Strong correlation among predictors, inflating coefficient variance.

Naive Bayes
:   Classifier assuming conditional independence of features.

NDCG
:   Ranking metric with graded relevance and position discount.

Nested cross-validation
:   Outer CV for evaluation around inner CV for tuning.

Normal equation
:   Closed-form least-squares solution.

Offset (Kafka)
:   Position of a message in a partition.

One-hot encoding
:   Binary column per category.

Overfitting
:   Fitting noise; poor generalisation.

p-value
:   Probability of data at least as extreme under the null.

p-hacking
:   Trying analyses until something is significant.

Parquet
:   Columnar compressed file format for analytics.

Partition pruning
:   Skipping data partitions excluded by a filter.

PCA
:   Projection onto directions of maximum variance.

Perplexity
:   Exponentiated average negative log-likelihood per token.

Point-in-time join
:   Joining features as they were at the event timestamp.

Power
:   Probability of detecting a true effect.

Precision
:   Share of predicted positives that are correct.

Prompt injection
:   Malicious instructions hidden in model inputs or retrieved content.

Propensity score
:   Probability of treatment given covariates.

PSI
:   Population stability index; drift measure.

Quantization
:   Representing weights/activations with fewer bits.

RAG
:   Retrieval-augmented generation.

Random forest
:   Bagged, feature-subsampled decision trees.

Recall
:   Share of actual positives that are found.

Regularisation
:   Constraining a model to reduce overfitting.

Reranker
:   Model that rescores retrieved candidates precisely (often a cross-encoder).

Residual connection
:   Adding a layer's input to its output.

RLHF
:   Reinforcement learning from human feedback.

RoPE
:   Rotary position embeddings encoding relative positions.

SCD
:   Slowly changing dimension; Type 2 keeps history.

Schema evolution
:   Changing a schema compatibly over time.

Shadow deployment
:   Running a new model on live traffic without using its output.

SHAP
:   Shapley-value attributions of a prediction to features.

Shuffle (Spark)
:   Redistribution of data across the cluster for wide operations.

Simpson's paradox
:   A trend reverses when groups are combined.

Skew (data)
:   Uneven distribution of keys causing straggler tasks.

Softmax
:   Turns logits into a probability distribution.

SRM
:   Sample ratio mismatch: experiment allocation differs from design.

Standard error
:   Standard deviation of an estimator's sampling distribution.

Star schema
:   Fact table surrounded by dimension tables.

Stratified sampling
:   Preserving class proportions in splits.

SVM
:   Maximum-margin classifier, optionally with kernels.

t-SNE
:   Non-linear 2-D visualisation preserving local neighbourhoods.

Target encoding
:   Replacing a category with a smoothed target mean (out-of-fold).

Temperature
:   Scaling of logits controlling sampling randomness.

Tokenizer
:   Converts text into model tokens.

Training/serving skew
:   Mismatch between features in training and in production.

Transfer learning
:   Reusing a pretrained model for a new task.

Transformer
:   Architecture of attention and feed-forward blocks with residuals.

Type I / II error
:   False positive / false negative.

UMAP
:   Manifold-learning method for visualisation and reduction.

Underfitting
:   Model too simple to capture the signal.

Vector database
:   Store with approximate nearest-neighbour search over embeddings.

Watermark (streaming)
:   Threshold of event time after which windows close.

Window function
:   SQL function computed over a set of rows related to the current row.

Z-order
:   Multi-column data clustering for file skipping.

</div>
