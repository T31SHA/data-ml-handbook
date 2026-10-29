# Classical Machine Learning

tagline: Every workhorse algorithm as a profile card: intuition, math, when to use it and when not to, assumptions, complexity, failure modes, hyperparameters with sane defaults, and executed code.

## The learning problem in one equation

$$\hat f = \arg\min_{f\in\mathcal F}\ \underbrace{\frac1n\sum_{i=1}^n L\big(y_i, f(\mathbf x_i)\big)}_{\text{empirical risk}} + \underbrace{\lambda\,\Omega(f)}_{\text{complexity penalty}}$$

Every model below is a choice of hypothesis class $\mathcal F$, loss $L$, penalty $\Omega$ and optimiser. **Intuition:** fit the data, but pay a price for complexity. The price $\lambda$ is set by validation, never by the training loss.

![](figures/bias_variance.svg)

![](figures/fits.svg)

!!! rule "Rule of thumb: the default ladder for tabular data"
    1) Dummy baseline. 2) Regularised linear / logistic model (scaled). 3) Gradient-boosted trees (LightGBM / HistGB) with early stopping. 4) Only then: ensembles, deep tabular nets, or feature work. On tabular data, GBMs win most benchmarks; feature engineering usually beats model switching.

## Decision boundaries side by side

![](figures/boundaries.png)

Same data (two noisy moons), eight models. Linear models draw one hyperplane; kNN and RBF-SVM follow local geometry; trees cut axis-aligned rectangles; ensembles smooth those rectangles; naive Bayes draws quadratic boundaries from per-class Gaussians.

## Benchmarking the zoo (executed)

@@code code/s07_models.py@@
@@out code/out/s07_models.txt@@

!!! tip "Pro tip: read benchmarks honestly"
    On this small, nearly linearly separable dataset (569 rows), scaled logistic regression and SVM tie with boosting at a fraction of the cost. The differences are within one standard deviation across folds. Prefer the simplest model inside the noise band.

## Linear models

<div class="profile" markdown="1">

### Linear regression (OLS, Ridge, Lasso, Elastic Net)

| Facet | Detail |
|---|---|
| Intuition | Best-fitting hyperplane; each coefficient is a "slope holding others fixed". |
| Math | OLS $\hat{\boldsymbol\beta} = (X^\top X)^{-1}X^\top\mathbf y$. Ridge $(X^\top X + \lambda I)^{-1}X^\top\mathbf y$. Lasso: $\min \lVert\mathbf y - X\boldsymbol\beta\rVert^2/2n + \alpha\lVert\boldsymbol\beta\rVert_1$ (coordinate descent, no closed form). |
| Use when | Baseline; interpretability; few rows; extrapolation-ish trends; many correlated features (Ridge); feature selection (Lasso). |
| Avoid when | Strong interactions / non-linearity without feature work; heavy outliers (use Huber). |
| Assumptions | Linearity, independent errors, constant variance; for inference also normal errors and no omitted confounders. |
| Complexity | Normal equation $O(np^2 + p^3)$; SGD $O(np)$ per epoch; predict $O(p)$. |
| Failure modes | Multicollinearity → unstable coefficients; unscaled features distort penalties; Lasso picks arbitrarily among correlated features. |
| Hyperparameters | Ridge `alpha` (1.0; search log-grid 1e-3..1e3), Lasso `alpha`, ElasticNet `l1_ratio` (0.5). Always `StandardScaler` first. |
| Code | `make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))` |

</div>

![](figures/l1l2.svg)

![](figures/reg_paths.svg)

!!! intuition "Why L1 gives zeros and L2 does not"
    The L1 ball has corners on the axes; expanding loss contours usually touch a corner first, where some coefficients are exactly 0. The L2 ball is round, so the touch point is generically off-axis: coefficients shrink but stay non-zero. Bayesian view: L2 = Gaussian prior, L1 = Laplace prior (spikier at 0).

<div class="profile" markdown="1">

### Logistic regression

| Facet | Detail |
|---|---|
| Intuition | Linear score $z = \mathbf w^\top\mathbf x + b$ squashed into a probability; boundary where $z = 0$. |
| Math | $p = \sigma(z) = 1/(1+e^{-z})$; loss $-[y\log p + (1-y)\log(1-p)]$; gradient $X^\top(\mathbf p - \mathbf y)/n$ (derived in Calculus). Multiclass: softmax. |
| Use when | Baseline classifier, calibrated probabilities, regulated domains (scorecards), sparse text (with TF-IDF). |
| Avoid when | Complex non-linear boundaries without feature engineering. |
| Assumptions | Log-odds linear in features; independent observations; no perfect separation (else weights → ∞; regularisation fixes it). |
| Complexity | $O(np)$ per iteration (L-BFGS / SAGA); predict $O(p)$. |
| Failure modes | Unscaled features slow convergence (`ConvergenceWarning`); `C` is the **inverse** of regularisation strength. |
| Hyperparameters | `C` = 1.0 (search 1e-3..1e3), `penalty` l2 / l1 (`solver="saga"` or `liblinear`), `class_weight`, `max_iter` ≥ 1000. |
| Code | `make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))` |

</div>

![](figures/sigmoid.svg)

!!! example "Worked example: reading an odds ratio"
    Coefficient for "has_premium" = 0.7 in a churn model. $e^{0.7} = 2.01$: premium users have about twice the **odds** of retention, holding other features fixed. At 50% baseline odds (1:1) that's 2:1, i.e. 67% probability.

## Instance-based and probabilistic models

<div class="profile" markdown="1">

### k-Nearest Neighbours

| Facet | Detail |
|---|---|
| Intuition | Predict like your $k$ closest neighbours vote (or average). No training. |
| Math | $\hat y = \text{mode}\{y_j : j\in N_k(\mathbf x)\}$; distance-weighted variant $w_j = 1/d_j$. |
| Use when | Small, low-dimensional data; baselines; recommendation-by-similarity; as part of ANN retrieval. |
| Avoid when | High dimensions (curse of dimensionality: all distances look equal), big $n$ at query time, mixed feature types. |
| Assumptions | Nearby points share labels; meaningful distance metric; features scaled. |
| Complexity | Train $O(1)$; predict $O(np)$ brute force, ~$O(\log n)$ with KD/ball trees in low $d$; use FAISS/HNSW at scale. |
| Hyperparameters | `n_neighbors` 5 (odd; tune 3..50), `weights` uniform/distance, `metric` (euclidean, cosine). |
| Code | `make_pipeline(StandardScaler(), KNeighborsClassifier(7))` |

</div>

<div class="profile" markdown="1">

### Naive Bayes

| Facet | Detail |
|---|---|
| Intuition | Apply Bayes' rule assuming features are independent given the class. Wrong assumption, surprisingly good ranking. |
| Math | $P(c\mid\mathbf x) \propto P(c)\prod_j P(x_j\mid c)$; compute in log space. Gaussian, Multinomial (counts), Bernoulli (binary) variants. |
| Use when | Text classification baselines, spam, tiny data, extremely fast training / streaming updates. |
| Avoid when | You need calibrated probabilities (they are pushed to 0/1) or features are strongly dependent. |
| Complexity | Train $O(np)$, predict $O(cp)$. |
| Hyperparameters | `alpha` Laplace smoothing (1.0) for Multinomial; `var_smoothing` (1e-9) for Gaussian. |

</div>

<div class="profile" markdown="1">

### Support Vector Machines

| Facet | Detail |
|---|---|
| Intuition | Widest possible street between classes; only the points on the edge (support vectors) matter. Kernels bend the street. |
| Math | $\min \frac12\lVert\mathbf w\rVert^2 + C\sum\xi_i$ s.t. $y_i(\mathbf w^\top\mathbf x_i + b) \geq 1 - \xi_i$. Equivalent hinge loss $\max(0, 1 - y f(\mathbf x))$. RBF kernel $k(\mathbf x,\mathbf x') = e^{-\gamma\lVert\mathbf x - \mathbf x'\rVert^2}$. |
| Use when | Medium data (≤ ~50k rows), high-dimensional sparse text (LinearSVC), clear margins. |
| Avoid when | Large $n$ for kernel SVM; you need probabilities (Platt scaling is expensive and approximate). |
| Complexity | Kernel training $O(n^2)$–$O(n^3)$; memory $O(n^2)$ kernel cache. LinearSVC $O(np)$. |
| Failure modes | Unscaled features; $\gamma$ too large → memorises each point; too small → linear. |
| Hyperparameters | `C` 1.0 and `gamma` "scale": grid on log scale jointly (C 0.1..100, γ 1e-4..1). |

</div>

![](figures/svm.svg)

## Trees and ensembles

<div class="profile" markdown="1">

### Decision tree (CART)

| Facet | Detail |
|---|---|
| Intuition | A flowchart of yes/no questions that greedily makes each child purer. |
| Math | Split maximising impurity decrease; Gini $1 - \sum p_k^2$, entropy $-\sum p_k\log_2 p_k$, regression: variance (MSE). |
| Use when | Explainable rules, mixed data types, building block for ensembles. |
| Avoid when | Stand-alone for accuracy: high variance, unstable to small data changes. |
| Assumptions | None about scale or distribution; no need to scale features. |
| Complexity | Train $O(p\,n\log n)$ per level; predict $O(\text{depth})$. |
| Hyperparameters | `max_depth` (3–8), `min_samples_leaf` (≥ 5–20), `ccp_alpha` (cost-complexity pruning). |

</div>

![](figures/tree.svg)

!!! example "Worked example: Gini gain"
    Parent: 10 yes / 10 no, Gini $= 1 - (0.5^2 + 0.5^2) = 0.5$. Split → left 8/2 (Gini $1 - 0.64 - 0.04 = 0.32$), right 2/8 (0.32). Weighted child Gini $= 0.32$, gain $= 0.18$. The tree tries every feature and threshold and takes the best gain.

<div class="profile" markdown="1">

### Random forest (bagging)

| Facet | Detail |
|---|---|
| Intuition | Many deep, decorrelated trees vote; averaging cancels their individual variance. |
| Math | Ensemble variance $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$: more trees kill the second term; random feature subsets (`max_features`) lower the correlation $\rho$. |
| Use when | Strong, low-tuning baseline; robust to outliers and noise; out-of-bag (OOB) error for free. |
| Avoid when | Extrapolation (predictions bounded by training range); huge data with latency limits (big models). |
| Complexity | Train $O(B\,p'\,n\log n)$, parallel; predict $O(B\cdot\text{depth})$. |
| Failure modes | Impurity-based importance biased toward high-cardinality features → use permutation importance. |
| Hyperparameters | `n_estimators` 300–1000, `max_features` "sqrt" (clf) / 1.0 or 0.33 (reg), `min_samples_leaf` 1–5, `n_jobs=-1`. |

</div>

<div class="profile" markdown="1">

### Gradient boosting (XGBoost, LightGBM, CatBoost, HistGB)

| Facet | Detail |
|---|---|
| Intuition | Add small trees one at a time, each fitting the mistakes (negative gradient) of the current ensemble. |
| Math | $F_m(\mathbf x) = F_{m-1}(\mathbf x) + \eta\,h_m(\mathbf x)$ where $h_m$ fits $r_i = -\partial L/\partial F(\mathbf x_i)$. XGBoost leaf weight $w^* = -\frac{\sum g_i}{\sum h_i + \lambda}$ and gain uses second-order (Hessian) info. |
| Use when | Tabular data, almost always the accuracy leader; native missing values; mixed types; ranking (LambdaMART). |
| Avoid when | Images / text / audio (deep learning), tiny data with extrapolation needs. |
| Complexity | Histogram-based: $O(n\,p)$ per tree for binning plus $O(\text{bins}\cdot p)$ for splits; GPU friendly. |
| Failure modes | Overfits with too many rounds → always early-stop on validation; leakage through target encoding; high LR. |
| Hyperparameters | `learning_rate` 0.05 (0.01–0.1), `n_estimators` high + early stopping (50 rounds), `num_leaves` 31 (LGBM) / `max_depth` 6 (XGB), `min_child_samples` 20, `subsample` 0.8, `colsample_bytree` 0.8, `reg_lambda` 1. |

</div>

![](figures/ensembles.svg)

!!! tip "Pro tip: CatBoost vs LightGBM vs XGBoost"
    LightGBM: fastest, leaf-wise growth (watch `num_leaves`). XGBoost: mature, great GPU, depth-wise. CatBoost: best with many categorical features (ordered target statistics avoid leakage), strong defaults. sklearn HistGB: no extra dependency, native categorical support. Differences after tuning are usually small.

!!! interview "Interview: bagging vs boosting"
    Bagging trains independent high-variance learners in parallel and averages them (reduces variance). Boosting trains weak, high-bias learners sequentially, each correcting the last (reduces bias, can overfit). Stacking trains a meta-model on out-of-fold predictions of diverse base models.

## Unsupervised learning

<div class="profile" markdown="1">

### k-means

| Facet | Detail |
|---|---|
| Intuition | Place $k$ centres; assign each point to its nearest; move centres to the mean; repeat. |
| Math | Minimise inertia $\sum_i \lVert\mathbf x_i - \boldsymbol\mu_{c(i)}\rVert^2$ (Lloyd's algorithm; converges to a local optimum). k-means++ seeding. |
| Use when | Spherical, similar-size clusters; vector quantisation; customer segments as a first pass. |
| Avoid when | Non-convex shapes, very different densities or sizes, outliers, categorical data (use k-modes). |
| Complexity | $O(n\,k\,p)$ per iteration; MiniBatchKMeans for big data. |
| Hyperparameters | `n_clusters` (elbow, silhouette, business need), `n_init` 10, scale features first. |

</div>

![](figures/kmeans.svg)

![](figures/elbow.svg)

<div class="profile" markdown="1">

### DBSCAN / HDBSCAN

| Facet | Detail |
|---|---|
| Intuition | Clusters are dense regions; a point with ≥ `min_samples` neighbours within `eps` is a core point; sparse points are noise. |
| Use when | Arbitrary shapes, unknown $k$, outlier detection, geospatial data. |
| Avoid when | Clusters of very different densities (DBSCAN; HDBSCAN handles this), high dimensions. |
| Complexity | $O(n\log n)$ with a spatial index, $O(n^2)$ worst case. |
| Hyperparameters | `eps` from the k-distance plot knee, `min_samples` ≈ 2·dims; HDBSCAN `min_cluster_size`. |

</div>

![](figures/dbscan.svg)

<div class="profile" markdown="1">

### Gaussian mixture models (EM)

| Facet | Detail |
|---|---|
| Intuition | Soft k-means: each cluster is a Gaussian with its own shape; points get membership probabilities. |
| Math | $p(\mathbf x) = \sum_k \pi_k\,\mathcal N(\mathbf x\mid\boldsymbol\mu_k,\Sigma_k)$. E-step: responsibilities $\gamma_{ik}$; M-step: weighted means/covariances. Choose $k$ by BIC. |
| Use when | Elliptical clusters, density estimation, soft assignments, anomaly scores via low likelihood. |
| Hyperparameters | `n_components`, `covariance_type` full/diag/tied/spherical, `n_init`. |

</div>

### Dimensionality reduction

| Method | Preserves | Speed | Use for | Caveat |
|---|---|---|---|---|
| PCA | global variance (linear) | very fast | compression, decorrelation, preprocessing | linear only; scale first |
| Truncated SVD | same, works on sparse | fast | TF-IDF (LSA), recommender factors | no centering |
| t-SNE | local neighbourhoods | slow | 2-D visualisation only | distances/cluster sizes meaningless; tune perplexity 5–50 |
| UMAP | local + some global | fast | visualisation, features for clustering | hyperparameter-sensitive (`n_neighbors`, `min_dist`) |
| Autoencoder | non-linear manifold | GPU | images, anomaly detection | needs data and tuning |
| LDA (linear discriminant) | class separation | fast | supervised projection | ≤ classes − 1 dims |

![](figures/pca.svg)

@@code code/s07_unsup.py@@
@@out code/out/s07_unsup.txt@@

!!! pitfall "Pitfall: k-means on moons"
    The output above: k-means scores ARI 0.255 on two moons because it can only draw a straight boundary between two centroids. Density-based (DBSCAN, HDBSCAN) and single-linkage clustering recover the shapes perfectly. Pick the algorithm whose cluster definition matches your data's geometry.

## Imbalanced data

@@code code/s07_imbalance.py@@
@@out code/out/s07_imbalance.txt@@

| Technique | How | Note |
|---|---|---|
| Right metric | PR-AUC, recall@precision, cost | accuracy is useless at 1% positives (above: 98.9% for "always 0") |
| Threshold tuning | choose on validation by cost / F1 | often the biggest win (F1 0.156 → 0.268 above) |
| Class weights | `class_weight="balanced"`, `scale_pos_weight` | distorts probabilities; recalibrate |
| Resampling | undersample majority; SMOTE (inside CV folds only) | rarely beats weights + threshold for GBMs |
| Anomaly framing | Isolation Forest, autoencoders | when positives are extremely rare/unlabelled |
| More signal | better features, more positive labels | the real fix |

!!! pitfall "Pitfall: class weights hurt PR-AUC here"
    In the executed run, `class_weight="balanced"` raised ROC-AUC (0.796 → 0.817) but halved PR-AUC (0.268 → 0.154). Reweighting changes what the model optimises; it is not a free lunch. Always compare on the metric that matches the business cost.

## Feature engineering that pays

| Data | Technique | Watch out |
|---|---|---|
| Numeric skewed | log1p, Box-Cox / Yeo-Johnson, quantile transform | fit on train only |
| Numeric for linear models | standardise, splines, interactions, binning | trees don't need scaling |
| Low-card categorical | one-hot (`handle_unknown="ignore"`) | sparse explosion |
| High-card categorical | target encoding (out-of-fold, smoothed), hashing, embeddings | leakage without OOF |
| Dates | year, month, dow, hour, holiday, cyclical sin/cos, "days since" | time zones |
| Text | TF-IDF (1–2 grams), sentence embeddings | vocab fit on train |
| Aggregates | per-entity rolling counts/means (RFM) | point-in-time correctness |
| Missing | indicator column + impute; GBMs handle natively | missingness can be signal |
| Geo | haversine distance, clusters, H3 cells | |

!!! example "Worked example: smoothed target encoding"
    Category "Mombasa" has 4 rows with mean target 0.75; global mean 0.20; smoothing $m = 10$. Encoded value $= \frac{n\bar y_c + m\bar y}{n + m} = \frac{4\cdot0.75 + 10\cdot0.20}{14} = 0.357$. Rare categories shrink toward the global mean (Beta-Binomial posterior mean from the Probability section).

<div class="card" markdown="1">

## Cheat card: Classical ML

<div class="cols" markdown="1">

#### Pick a model
- Tabular, accuracy → LightGBM / XGBoost / CatBoost (early stop)
- Need explanation / tiny data → logistic / linear (+ splines)
- Text baseline → TF-IDF + logistic / LinearSVC / NB
- Many features, few rows → Ridge / Lasso / ElasticNet
- Non-convex clusters → HDBSCAN; spherical → k-means
- Visualise high-dim → UMAP / t-SNE (never cluster on t-SNE)

#### Scaling needed?
- Yes: linear, logistic, SVM, kNN, k-means, PCA, neural nets
- No: trees, RF, GBMs

#### Sane defaults
- Logistic: C=1, max_iter=2000
- RF: 500 trees, max_features="sqrt"
- LGBM: lr 0.05, num_leaves 31, early stopping 50, subsample/colsample 0.8
- SVM: C=1, gamma="scale"; kNN: k=5–15, distance weights

#### Complexity (train)
- OLS $O(np^2)$; logistic $O(np)$/iter
- Tree $O(pn\log n)$; RF ×B (parallel)
- Kernel SVM $O(n^2..n^3)$; kNN predict $O(np)$
- k-means $O(nkp)$/iter

#### Failure modes
- Leakage in preprocessing → Pipeline
- High-card target encoding → out-of-fold
- Imbalanced → PR-AUC + threshold tuning
- RF impurity importance → use permutation
- t-SNE distances → don't interpret

#### Interview one-liners
- Bias–variance: simple underfits, complex overfits; regularise and validate
- L1 sparse (corners), L2 shrinks (round)
- Bagging ↓ variance, boosting ↓ bias

</div>
</div>
