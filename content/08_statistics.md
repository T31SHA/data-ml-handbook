# Statistics, Experimentation & Causal Inference

tagline: From samples to decisions: estimation, intervals, tests, power, A/B testing, the pitfalls that fake discoveries, and the causal toolkit for when correlation is not enough.

## Descriptive statistics that survive contact with data

| Measure | Formula | Robust? | Use when |
|---|---|---|---|
| Mean | $\bar x = \frac1n\sum x_i$ | no (one outlier moves it) | symmetric data, totals matter |
| Median | middle order statistic | yes (50% breakdown) | skewed data: income, latency |
| Trimmed mean | mean of central $1-2\alpha$ | partly | outliers present but mean-like summary needed |
| Std. deviation | $s = \sqrt{\frac{1}{n-1}\sum(x_i-\bar x)^2}$ | no | Gaussian-ish spread |
| IQR | $Q_3 - Q_1$ | yes | boxplots, outlier fences $Q_1 - 1.5\,\text{IQR}$ |
| MAD | $\text{median}(\lvert x_i - \tilde x\rvert)$ | yes | robust z-score: $0.6745(x-\tilde x)/\text{MAD}$ |
| Coef. of variation | $s/\bar x$ | no | compare spread across scales |
| Skewness | $\mathbb{E}[(X-\mu)^3]/\sigma^3$ | no | > 1: consider log transform |
| Pearson $r$ | $\frac{\text{cov}(x,y)}{s_x s_y}$ | no | linear association |
| Spearman $\rho$ | Pearson on ranks | yes | monotone association |

!!! intuition "Why n − 1 (Bessel's correction)"
    Deviations are measured from $\bar x$, which was fitted to the same data, so they are systematically too small. One degree of freedom is "used up" by estimating the mean: $\mathbb{E}\big[\sum(x_i-\bar x)^2\big] = (n-1)\sigma^2$. Dividing by $n-1$ makes $s^2$ unbiased.

![](figures/correlation.svg)

!!! pitfall "Pitfall: correlation summaries hide shape"
    Anscombe's quartet and the "datasaurus" have identical means, variances and $r$. $r = 0$ does not mean independence (a parabola has $r \approx 0$). Always plot. Use mutual information or distance correlation for non-linear dependence.

## Sampling distributions, the CLT and the standard error

![](figures/clt.svg)

$$\bar X \;\approx\; \mathcal{N}\!\left(\mu,\ \frac{\sigma^2}{n}\right) \qquad \text{SE}(\bar x) = \frac{s}{\sqrt n}$$

**Intuition:** averaging cancels independent noise; spread shrinks like $1/\sqrt n$, so 4× the data halves the error. **Example:** latency sample $s = 80$ ms, $n = 400$: $\text{SE} = 80/20 = 4$ ms, 95% CI $\bar x \pm 7.8$ ms.

!!! pitfall "Pitfall: when the CLT is slow"
    Heavy tails (revenue per user, Pareto-like) need far more than "n = 30". Dependence (time series, users with many sessions) shrinks the effective $n$; use cluster-robust or block-bootstrap SEs. Ratios of means (CTR = clicks / views per user) need the delta method.

### Confidence intervals

| Quantity | 95% interval | Notes |
|---|---|---|
| Mean | $\bar x \pm t_{0.975,\,n-1}\,s/\sqrt n$ | $t \to 1.96$ for large $n$ |
| Proportion | Wilson: $\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + z^2/n}$ | Wald $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$ fails near 0/1 |
| Difference of means | $(\bar x_1 - \bar x_2) \pm t\sqrt{s_1^2/n_1 + s_2^2/n_2}$ | Welch |
| Any statistic | bootstrap percentile $[q_{2.5}, q_{97.5}]$ | medians, AUC, ratios |
| Ratio $\bar y/\bar x$ | delta method: $\text{Var} \approx \frac{1}{\bar x^2}\big(\text{Var}\,\bar y - 2R\,\text{Cov} + R^2\,\text{Var}\,\bar x\big)$ | CTR, revenue per session |

!!! rule "Rule of thumb: what a 95% CI means"
    If you repeated the whole experiment many times, 95% of the intervals built this way would contain the true value. It is **not** "95% probability the parameter is inside this particular interval" (that is a Bayesian credible interval).

![](figures/bootstrap.svg)

## Maximum likelihood estimation

$$\hat\theta_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^n \log p(x_i\mid\theta)$$

![](figures/mle.svg)

!!! derivation "Derivation: Gaussian MLE"
    $\ell(\mu,\sigma) = -\frac n2\log(2\pi) - n\log\sigma - \frac{1}{2\sigma^2}\sum(x_i-\mu)^2$. Setting $\partial\ell/\partial\mu = \frac{1}{\sigma^2}\sum(x_i - \mu) = 0$ gives $\hat\mu = \bar x$. Setting $\partial\ell/\partial\sigma = -\frac n\sigma + \frac{1}{\sigma^3}\sum(x_i-\mu)^2 = 0$ gives $\hat\sigma^2 = \frac1n\sum(x_i-\bar x)^2$, which is biased (divides by $n$). Minimising squared error is MLE under Gaussian noise; minimising absolute error is MLE under Laplace noise; log-loss is MLE for Bernoulli labels.

| Property | Meaning |
|---|---|
| Consistent | $\hat\theta \to \theta$ as $n\to\infty$ |
| Asymptotically normal | $\hat\theta \approx \mathcal{N}(\theta, I(\theta)^{-1}/n)$, Fisher information $I = -\mathbb{E}[\partial^2_\theta \log p]$ |
| Invariant | MLE of $g(\theta)$ is $g(\hat\theta)$ |
| Can overfit | small $n$, many parameters: add a prior (MAP) = regularisation |

@@code code/s06_mle_bh.py@@
@@out code/out/s06_mle_bh.txt@@

## Hypothesis testing

![](figures/hypothesis.svg)

1. State $H_0$ (no effect) and $H_1$; choose $\alpha$ (usually 0.05) **before** looking.
2. Compute a test statistic, e.g. $t = \frac{\bar x_1 - \bar x_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}$.
3. p-value $= P(\text{statistic at least this extreme}\mid H_0)$. Reject if $p < \alpha$.
4. Report the **effect size and its CI**, not just $p$.

| | $H_0$ true | $H_0$ false |
|---|---|---|
| Reject | Type I error, rate $\alpha$ | correct: power $1-\beta$ |
| Don't reject | correct | Type II error, rate $\beta$ |

!!! example "Worked example: a two-sample t-test"
    $\bar x_A = 100,\ s_A = 15,\ n_A = 40$; $\bar x_B = 108,\ s_B = 25,\ n_B = 45$. $\text{SE} = \sqrt{225/40 + 625/45} = \sqrt{5.63 + 13.89} = 4.42$. $t = 8/4.42 = 1.81$; with ~73 df, $p \approx 0.07$: not significant at 5% even though the effect (8 points, $d \approx 0.39$) may matter. Conclusion: underpowered, not "no effect".

### Which test?

| Question | Parametric | Non-parametric / robust |
|---|---|---|
| One mean vs value | one-sample t | Wilcoxon signed-rank |
| Two independent groups | Welch t-test | Mann–Whitney U, permutation test |
| Paired / before-after | paired t | Wilcoxon signed-rank |
| 3+ groups | one-way ANOVA (+ Tukey HSD) | Kruskal–Wallis (+ Dunn) |
| Two categorical vars | χ² test of independence | Fisher exact (small counts) |
| Two proportions | two-proportion z-test | Fisher exact, bootstrap |
| Correlation | Pearson r test | Spearman / Kendall |
| Normality | Shapiro–Wilk ($n < 5000$) | Q-Q plot (better) |
| Equal variances | Bartlett | Levene / Brown–Forsythe |
| Distribution shift | | Kolmogorov–Smirnov, PSI |
| Model A vs B (same folds) | corrected resampled t | McNemar (classifier errors) |

@@code code/s06_tests.py@@
@@out code/out/s06_tests.txt@@

!!! interview "Interview: what is a p-value?"
    The probability, assuming the null is true, of seeing data at least as extreme as observed. It is **not** the probability the null is true, not the probability the result is a fluke, and not a measure of effect size. A tiny effect with huge $n$ gives tiny $p$.

## Power and sample size

$$n_{\text{per arm}} = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2\cdot 2\sigma^2}{\delta^2}\;\approx\;\frac{16\,\sigma^2}{\delta^2}\quad(\alpha = 0.05,\ 80\%\text{ power})$$

**Intuition:** detect an effect $\delta$ against noise $\sigma$; halving the effect needs 4× the sample. **Example:** baseline conversion 10% ($\sigma^2 = 0.09$), MDE 1 point: $n \approx 16\cdot0.09/0.0001 = 14{,}400$ per arm.

![](figures/power.svg)

## A/B testing done properly

@@code code/s06_ab.py@@
@@out code/out/s06_ab.txt@@

| Step | What to do | Failure if skipped |
|---|---|---|
| 1. Hypothesis & metric | one primary metric, guardrails (latency, revenue, errors) | metric shopping later |
| 2. Unit & randomisation | randomise by user (hash of id + salt), not by request | contamination, dependent samples |
| 3. Power | MDE, $\alpha$, power → $n$ and duration (≥ 1–2 full weeks) | underpowered "no result" |
| 4. A/A test & SRM | check sample-ratio mismatch with χ² | broken bucketing invalidates everything |
| 5. Run without peeking | or use sequential tests (mSPRT, alpha spending) | false-positive rate balloons to 20–30% |
| 6. Analyse | effect + CI, CUPED variance reduction, segments pre-registered | |
| 7. Decide | practical significance vs cost; ramp-up | shipping noise |

!!! derivation "Derivation: CUPED"
    Use a pre-experiment covariate $X$ (e.g. last month's spend). Define $Y_{\text{cv}} = Y - \theta(X - \bar X)$. Then $\text{Var}(Y_{\text{cv}}) = \text{Var}(Y) - 2\theta\,\text{Cov}(X,Y) + \theta^2\text{Var}(X)$, minimised at $\theta = \text{Cov}(X,Y)/\text{Var}(X)$, giving $\text{Var}(Y)(1-\rho^2)$. With $\rho = 0.85$ the variance falls by 72%, which is like getting 3.6× more users. The mean is unbiased because $X$ was measured before treatment.

!!! pitfall "Pitfalls specific to online experiments"
    Novelty and primacy effects (run long enough); network effects and interference (cluster or switchback randomisation); Simpson's paradox when the traffic mix changes mid-test; ratio metrics analysed per-session instead of per-user; Twyman's law: any figure that looks interesting or different is usually wrong.

## Statistics pitfalls that manufacture discoveries

### P-hacking and the garden of forking paths

![](figures/phacking.svg)

Trying many analyses (outlier rules, subgroups, covariates, stopping times) and reporting the one that "works" inflates false positives far beyond $\alpha$. With 20 independent looks, $P(\geq 1 \text{ false positive}) = 1 - 0.95^{20} = 64\%$.

!!! rule "Rule of thumb: defenses"
    Pre-register the hypothesis, metric and analysis; hold out a confirmation set; report every test run; prefer effect sizes and CIs; replicate. In ML: the test set is touched once, and every model selection decision is made on validation data.

### Multiple comparisons

| Method | Controls | Rule | When |
|---|---|---|---|
| Bonferroni | FWER | reject if $p_i < \alpha/m$ | few tests, any false positive is costly |
| Holm | FWER | step-down: sort $p$; reject while $p_{(k)} < \alpha/(m-k+1)$ | always better than Bonferroni |
| Benjamini–Hochberg | FDR | largest $k$ with $p_{(k)} \leq k\alpha/m$; reject $1..k$ | many tests: genomics, feature screening, many metrics |

The executed snippet in the MLE section shows it: with 90 nulls and 10 real effects, uncorrected testing made 4 false discoveries; BH found all 10 real effects with 0 false ones.

### Simpson's paradox

![](figures/simpson.svg)

A trend in every subgroup can reverse when groups are pooled because a confounder (group membership) is related to both $X$ and $Y$. **Classic:** a treatment that works better for both mild and severe patients looks worse overall because it was given mostly to severe cases. **Resolution:** the causal graph decides. Stratify when the grouping variable is a confounder; do *not* stratify when it is a mediator.

### More traps

| Trap | What happens | Fix |
|---|---|---|
| Regression to the mean | extreme units look "improved" on re-measure | control group |
| Survivorship bias | you only see winners (funds, startups, planes that returned) | include failures / dropouts |
| Selection / collider bias | conditioning on a common effect creates fake correlation | draw the DAG; don't condition on colliders |
| Base-rate neglect | 99%-accurate test, 1% prevalence → mostly false positives | Bayes, report precision |
| Ecological fallacy | group-level correlation applied to individuals | individual-level data |
| Berkson's paradox | hospital data: two diseases look negatively correlated | sample from population |
| Optional stopping | peeking until $p < 0.05$ | sequential designs |
| Garden of forking paths | flexible analysis | pre-registration |
| Look-elsewhere | many segments, one "significant" | FDR control, holdout |

## Regression for inference

$$y = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p + \varepsilon,\qquad \hat{\boldsymbol\beta} = (X^\top X)^{-1}X^\top \mathbf{y},\qquad \text{SE}(\hat\beta_j) = \hat\sigma\sqrt{[(X^\top X)^{-1}]_{jj}}$$

| Assumption | Check | If violated |
|---|---|---|
| Linearity | residual vs fitted plot | transforms, splines, GAMs |
| Independence | domain, Durbin–Watson | clustered SE, mixed models |
| Homoscedasticity | Breusch–Pagan, fan-shaped residuals | robust (HC3) SE, log $y$, WLS |
| Normal errors | Q-Q plot | matters only for small-n inference |
| No perfect multicollinearity | VIF > 5–10 | drop / combine, ridge |
| Exogeneity $\mathbb{E}[\varepsilon\mid X]=0$ | domain knowledge | omitted variable bias → causal methods |

**Coefficient reading:** $\beta_j$ is the expected change in $y$ for a one-unit change in $x_j$ *holding the others fixed*. Log–level: $100\beta\%$ change in $y$; level–log: $\beta/100$ per 1% change in $x$; log–log: elasticity. GLMs: logistic ($e^{\beta}$ = odds ratio), Poisson (rate ratio).

## Bayesian vs frequentist in one table

| | Frequentist | Bayesian |
|---|---|---|
| Parameter | fixed unknown | random variable with a prior |
| Output | point estimate, CI, p-value | posterior distribution, credible interval |
| A/B answer | "reject $H_0$ at 5%" | "$P(B > A) = 97\%$, expected loss 0.02%" |
| Peeking | inflates error | posterior is valid any time (but decisions still need care) |
| Small data | weak | priors help (shrinkage) |
| Tools | scipy, statsmodels | PyMC, Stan, NumPyro, conjugate formulas |

## Causal inference basics

Prediction asks "what will $Y$ be?"; causal inference asks "what would $Y$ be **if we changed** $X$?" Potential outcomes: each unit has $Y(1)$ and $Y(0)$; we observe only one. Average treatment effect $\text{ATE} = \mathbb{E}[Y(1) - Y(0)]$.

![](figures/causal_dags.svg)

| Structure | Graph | Should you adjust? |
|---|---|---|
| Confounder | $X \leftarrow Z \rightarrow Y$ | **yes**, else bias (Simpson) |
| Mediator | $X \rightarrow M \rightarrow Y$ | no (for the total effect) |
| Collider | $X \rightarrow C \leftarrow Y$ | **never**: creates spurious association |
| Instrument | $Z \rightarrow X \rightarrow Y$ | use for IV estimation, not as a control |

### The toolbox

| Method | Idea | Key assumption | Example |
|---|---|---|---|
| Randomised experiment | randomisation breaks all confounding | compliance, no interference | A/B test |
| Regression adjustment | control for confounders | all confounders measured | observational analysis |
| Propensity score (matching, IPW) | reweight to balance $P(T=1\mid X)$ | no unmeasured confounding, overlap | marketing uplift |
| Difference-in-differences | $(Y^T_{\text{post}} - Y^T_{\text{pre}}) - (Y^C_{\text{post}} - Y^C_{\text{pre}})$ | parallel trends | policy rollout in one region |
| Instrumental variables | variation in $X$ caused only by $Z$ | relevance, exclusion | distance to college → education |
| Regression discontinuity | compare just above / below a cutoff | no manipulation at cutoff | credit score threshold |
| Synthetic control | weighted combination of control units | stable pre-period fit | one country / city treated |
| Double ML / causal forests | ML for nuisance functions, orthogonalised estimate | unconfoundedness | heterogeneous effects (EconML) |
| Uplift modelling | model $\mathbb{E}[Y\mid X,T=1] - \mathbb{E}[Y\mid X,T=0]$ | randomised training data | who to send a coupon |

!!! example "Worked example: difference-in-differences"
    City A gets a new feature; city B does not. Weekly orders before/after: A 100 → 130, B 90 → 105. Naive effect = 30; DiD $= (130-100) - (105-90) = 30 - 15 = 15$ orders/week. Half of A's growth was the seasonal trend shared with B.

!!! derivation "Derivation: inverse propensity weighting"
    With propensity $e(x) = P(T=1\mid x)$ and unconfoundedness, $\mathbb{E}\left[\frac{T\,Y}{e(X)}\right] = \mathbb{E}\left[\frac{\mathbb{E}[T\mid X]\,\mathbb{E}[Y(1)\mid X]}{e(X)}\right] = \mathbb{E}[Y(1)]$. So $\widehat{\text{ATE}} = \frac1n\sum\left[\frac{T_iY_i}{\hat e_i} - \frac{(1-T_i)Y_i}{1-\hat e_i}\right]$. Clip $\hat e$ to $[0.01, 0.99]$: extreme weights explode the variance.

!!! interview "Interview: your model says discount users churn less. Should we give everyone a discount?"
    No: that is a correlation. Discounts may target loyal users (confounding) or churners may never see the offer. Run a randomised test, or use uplift modelling / propensity methods with explicit assumptions.

<div class="card" markdown="1">

## Cheat card: Statistics & causal inference

<div class="cols" markdown="1">

#### Estimation
- $\text{SE} = s/\sqrt n$; 95% CI ≈ estimate ± 2 SE
- Proportions: Wilson interval
- Anything else: bootstrap (5–10k resamples)
- Ratios: delta method; clustered data: cluster SE

#### Testing
- Welch t by default (never assume equal variance)
- Skewed / ordinal → Mann–Whitney or permutation
- Categorical → χ² (Fisher if expected < 5)
- 3+ groups → ANOVA + Tukey; Kruskal–Wallis
- Report effect size: Cohen's $d$ (0.2 / 0.5 / 0.8), lift with CI

#### Power
- $n \approx 16\sigma^2/\delta^2$ per arm (α 0.05, power 0.8)
- Proportions: $\sigma^2 = p(1-p)$
- Halve MDE → 4× sample

#### A/B checklist
- Randomise by user; hash(id + salt)
- SRM χ² check, A/A test
- Full weeks; no peeking (or sequential)
- CUPED: $-$(1 − ρ²) variance
- Guardrail metrics; pre-registered segments

#### Multiple testing
- $1-(1-\alpha)^m$ false-positive chance
- Holm for FWER, BH for FDR

#### Causal
- Confounder: adjust. Mediator: don't (total effect). Collider: never.
- No experiment? DiD, IV, RDD, synthetic control, propensity
- Ask "what is the counterfactual?"

#### Pitfalls
- p-hacking, Simpson, survivorship, regression to mean, base rates, Berkson

</div>
</div>
