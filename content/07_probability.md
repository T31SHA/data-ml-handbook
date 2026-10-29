# Probability, Information Theory & Discrete Maths

tagline: The language of uncertainty: rules, Bayes, random variables, distributions, Markov chains, entropy, plus the combinatorics and graphs that underpin algorithms.

## Rules of probability

| Rule | Formula | Intuition |
|---|---|---|
| Complement | $P(A^c) = 1 - P(A)$ | "at least one" $= 1 - P(\text{none})$ |
| Union | $P(A\cup B) = P(A) + P(B) - P(A\cap B)$ | don't double count |
| Conditional | $P(A\mid B) = \dfrac{P(A\cap B)}{P(B)}$ | shrink the universe to $B$ |
| Product | $P(A\cap B) = P(A\mid B)P(B)$ | chain rule: $P(A,B,C) = P(A)P(B\mid A)P(C\mid A,B)$ |
| Independence | $P(A\cap B) = P(A)P(B)$ | knowing $B$ tells nothing about $A$ |
| Total probability | $P(A) = \sum_i P(A\mid B_i)P(B_i)$ | average over exhaustive cases |
| **Bayes** | $P(H\mid D) = \dfrac{P(D\mid H)\,P(H)}{P(D)}$ | posterior ∝ likelihood × prior |

!!! example "Worked example: the base-rate fallacy"
    Prevalence 1%, sensitivity 99%, false-positive rate 5%. $P(\text{sick}\mid +) = \frac{0.99\cdot0.01}{0.99\cdot0.01 + 0.05\cdot0.99} = \frac{0.0099}{0.0594} \approx 16.7\%$. Most positives are false alarms because healthy people vastly outnumber sick ones. The same arithmetic explains low precision in fraud and anomaly detection.

@@code code/s05_bayes_markov.py@@
@@out code/out/s05_bayes_markov.txt@@

## Bayesian updating

$$\underbrace{p(\theta\mid D)}_{\text{posterior}} = \frac{\overbrace{p(D\mid\theta)}^{\text{likelihood}}\ \overbrace{p(\theta)}^{\text{prior}}}{\underbrace{\int p(D\mid\theta')p(\theta')\,d\theta'}_{\text{evidence}}}$$

![](figures/bayes_update.svg)

!!! derivation "Derivation: Beta–Binomial conjugacy"
    Prior $\theta \sim \text{Beta}(\alpha,\beta) \propto \theta^{\alpha-1}(1-\theta)^{\beta-1}$. Likelihood of $k$ successes in $n$ trials $\propto \theta^k(1-\theta)^{n-k}$. Posterior $\propto \theta^{\alpha+k-1}(1-\theta)^{\beta+n-k-1}$, i.e. $\text{Beta}(\alpha + k,\ \beta + n - k)$. The prior acts like $\alpha + \beta$ pseudo-observations. Posterior mean $\frac{\alpha+k}{\alpha+\beta+n}$ shrinks the raw rate $k/n$ toward the prior mean, which is exactly smoothed target encoding and Bayesian A/B testing.

| Prior | Likelihood | Posterior | Use |
|---|---|---|---|
| Beta | Binomial | Beta | conversion rates, CTR |
| Gamma | Poisson | Gamma | event rates |
| Normal (known σ) | Normal | Normal | measurement fusion, Kalman filter |
| Dirichlet | Multinomial | Dirichlet | topic models, category probabilities |

**MAP vs MLE:** $\hat\theta_{\text{MAP}} = \arg\max[\log p(D\mid\theta) + \log p(\theta)]$. A Gaussian prior gives L2 (ridge) regularisation, a Laplace prior gives L1 (lasso). When conjugacy fails, use MCMC (PyMC, Stan, NumPyro) or variational inference.

## Random variables, expectation and variance

| Quantity | Definition | Key properties |
|---|---|---|
| Expectation | $\mathbb{E}[X] = \sum_x x\,p(x)$ or $\int x\,f(x)\,dx$ | linear always: $\mathbb{E}[aX + bY] = a\mathbb{E}X + b\mathbb{E}Y$ |
| LOTUS | $\mathbb{E}[g(X)] = \int g(x)f(x)\,dx$ | no need for the distribution of $g(X)$ |
| Variance | $\operatorname{Var}X = \mathbb{E}[(X-\mu)^2] = \mathbb{E}X^2 - (\mathbb{E}X)^2$ | $\operatorname{Var}(aX + b) = a^2\operatorname{Var}X$ |
| Covariance | $\operatorname{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}X\,\mathbb{E}Y$ | $\operatorname{Var}(X+Y) = \operatorname{Var}X + \operatorname{Var}Y + 2\operatorname{Cov}$ |
| Correlation | $\rho = \operatorname{Cov}(X,Y)/(\sigma_X\sigma_Y) \in [-1,1]$ | independence ⇒ $\rho = 0$, not conversely |
| Tower rule | $\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X\mid Y]]$ | average of group averages |
| Total variance | $\operatorname{Var}X = \mathbb{E}[\operatorname{Var}(X\mid Y)] + \operatorname{Var}(\mathbb{E}[X\mid Y])$ | within + between (ANOVA) |
| Mean of $n$ iid | $\operatorname{Var}(\bar X) = \sigma^2/n$ | standard error $\sigma/\sqrt n$ |

!!! example "Worked example: a fair die"
    $\mathbb{E}X = 3.5$; $\mathbb{E}X^2 = (1+4+9+16+25+36)/6 = 91/6$; $\operatorname{Var}X = 91/6 - 12.25 \approx 2.917$. The sum of two dice has mean 7 and variance $5.83$ (variances add for independent variables).

**Inequalities:** Markov $P(X\ge a) \le \mathbb{E}X/a$; Chebyshev $P(\lvert X-\mu\rvert \ge k\sigma) \le 1/k^2$; Jensen $f(\mathbb{E}X) \le \mathbb{E}f(X)$ for convex $f$ (why the ELBO is a lower bound, why $\log$ of an average ≥ average of logs); Hoeffding $P(\lvert\bar X - \mu\rvert \ge t) \le 2e^{-2nt^2}$ for bounded variables (generalisation bounds).

## Distributions

![](figures/distributions.svg)

| Distribution | Models | pmf / pdf | Mean | Variance |
|---|---|---|---|---|
| Bernoulli($p$) | one yes/no | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ |
| Binomial($n,p$) | successes in $n$ trials | $\binom nk p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ |
| Geometric($p$) | trials to first success | $(1-p)^{k-1}p$ | $1/p$ | $(1-p)/p^2$ |
| Poisson($\lambda$) | counts per interval | $\lambda^ke^{-\lambda}/k!$ | $\lambda$ | $\lambda$ |
| Neg. binomial | over-dispersed counts | — | $\mu$ | $\mu + \mu^2/r$ |
| Uniform($a,b$) | equally likely | $1/(b-a)$ | $(a+b)/2$ | $(b-a)^2/12$ |
| Normal($\mu,\sigma^2$) | sums of many effects | $\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ |
| Log-normal | multiplicative effects (prices) | $\ln X \sim \mathcal N$ | $e^{\mu+\sigma^2/2}$ | — |
| Exponential($\lambda$) | waiting times, memoryless | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ |
| Gamma($k,\theta$) | sum of $k$ exponentials | — | $k\theta$ | $k\theta^2$ |
| Beta($\alpha,\beta$) | probabilities | $\propto x^{\alpha-1}(1-x)^{\beta-1}$ | $\frac{\alpha}{\alpha+\beta}$ | — |
| Student $t_\nu$ | small-sample means, robust errors | heavy tails | 0 | $\frac{\nu}{\nu-2}$ |
| $\chi^2_k$ | sum of $k$ squared standard normals | — | $k$ | $2k$ |
| Pareto / power law | wealth, popularity (80/20) | $\propto x^{-(\alpha+1)}$ | — | may be infinite |

![](figures/normal.svg)

**Multivariate normal** $\mathcal N(\boldsymbol\mu, \Sigma)$: density $\propto \exp\!\big(-\tfrac12(\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)\big)$. Contours are ellipses along the eigenvectors of $\Sigma$; the **Mahalanobis distance** $\sqrt{(\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)}$ is the natural outlier score. Marginals and conditionals of a Gaussian are Gaussian (the basis of Gaussian processes and Kalman filters).

```python
from scipy import stats
stats.norm.cdf(1.96), stats.norm.ppf(0.975)        # 0.975, 1.96   (CDF and its inverse, the quantile)
stats.binom(10, .3).pmf(3); stats.poisson(3).sf(5) # sf = 1 - cdf (survival), more precise in the tail
```

## Markov chains and stochastic processes

![](figures/markov.svg)

- **Markov property:** $P(X_{t+1}\mid X_t, X_{t-1},\dots) = P(X_{t+1}\mid X_t)$. Transition matrix $P$ with rows summing to 1; distribution after $n$ steps $\boldsymbol\pi_n = \boldsymbol\pi_0P^n$.
- **Stationary distribution:** $\boldsymbol\pi = \boldsymbol\pi P$, the left eigenvector for eigenvalue 1; unique and reached from any start if the chain is irreducible and aperiodic (verified in the listing above). PageRank is the stationary distribution of a random surfer with teleportation.
- **Absorbing chains:** expected steps to absorption via the fundamental matrix $N = (I - Q)^{-1}$ (customer journeys, churn).
- **Hidden Markov models:** forward algorithm (likelihood), Viterbi (most likely state path), Baum–Welch (EM).
- **MCMC:** construct a chain whose stationary distribution is the posterior (Metropolis–Hastings, Gibbs, HMC/NUTS).
- **Poisson process:** events at rate $\lambda$, exponential gaps; **random walk / Brownian motion:** finance, diffusion models.

## Information theory

| Quantity | Formula | Intuition | ML use |
|---|---|---|---|
| Entropy | $H(p) = -\sum_x p(x)\log p(x)$ | average surprise; bits to encode | decision-tree splits, max-entropy priors |
| Cross-entropy | $H(p,q) = -\sum_x p(x)\log q(x)$ | bits when coding $p$ with model $q$ | classification loss, LM perplexity $= e^{H}$ |
| KL divergence | $D_{KL}(p\Vert q) = \sum p\log\frac pq = H(p,q) - H(p) \ge 0$ | extra bits for using the wrong model; asymmetric | VAEs, RLHF penalty, distillation, drift |
| Mutual information | $I(X;Y) = H(X) - H(X\mid Y)$ | information shared | feature selection, InfoNCE |
| Jensen–Shannon | symmetrised, bounded KL | distance between distributions | drift, GAN theory |

!!! example "Worked example: entropy of a split"
    A node with 8 positives and 8 negatives has $H = 1$ bit. A split into (7+, 1−) and (1+, 7−): each child has $H = -\frac78\log_2\frac78 - \frac18\log_2\frac18 \approx 0.544$. Information gain $= 1 - 0.544 = 0.456$ bits.

## Discrete mathematics essentials

| Topic | Key facts | Where it bites |
|---|---|---|
| Sets | $\lvert A\cup B\rvert = \lvert A\rvert + \lvert B\rvert - \lvert A\cap B\rvert$; Jaccard $\frac{\lvert A\cap B\rvert}{\lvert A\cup B\rvert}$ | SQL joins, deduplication (MinHash) |
| Logic | De Morgan: $\neg(A\land B) = \neg A\lor\neg B$; contrapositive ≡ implication | filters, three-valued SQL logic with NULL |
| Counting | permutations $\frac{n!}{(n-k)!}$; combinations $\binom nk$; with repetition $\binom{n+k-1}{k}$; pigeonhole | hyper-parameter grids, A/B segments, hash collisions |
| Birthday bound | collision likely after $\approx\sqrt{N}$ draws ($\sqrt{2N\ln2}$ for 50%) | ID/hash sizing, sampling |
| Graphs | adjacency matrix/list; BFS (shortest unweighted), DFS, Dijkstra $O((V+E)\log V)$, topological sort for DAGs | Airflow/dbt DAGs, lineage, knowledge graphs, GNNs |
| Trees & heaps | $n$ nodes ⇒ $n-1$ edges; balanced BST/B-tree $O(\log n)$; heap top-$k$ $O(n\log k)$ | database indexes, top-k queries |
| Big-O | $1 < \log n < n < n\log n < n^2 < 2^n < n!$ | algorithm and query design |
| Recurrences | $T(n) = 2T(n/2) + n \Rightarrow O(n\log n)$ (master theorem) | divide-and-conquer, merge sort |
| Modular arithmetic | `hash(key) % n` sharding; consistent hashing moves $\approx 1/n$ keys on resize | partitioning, Kafka keys |
| Induction | base case + ($k \Rightarrow k+1$) | proving loop invariants, recursive correctness |

<div class="card" markdown="1">

## Cheat card: Probability

<div class="cols" markdown="1">

#### Rules
- $P(A\mid B) = P(A\cap B)/P(B)$
- Bayes: posterior ∝ likelihood × prior
- total probability: average over cases
- independence: $P(A\cap B) = P(A)P(B)$

#### Moments
- $\mathbb{E}$ is linear, always
- $\operatorname{Var}X = \mathbb{E}X^2 - (\mathbb{E}X)^2$
- $\operatorname{Var}(\bar X) = \sigma^2/n$
- $\rho = 0$ does not imply independence

#### Pick a distribution
- yes/no → Bernoulli/Binomial
- counts → Poisson (over-dispersed: NB)
- waiting time → Exponential/Gamma
- rates/probabilities → Beta
- sums → Normal; products → log-normal

#### Bayes in practice
- Beta(α,β) + k/n → Beta(α+k, β+n−k)
- MAP with Gaussian prior = ridge
- rare events: mind the base rate

#### Information
- entropy = expected surprise
- CE loss = $H(p) + D_{KL}(p\Vert q)$
- perplexity = $e^{\text{CE}}$

#### Markov
- $\boldsymbol\pi_n = \boldsymbol\pi_0P^n$
- stationary: $\boldsymbol\pi = \boldsymbol\pi P$
- MCMC samples posteriors

</div>
</div>
