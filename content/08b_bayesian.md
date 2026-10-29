# Bayesian Inference in Practice

tagline: Priors, posteriors, conjugacy, MCMC and the diagnostics that tell you whether to trust a chain.

## Bayes' theorem as an update rule

$$p(\theta \mid y) = \frac{p(y \mid \theta)\,p(\theta)}{p(y)} \;\propto\; p(y \mid \theta)\,p(\theta)$$

The posterior is the prior reweighted by the likelihood. The evidence $p(y)=\int p(y\mid\theta)p(\theta)\,d\theta$ is a normalising constant for inference about $\theta$, but it is the quantity that matters for model comparison (Bayes factors).

!!! intuition "Intuition"
    A prior is a set of pseudo-observations. With a Beta$(a,b)$ prior on a rate you behave as if you had already seen $a-1$ successes and $b-1$ failures. As data accumulate the likelihood dominates and reasonable priors converge to the same answer.

![](figures/v2_bayes_update.svg)

## Conjugate families

| Likelihood | Conjugate prior | Posterior | Typical use |
|---|---|---|---|
| Bernoulli / Binomial | Beta$(a,b)$ | Beta$(a+k,\,b+n-k)$ | conversion rates |
| Poisson | Gamma$(\alpha,\beta)$ | Gamma$(\alpha+\sum y,\,\beta+n)$ | counts, event rates |
| Normal, known $\sigma^2$ | Normal$(\mu_0,\tau_0^2)$ | Normal, precision-weighted mean | measurement averaging |
| Multinomial | Dirichlet$(\alpha)$ | Dirichlet$(\alpha+c)$ | category shares, topic models |

!!! derivation "Derivation: Normal mean with known variance"
    With prior $\mu\sim\mathcal N(\mu_0,\tau_0^2)$ and $n$ observations with mean $\bar y$, the posterior precision is the sum of precisions:
    $$\frac{1}{\tau_n^2}=\frac{1}{\tau_0^2}+\frac{n}{\sigma^2},\qquad \mu_n=\tau_n^2\left(\frac{\mu_0}{\tau_0^2}+\frac{n\bar y}{\sigma^2}\right).$$
    The posterior mean is a precision-weighted average of the prior mean and the sample mean; this is *shrinkage*.

## Markov chain Monte Carlo

When the posterior has no closed form, draw from it. Random-walk Metropolis proposes $\theta'\sim q(\cdot\mid\theta)$ and accepts with probability

$$\alpha=\min\left(1,\;\frac{p(y\mid\theta')\,p(\theta')}{p(y\mid\theta)\,p(\theta)}\right),$$

which needs only the *unnormalised* posterior. The chain's stationary distribution is the posterior (detailed balance). Hamiltonian Monte Carlo (NUTS, used by Stan and PyMC) uses gradients to make long, low-rejection moves and is the default for continuous parameters.

@@code code/n01_bayes_mcmc.py@@
@@out code/out/n01_bayes_mcmc.txt@@

!!! pitfall "Pitfall: sampling a bounded parameter on its raw scale"
    Sampling $p\in(0,1)$ directly wastes proposals outside the support. Transform to an unconstrained scale (logit, log) and add the log-Jacobian, as in the listing above. Forgetting the Jacobian silently changes the prior.

## Diagnostics you must report

| Diagnostic | What it checks | Healthy value |
|---|---|---|
| $\hat R$ (split, rank-normalised) | between- vs within-chain variance across $\ge 4$ chains | $< 1.01$ |
| Bulk / tail ESS | effective independent draws | $> 400$ per quantity reported |
| Divergent transitions (HMC) | geometry the sampler cannot follow | 0; reparameterise if not |
| Posterior predictive check | does simulated data resemble observed data? | no systematic misfit |

## Hierarchical models and partial pooling

For groups $j$ (stores, schools, countries) with few observations each:

$$y_{ij}\sim\mathcal N(\theta_j,\sigma^2),\quad \theta_j\sim\mathcal N(\mu,\tau^2).$$

Each group estimate is pulled toward $\mu$ in proportion to how noisy it is: $\hat\theta_j \approx w_j\bar y_j+(1-w_j)\mu$ with $w_j=\tau^2/(\tau^2+\sigma^2/n_j)$. This beats both "no pooling" (overfits small groups) and "complete pooling" (ignores real differences).

!!! tip "Pro tip"
    If a hierarchical model produces divergences, switch to the non-centred parameterisation $\theta_j=\mu+\tau z_j$, $z_j\sim\mathcal N(0,1)$. It removes the funnel between $\tau$ and $\theta_j$.

!!! interview "Interview question"
    *What is the difference between a credible interval and a confidence interval?* A 95% credible interval contains the parameter with posterior probability 0.95 given the data and prior. A 95% confidence interval is produced by a procedure that covers the fixed true parameter in 95% of repeated samples; any single interval either covers it or not.

## Cheat card: Bayesian inference

- Posterior $\propto$ likelihood $\times$ prior; the evidence matters only for model comparison.
- Use conjugate updates for rates and counts; they are exact and instant.
- Use NUTS for everything else; run 4 chains, check $\hat R<1.01$, ESS $>400$, zero divergences.
- Transform bounded parameters and include the Jacobian.
- Partial pooling via hierarchical priors is the default for grouped data.
- Always do a posterior predictive check before interpreting parameters.
