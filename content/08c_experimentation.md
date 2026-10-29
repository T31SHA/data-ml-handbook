# Experimentation and A/B Testing at Scale

tagline: Design, variance reduction, sequential monitoring and the guardrails that keep online experiments honest.

## Anatomy of a trustworthy experiment

1. **Hypothesis and decision rule** written before launch: primary metric, minimum detectable effect (MDE), $\alpha$, power, duration.
2. **Randomisation unit** equals the analysis unit (user, session, cluster). If they differ, use cluster-robust or delta-method variances.
3. **Guardrail metrics** (latency, errors, revenue) with their own thresholds.
4. **Sample-ratio mismatch (SRM) check** before reading any metric.
5. **Fixed horizon or a valid sequential procedure**, never ad-hoc peeking.

## Sample size

For a two-sided test of a difference in means with equal allocation:

$$n_{\text{per arm}}=\frac{2\,(z_{1-\alpha/2}+z_{1-\beta})^2\,\sigma^2}{\delta^2}.$$

!!! example "Worked example"
    Baseline conversion 5%, so $\sigma^2=0.05\times0.95=0.0475$. For an absolute MDE $\delta=0.005$ with $\alpha=0.05$, power 0.8: $n=2(1.96+0.84)^2(0.0475)/0.005^2\approx 29\,800$ users per arm.

## Variance reduction with CUPED

Use a pre-experiment covariate $X$ that is independent of assignment:

$$Y^{\text{cv}}=Y-\theta\,(X-\bar X),\qquad \theta=\frac{\operatorname{Cov}(Y,X)}{\operatorname{Var}(X)},\qquad \operatorname{Var}(Y^{\text{cv}})=(1-\rho^2)\operatorname{Var}(Y).$$

The treatment-effect estimate stays unbiased because $X$ is balanced by randomisation; its variance shrinks by the factor $1-\rho^2$, which is equivalent to running the test with $1/(1-\rho^2)$ times as many users.

![](figures/v2_cuped.svg)

@@code code/n02_cuped_srm.py@@
@@out code/out/n02_cuped_srm.txt@@

!!! pitfall "Pitfall: SRM"
    If the observed split differs from the design split (chi-square $p<0.001$ is the common threshold), assignment or logging is broken: bots filtered in one arm, redirects dropping users, or caching. The effect estimate is untrustworthy until the cause is found. In the listing, $p=0.012$ is below that threshold's concern level but is worth watching on the next daily check.

## Ratio metrics and the delta method

For metrics like clicks per session where the unit of randomisation is the user, the metric $R=\bar Y/\bar X$ is a ratio of means. Its variance is

$$\operatorname{Var}(R)\approx\frac{1}{n\,\mu_X^2}\left(\sigma_Y^2-2R\,\sigma_{XY}+R^2\sigma_X^2\right).$$

Treating sessions as independent understates the variance and inflates false positives.

## Peeking and sequential testing

Checking a fixed-horizon p-value every day and stopping at the first $p<0.05$ can push the false-positive rate above 20%. Valid alternatives:

| Method | Idea | When to use |
|---|---|---|
| Group sequential (O'Brien-Fleming) | pre-planned looks with stricter early boundaries | fixed number of interim analyses |
| Alpha spending (Lan-DeMets) | spend $\alpha$ as a function of information fraction | flexible look timing |
| Always-valid inference (mSPRT, confidence sequences) | intervals valid at every sample size | continuous dashboards |
| Bayesian decision with expected loss | stop when the expected loss of shipping is below a threshold | product decisions, not error-rate guarantees |

## Interference, novelty and long-term effects

- **Network interference** (marketplaces, social): treated users affect control users. Use cluster randomisation or switchback designs.
- **Novelty and primacy**: plot the effect by days since exposure; if it decays, extend the test.
- **Long-term effects**: keep a small long-running holdout, or use surrogate metrics validated against past long-run outcomes.

!!! rule "Rule of thumb"
    Run for whole weeks (at least one, preferably two) to average out day-of-week effects, and never fewer than the pre-computed sample size.

## Cheat card: Experimentation

- Pre-register metric, MDE, $\alpha$, power and duration.
- Check SRM first. A failed SRM check invalidates the readout.
- Use CUPED with a pre-period version of the metric; expect 30-70% variance reduction.
- Ratio metrics need the delta method or a bootstrap over the randomisation unit.
- No peeking without a sequential method.
- Guardrails and long-term holdouts protect against local wins and global losses.
