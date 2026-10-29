# Causal Inference in Practice

tagline: From DAGs to doubly robust estimators, difference-in-differences and the assumptions each one buys you.

## Potential outcomes and identification

Each unit has potential outcomes $Y_i(1)$ and $Y_i(0)$; we observe only one. The average treatment effect is $\tau=\mathbb E[Y(1)-Y(0)]$. Observational data identify $\tau$ under three assumptions:

- **Consistency**: the observed outcome equals the potential outcome under the received treatment.
- **Conditional exchangeability**: $Y(t)\perp T\mid X$ (no unmeasured confounding).
- **Positivity**: $0<P(T=1\mid X)<1$ for all $X$ in the population.

## DAGs and the backdoor criterion

![](figures/v2_dag.svg)

Adjust for a set $Z$ that blocks every backdoor path from $T$ to $Y$ and contains no descendant of $T$. In the diagram, adjust for $X$; do **not** adjust for the mediator $M$ (that would remove part of the effect) or for colliders (that opens spurious paths).

!!! pitfall "Pitfall: controlling for everything"
    Adding every available variable to a regression is not conservative. Conditioning on a collider (for example, "was hired" when studying the effect of education on ability) creates bias that did not exist before.

## Estimators

| Estimator | Formula | Consistent if |
|---|---|---|
| Regression adjustment | $\frac1n\sum(\hat\mu_1(X_i)-\hat\mu_0(X_i))$ | outcome model correct |
| IPW | $\frac1n\sum\left(\frac{T_iY_i}{\hat e(X_i)}-\frac{(1-T_i)Y_i}{1-\hat e(X_i)}\right)$ | propensity model correct |
| AIPW / doubly robust | regression + IPW-weighted residuals | either model correct |
| Double ML | AIPW with cross-fitted ML nuisances | nuisances converge at $n^{-1/4}$ |

The AIPW influence function, whose sample mean is the estimate and whose standard deviation gives the standard error:

$$\psi_i=\hat\mu_1(X_i)-\hat\mu_0(X_i)+\frac{T_i(Y_i-\hat\mu_1(X_i))}{\hat e(X_i)}-\frac{(1-T_i)(Y_i-\hat\mu_0(X_i))}{1-\hat e(X_i)}.$$

@@code code/n03_ipw_aipw.py@@
@@out code/out/n03_ipw_aipw.txt@@

!!! tip "Pro tip"
    Inspect the propensity distribution by arm before estimating anything. Extreme weights signal positivity violations; trim the population (and say so) rather than silently clipping.

## Difference-in-differences

With treated and control groups observed before and after a change, under **parallel trends**:

$$\hat\tau_{\text{DiD}}=(\bar Y_{T,\text{post}}-\bar Y_{T,\text{pre}})-(\bar Y_{C,\text{post}}-\bar Y_{C,\text{pre}}).$$

Estimate with two-way fixed effects and cluster standard errors by unit. Test the assumption with an event study: coefficients on leads (periods before treatment) should be near zero.

@@code code/n04_did.py@@
@@out code/out/n04_did.txt@@

!!! pitfall "Pitfall: staggered adoption"
    With treatment starting at different times and effects that vary over time, the plain two-way fixed-effects estimator can weight some comparisons negatively. Use Callaway-Sant'Anna, Sun-Abraham or stacked DiD.

## Other designs

| Design | Key assumption | Example |
|---|---|---|
| Instrumental variables | instrument affects $Y$ only through $T$ (exclusion) and is relevant | distance to college for education |
| Regression discontinuity | units just above and below a cutoff are comparable | scholarship at a test-score threshold |
| Synthetic control | a weighted mix of donors reproduces the treated unit's pre-trend | a policy in one region |
| Uplift / heterogeneous effects | CATE identifiable conditional on $X$ | targeting a promotion |

## Cheat card: Causal inference

- Write the DAG first; choose the adjustment set from it, not from p-values.
- Check positivity (overlap) before estimating.
- Prefer doubly robust estimators with cross-fitting for observational data.
- DiD needs parallel trends; show the event-study leads.
- Report the estimand (ATE, ATT, LATE) explicitly; different designs identify different ones.
- Run sensitivity analysis (E-values, Rosenbaum bounds) for unmeasured confounding.
