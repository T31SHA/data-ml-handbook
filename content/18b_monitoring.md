# ML Monitoring and Reliability

tagline: Detecting data drift, concept drift and silent failures, and responding with alerts, rollbacks and retraining.

## What can go wrong

| Failure | Signal | Detect with |
|---|---|---|
| Pipeline break (nulls, schema change) | sudden jump in null rate, cardinality | data contracts, schema checks |
| Covariate drift $P(X)$ changes | input distributions shift | PSI, KS, embedding distance |
| Prior shift $P(Y)$ changes | prediction mean moves | label-free: prediction distribution |
| Concept drift $P(Y\mid X)$ changes | accuracy falls with stable inputs | delayed labels, proxy metrics |
| Training/serving skew | offline good, online bad | log features at serving and compare |

## Drift statistics

Population Stability Index over $B$ bins with reference shares $a_b$ and current shares $c_b$:

$$\text{PSI}=\sum_{b=1}^{B}(c_b-a_b)\ln\frac{c_b}{a_b}.$$

Common thresholds: below 0.1 stable, 0.1-0.25 moderate, above 0.25 large.

![](figures/v2_drift.svg)

@@code code/n10_drift.py@@
@@out code/out/n10_drift.txt@@

!!! pitfall "Pitfall: p-values on large samples"
    With thousands of rows, KS flags trivially small shifts as significant, as the 0.25 sd row shows. Alert on effect sizes (PSI, KS statistic) with thresholds tied to model impact, not on p-values.

## Performance monitoring with delayed labels

When labels arrive late, track proxies (click-through, complaint rate), prediction-distribution stability and calibration on the subset already labelled. Estimate performance under covariate shift by importance weighting with a domain classifier: $w(x)=\frac{P(\text{current}\mid x)}{P(\text{reference}\mid x)}$.

## Operating the system

- **Service level objectives** for latency, availability and freshness of features.
- **Shadow and canary deployments** before full rollout; automated rollback on guardrail breach.
- **Retraining policy**: scheduled, triggered by drift, or both. Every retrained model passes the same evaluation gate as the first.
- **Runbooks** for each alert: what it means, the first three things to check, and who owns it.

!!! rule "Rule of thumb"
    Monitor inputs, outputs and outcomes. Inputs warn earliest, outcomes matter most.

## Cheat card: Monitoring

- Data contracts catch breaks; drift metrics catch shifts; outcome metrics catch harm.
- PSI below 0.1 stable, above 0.25 investigate.
- Alert on effect size, not p-value.
- Log serving features to detect training/serving skew.
- Canary, shadow, rollback and gated retraining.
