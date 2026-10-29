# Decision Flowcharts

tagline: One page per recurring decision: which model, which metric, which statistical test, which storage or processing engine. Follow the arrows, then read the rules underneath before committing.

## How to use these charts

Each chart encodes the default answer an experienced engineer would reach in under a minute. Defaults are starting points, not verdicts: always beat a trivial baseline, validate on data that looks like production, and revisit the choice when constraints change (latency, interpretability, cost, team skills).

| Chart | Question | Companion sections |
|---|---|---|
| A | Which model family should I start with? | 09 Classical ML, 11–14 Deep learning, LLMs, applied AI |
| B | Which metric should I optimise and report? | 10 Evaluation |
| C | Which statistical test answers my question? | 08 Statistics |
| D | Which storage / processing engine fits the workload? | 16–17 Data engineering |
| E | Why is my model not training? (in section 28) | 28 Debugging |

!!! rule "Rule of thumb: decisions you can reverse vs ones you can't"
    Model and metric choices are cheap to revisit: iterate fast. Storage engines, table formats, data contracts and event schemas are expensive to change later: spend the design time there.

<div class="land" markdown="1">

## A. Which model?

![](figures/flow_model.svg)

!!! rule "Rules that override the chart"
    Need predictions in < 1 ms on a CPU → linear model or small GBM. Regulated decision with adverse-action reasons → monotone-constrained GBM or scorecard + SHAP. Less than ~1,000 labelled rows → simple models with strong priors, or pretrained embeddings + logistic. Extrapolation beyond the training range → linear/parametric component, not trees alone.

</div>

<div class="land" markdown="1">

## B. Which metric?

![](figures/flow_metric.svg)

!!! rule "Rules that override the chart"
    The metric must mirror the business cost: if a false negative costs 40× a false positive, optimise expected cost and choose the threshold from it. Report one primary metric, a few guardrails (latency, calibration, fairness gaps), and always the baseline's score. Ranking problems: evaluate the top of the list (NDCG@k), not the full list.

</div>

<div class="land" markdown="1">

## C. Which statistical test?

![](figures/flow_test.svg)

!!! rule "Rules that override the chart"
    Large samples make everything "significant": always report effect size with a confidence interval. Heavy tails or outliers → permutation test or bootstrap CI. Many tests → control FDR (Benjamini–Hochberg). Observational data → the test answers association, not causation; see causal inference in section 08.

</div>

<div class="land" markdown="1">

## D. Which storage or processing engine?

![](figures/flow_storage.svg)

!!! rule "Rules that override the chart"
    Start with the simplest engine that meets the SLA: Postgres + DuckDB covers far more than people expect. Choose open table formats (Iceberg/Delta) when several engines must read the same data. Streaming only when value decays in seconds to minutes. Team familiarity and managed operations often matter more than benchmark speed.

</div>
