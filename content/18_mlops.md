# MLOps & Production ML System Design

tagline: Turning a notebook into a dependable service: lifecycle and maturity, experiment tracking and reproducibility, feature stores, serving patterns, monitoring and drift, retraining, and the system-design patterns interviewers and production both demand.

## The ML lifecycle

![](figures/mlops.svg)

| Maturity | Characteristics | Next step |
|---|---|---|
| 0 Manual | notebooks, manual deploys, no monitoring | version code + data; scripted training |
| 1 Pipeline | automated training pipeline, model registry, basic monitoring | CI for pipelines, data validation |
| 2 CI/CD/CT | automated tests, continuous training on triggers, canary deploys, drift-based retraining | feature store, lineage, self-serve platform |

!!! rule "Rule of thumb: the ML code is the small box"
    In production systems, model code is a few percent of the total; data collection, validation, feature pipelines, serving, monitoring and configuration dominate (Sculley et al., "Hidden Technical Debt in ML Systems"). Invest in the plumbing early.

## Experiment tracking and reproducibility

@@code code/s15_mlflow.py@@
@@out code/out/s15_mlflow.txt@@

### Reproducibility checklist

| # | Item | How |
|---|---|---|
| 1 | Code version | git commit hash logged with every run; clean working tree |
| 2 | Data version | immutable snapshot / table version (Delta/Iceberg time travel, DVC, lakeFS) + hash |
| 3 | Environment | lock file (`uv.lock`, `poetry.lock`, pinned `requirements.txt`), Docker image digest |
| 4 | Randomness | seeds for Python, NumPy, framework; deterministic algorithms where needed; log the seed |
| 5 | Config | all hyperparameters in a config file (Hydra/YAML) logged as params, not hard-coded |
| 6 | Splits | split indices or split logic + seed saved; time cutoffs explicit |
| 7 | Metrics | primary + guardrail metrics, per-slice metrics, with CIs or fold std |
| 8 | Artifacts | model, preprocessing pipeline, feature list, thresholds, calibration in one versioned package |
| 9 | Lineage | run → data version → code → model version → deployment |
| 10 | Hardware | GPU type, driver/CUDA versions (numerics differ) |
| 11 | Re-run test | someone else reproduces the metric from the run id alone |
| 12 | Model card | intended use, data, metrics, limits, owners |

!!! pitfall "Pitfall: non-determinism on GPUs"
    Even with seeds, cuDNN autotuning and atomic floating-point adds make runs differ slightly. Set `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` when exact reproducibility matters; otherwise report variance across 3–5 seeds instead of a single number.

## Feature stores and training/serving skew

![](figures/feature_store.svg)

| Concern | Feature store answer |
|---|---|
| Same logic offline and online | one feature definition materialised to both stores |
| Point-in-time correctness | training sets built with as-of joins on event timestamps: no leakage from the future |
| Low-latency lookup | online store (Redis, DynamoDB, Bigtable) keyed by entity |
| Reuse and discovery | registry with owners, descriptions, lineage |
| Freshness | batch, streaming or on-demand features with TTLs |

!!! pitfall "Pitfall: training/serving skew"
    The model sees one feature computation in training (SQL over the warehouse) and another in serving (Java in the app). Small differences (time zones, null handling, rounding, different aggregation windows) silently destroy accuracy. Share code or definitions, log the features actually served, and compare them with training distributions.

## Serving patterns

![](figures/serving_arch.svg)

![](figures/inference_patterns.svg)

| Pattern | Latency | Use | Notes |
|---|---|---|---|
| Batch scoring | hours | churn lists, recommendations precomputed nightly | simplest; results in a table/cache |
| Online (sync API) | ms–100s of ms | fraud check, search ranking, pricing | needs online features, autoscaling, p99 SLOs |
| Streaming | seconds | real-time anomaly detection, event enrichment | model in Flink/Kafka Streams consumer |
| Edge / on-device | ms, offline | mobile vision, keyboards | quantisation, TFLite/Core ML/ONNX |
| Async / queue | seconds–minutes | document processing, LLM jobs | job queue + callback/webhook |

### Deployment strategies

| Strategy | How | Risk control |
|---|---|---|
| Shadow | new model scores live traffic, results not used | compare predictions and latency safely |
| Canary | small % of traffic → new model, ramp up | automatic rollback on metric regression |
| Blue/green | two full environments, switch traffic | instant rollback |
| A/B test | randomised split, measure business metric | causal evidence of impact |
| Interleaving | mix results of two rankers in one list | sensitive ranking comparison |
| Multi-armed bandit | shift traffic to the winner adaptively | less regret, weaker inference |

### Executed: a model behind an HTTP API (from the end-to-end project)

@@code code/proj05_serve.py@@
@@out code/out/proj05_serve.txt@@

## Monitoring and drift

| Layer | Monitor | Example alert |
|---|---|---|
| Service | latency p50/p95/p99, error rate, throughput, saturation | p99 > 200 ms for 10 min |
| Data | schema, nulls, ranges, volume, freshness | null rate of `income` > 5% |
| Feature drift | PSI/KS/JS per feature vs training | PSI > 0.25 |
| Prediction drift | score distribution, positive rate | positive rate doubled |
| Performance | metric on labelled feedback (delayed) | weekly AUC down 3 points |
| Business | conversion, revenue, complaints | |
| Fairness | metrics per segment | TPR gap > 5 points |

| Drift type | Definition | Example |
|---|---|---|
| Covariate (data) drift | $P(X)$ changes | new user acquisition channel |
| Prior (label) shift | $P(Y)$ changes | fraud rate spikes during holidays |
| Concept drift | $P(Y\mid X)$ changes | same behaviour now means something else (new fraud pattern) |
| Upstream change | pipeline bug, unit change | currency switched from KES to USD |

$$\text{PSI} = \sum_{b}(a_b - e_b)\ln\frac{a_b}{e_b}$$

**Intuition:** a symmetric KL-like distance between binned training ($e$) and production ($a$) proportions. **Example:** one bin moving from 10% to 15% contributes $(0.15 - 0.10)\ln(1.5) = 0.020$. Thresholds: < 0.1 stable, 0.1–0.25 moderate, > 0.25 significant.

@@code code/s13_psi.py@@
@@out code/out/s13_psi.txt@@

!!! tip "Pro tip: retraining triggers"
    Retrain on a schedule matched to how fast the world changes (weekly/monthly), plus triggers on performance drops or significant drift. Always gate promotion: the challenger must beat the champion on a recent holdout and pass slice/fairness checks, then go through shadow or canary.

## Production ML design patterns

| Pattern | Problem | Solution |
|---|---|---|
| Pipeline-as-artifact | preprocessing differs between train and serve | serialise preprocessing + model together (sklearn Pipeline, TF Transform, ONNX) |
| Feature store | skew, reuse, point-in-time joins | central offline/online feature definitions |
| Two-stage (retrieve → rank) | millions of candidates, tight latency | cheap recall model + expensive precise ranker |
| Cascade | most inputs are easy | cheap model first; escalate uncertain cases to a heavier model or a human |
| Fallback / default | model or feature service down | rules or cached predictions; never fail closed silently |
| Champion/challenger | safe model updates | shadow + canary + automated comparison |
| Human-in-the-loop | high stakes, low confidence | review queue; labels feed retraining |
| Embeddings as a service | many teams need representations | shared embedding models + vector index |
| Batch + real-time hybrid (lambda-style features) | freshness vs cost | nightly heavy aggregates + streaming deltas |
| Rebalancing / calibration layer | priors shift | recalibrate scores without full retraining |
| Model registry + stages | who deployed what | registry with versions, stages, approvals, lineage |
| Kill switch / feature flags | bad model in production | instant disable without deploy |

### System design interview template

| Step | Content |
|---|---|
| 1. Clarify | business goal, users, scale (QPS, data size), latency, constraints, success metric |
| 2. Frame | ML task (classification/ranking/regression), labels and their delay, offline metric ↔ online metric |
| 3. Data | sources, label generation, features (user, item, context, cross), point-in-time correctness |
| 4. Model | baseline (heuristic, logistic), then GBM / two-tower / Transformer; trade-offs |
| 5. Training | splits by time, imbalance handling, retraining cadence, infra |
| 6. Serving | batch vs online, architecture (retrieve → rank → re-rank), caching, latency budget |
| 7. Evaluation | offline metrics, A/B test design, guardrails |
| 8. Monitoring | drift, performance, fairness, alerting, fallback |
| 9. Iterate | feedback loops, exploration, cost |

!!! interview "Interview: design a fraud detection system"
    Label = chargeback (delayed 30–90 days) plus analyst decisions. Features: transaction amount and merchant, velocity aggregates (count/sum over 1 h/24 h per card, device, IP) from streaming, graph features (shared devices). Model: GBM, calibrated; latency budget ~50 ms; rules for known patterns as a first stage. Threshold by cost (fraud loss vs friction), with a review band sent to analysts. Monitor PR-AUC on matured labels, alert rate, and drift; retrain weekly. Handle adversarial adaptation with fast retraining and anomaly detectors.

<div class="card" markdown="1">

## Cheat card: MLOps & system design

<div class="cols" markdown="1">

#### Track every run
- Code commit, data version, environment lock, seed, config, metrics, artifacts
- MLflow / W&B; registry with stages

#### Ship safely
- Pipeline (preprocess + model) as one artifact
- Shadow → canary → full; automatic rollback
- A/B test for business impact; kill switch

#### Serve
- Batch if hours are fine; online for ms decisions
- Retrieve → rank → re-rank for large catalogues
- Cache, autoscale, p99 SLOs, fallbacks

#### Features
- Feature store: offline/online parity
- Point-in-time joins; log served features

#### Monitor
- Service, data, drift (PSI > 0.25), predictions, performance, business, fairness
- Covariate vs label vs concept drift

#### Retrain
- Schedule + triggers; challenger must beat champion on recent data and slices

#### Interview template
- Clarify → frame → data → model → training → serving → evaluation → monitoring → iterate

</div>
</div>
