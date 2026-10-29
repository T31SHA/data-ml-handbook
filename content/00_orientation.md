# How to Use This Guide

tagline: The map of the territory, the conventions used on every page, and the one diagram to remember.

## The end-to-end map

![](figures/pipeline.svg)

Every ML or data project moves through the same loop: **ingest → understand → clean → engineer → split → model → evaluate → ship → monitor → retrain**. The sections follow that loop, then go deeper into the mathematics, the specialised domains (LLMs, vision, time series, recommender systems) and the data platform underneath it all.

| Part | Sections | Read it when you need to… |
|---|---|---|
| Foundations | 01 Python stack · 02 CSV→model workflow · 03 worked project · 04 visualisation | go from a file to a trustworthy baseline quickly |
| Mathematics | 05 linear algebra · 06 calculus & optimisation · 07 probability · 08 statistics & causal inference | understand *why* algorithms work, derive gradients, run experiments |
| Machine learning | 09 classical ML · 10 evaluation & tuning · 11 deep learning · 12 architectures | choose, train, tune and diagnose models |
| AI engineering | 13 LLM engineering · 14 applied AI domains | build RAG/agents, CV, RL, recsys, forecasting, anomaly detection, fair & explainable systems |
| Data engineering | 15 SQL · 16 foundations · 17 Spark/Kafka/Airflow/dbt | design pipelines, warehouses and lakehouses |
| Production | 18 MLOps & system design · 19 debugging playbooks · 20 platform toolbox | ship, monitor and fix systems in production |
| Decisions & recall | 21 decision flowcharts · 22 reference, glossary, interview · Index | decide fast, revise fast, find anything |

## Conventions

!!! intuition "Intuition"
    The plain-language idea. If you only read one box, read this one.

!!! derivation "Derivation"
    The maths, step by step, so you can re-derive it in an interview or debug an implementation.

!!! example "Worked example"
    A tiny numeric example you can check by hand.

!!! pitfall "Pitfall"
    A mistake that silently produces wrong results. These are the expensive ones.

!!! tip "Pro tip"
    What experienced practitioners do differently.

!!! rule "Rule of thumb"
    A sane default that is right most of the time.

!!! interview "Interview question"
    A frequently asked question with a crisp model answer.

**Code listings.** Every Python and SQL listing marked with an *Output* panel was executed on the build machine (Python 3.13, scikit-learn, LightGBM, XGBoost, PyTorch 2 (CPU), TensorFlow 2 / Keras 3, DuckDB, Polars, statsmodels, MLflow, pandera). The output shown is the real output. Listings for cluster tools (Spark, Kafka, Airflow, dbt, Kubernetes) are reference syntax and are labelled as not executed.

**Notation.** Scalars $x$, vectors $\mathbf{x}$ (column by default), matrices $X$, transpose $X^\top$, data matrix $X \in \mathbb{R}^{n \times d}$ ($n$ rows/samples, $d$ features), parameters $\mathbf{w}, \theta$, prediction $\hat{y}$, loss $\mathcal{L}$, learning rate $\eta$, expectation $\mathbb{E}$, probability $P$, normal $\mathcal{N}(\mu, \sigma^2)$, indicator $\mathbb{1}[\cdot]$, element-wise product $\odot$.

## The ten commandments of applied ML

1. **Define the decision first.** What action will a prediction trigger, what does each error type cost, and what is the baseline?
2. **Split before you look.** Hold out the test set before EDA-driven choices; split by time or by group when the data has that structure.
3. **Fit every transformation on training data only.** Put it in a pipeline.
4. **Beat a dumb baseline** on the right metric, with a confidence interval.
5. **Start simple** (linear / gradient boosting / pretrained model) and add complexity only when validation says so.
6. **Look at the errors**, slice by slice, not just at the aggregate metric.
7. **Touch the test set once.**
8. **Make it reproducible:** versioned data, code, config, seeds, environment.
9. **Ship with monitoring:** data quality, drift, performance, latency, cost.
10. **Write it down:** model card, data lineage, assumptions, known failure modes.
