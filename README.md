# Data, ML & AI Engineering Handbook

A reference for data and machine learning engineering, from a raw CSV to production systems: the mathematics, the code and the judgement in between.

[![CI](https://img.shields.io/github/actions/workflow/status/T31SHA/data-ml-handbook/ci.yml?branch=main&label=ci&style=flat-square&color=555555)](https://github.com/T31SHA/data-ml-handbook/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/T31SHA/data-ml-handbook?label=release&style=flat-square&color=555555)](https://github.com/T31SHA/data-ml-handbook/releases/latest)
[![Code: MIT](https://img.shields.io/badge/code-MIT-555555?style=flat-square)](LICENSE)
[![Content: CC BY 4.0](https://img.shields.io/badge/content-CC%20BY%204.0-555555?style=flat-square)](LICENSE-CONTENT)

[Read online](https://t31sha.github.io/data-ml-handbook/) · [Download the PDF](https://github.com/T31SHA/data-ml-handbook/releases/latest) · [Notebooks](notebooks/) · [Contributing](CONTRIBUTING.md)

## About

Most references give either the theory or the code. Each topic here follows the same sequence: intuition, mathematics with derivations, a worked numeric example, runnable code with its real output, guidance on when to use it, and common pitfalls.

| Item | Detail |
|---|---|
| Scope | 33 chapters, 175 pages: Python data stack, mathematics, statistics, Bayesian inference, experimentation, causal inference, classical and deep learning, LLM engineering and fine-tuning, forecasting, recommender systems, SQL, data modeling, distributed systems, MLOps and monitoring |
| Mathematics | 881 typeset formulas with derivations |
| Figures | 95 figures, all regenerable from `figures/src/` |
| Code | 61 executed listings; outputs printed in the book come from real runs and are checked in CI |
| Reference | decision flowcharts, debugging playbooks, interview problem sets, glossary and index |

## Contents

| # | Chapter | Read |
|---|---|---|
| 00 | [How to Use This Guide](content/00_orientation.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/00_orientation/) |
| 01 | [Python, NumPy, pandas, Polars & DuckDB](content/01_python.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/01_python/) · [Colab](https://colab.research.google.com/github/T31SHA/data-ml-handbook/blob/main/notebooks/01_python.ipynb) |
| 02 | [From CSV to Model: The Workflow](content/02_workflow.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/02_workflow/) · [Colab](https://colab.research.google.com/github/T31SHA/data-ml-handbook/blob/main/notebooks/02_workflow.ipynb) |
| 03 | [Worked Project: California Housing, End to End](content/03_project.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/03_project/) · [Colab](https://colab.research.google.com/github/T31SHA/data-ml-handbook/blob/main/notebooks/03_project.ipynb) |
| 04 | [Data Visualisation](content/04_viz.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/04_viz/) · [Colab](https://colab.research.google.com/github/T31SHA/data-ml-handbook/blob/main/notebooks/04_viz.ipynb) |
| 05 | [Linear Algebra](content/05_linalg.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/05_linalg/) · [Colab](https://colab.research.google.com/github/T31SHA/data-ml-handbook/blob/main/notebooks/05_linalg.ipynb) |
| 06 | [Calculus & Optimisation](content/06_calculus.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/06_calculus/) · [Colab](https://colab.research.google.com/github/T31SHA/data-ml-handbook/blob/main/notebooks/06_calculus.ipynb) |
| 07 | [Probability, Information Theory & Discrete Maths](content/07_probability.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/07_probability/) · [Colab](https://colab.research.google.com/github/T31SHA/data-ml-handbook/blob/main/notebooks/07_probability.ipynb) |
| 08 | [Statistics, Experimentation & Causal Inference](content/08_statistics.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/08_statistics/) · [Colab](https://colab.research.google.com/github/T31SHA/data-ml-handbook/blob/main/notebooks/08_statistics.ipynb) |
| 09 | [Bayesian Inference in Practice](content/08b_bayesian.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/08b_bayesian/) |
| 10 | [Experimentation and A/B Testing at Scale](content/08c_experimentation.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/08c_experimentation/) |
| 11 | [Causal Inference in Practice](content/08d_causal.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/08d_causal/) |
| 12 | [Classical Machine Learning](content/09_classical_ml.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/09_classical_ml/) · [Colab](https://colab.research.google.com/github/T31SHA/data-ml-handbook/blob/main/notebooks/09_classical_ml.ipynb) |
| 13 | [Evaluation, Validation & Tuning](content/10_evaluation.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/10_evaluation/) · [Colab](https://colab.research.google.com/github/T31SHA/data-ml-handbook/blob/main/notebooks/10_evaluation.ipynb) |
| 14 | [Deep Learning Foundations](content/11_deep_learning.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/11_deep_learning/) · [Colab](https://colab.research.google.com/github/T31SHA/data-ml-handbook/blob/main/notebooks/11_deep_learning.ipynb) |
| 15 | [Architectures: CNNs, RNNs, Transformers](content/12_architectures.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/12_architectures/) · [Colab](https://colab.research.google.com/github/T31SHA/data-ml-handbook/blob/main/notebooks/12_architectures.ipynb) |
| 16 | [LLM Engineering](content/13_llm_engineering.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/13_llm_engineering/) · [Colab](https://colab.research.google.com/github/T31SHA/data-ml-handbook/blob/main/notebooks/13_llm_engineering.ipynb) |
| 17 | [LLM Fine-Tuning, Evaluation and Retrieval in Depth](content/13b_llm_adaptation.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/13b_llm_adaptation/) |
| 18 | [Applied AI: Vision, Generative, RL, RecSys, Time Series, Anomalies, Fairness](content/14_applied_ai.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/14_applied_ai/) · [Colab](https://colab.research.google.com/github/T31SHA/data-ml-handbook/blob/main/notebooks/14_applied_ai.ipynb) |
| 19 | [Time Series Forecasting in Depth](content/14b_time_series.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/14b_time_series/) |
| 20 | [Recommender Systems in Depth](content/14c_recsys.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/14c_recsys/) |
| 21 | [SQL for Analytics & Engineering](content/15_sql.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/15_sql/) · [Colab](https://colab.research.google.com/github/T31SHA/data-ml-handbook/blob/main/notebooks/15_sql.ipynb) |
| 22 | [Data Engineering Foundations](content/16_data_engineering.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/16_data_engineering/) · [Colab](https://colab.research.google.com/github/T31SHA/data-ml-handbook/blob/main/notebooks/16_data_engineering.ipynb) |
| 23 | [Data Modeling for Analytics](content/16b_data_modeling.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/16b_data_modeling/) |
| 24 | [Spark, Kafka, Airflow & dbt](content/17_big_data_tools.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/17_big_data_tools/) |
| 25 | [Distributed Data Systems](content/17b_distributed.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/17b_distributed/) |
| 26 | [MLOps & Production ML System Design](content/18_mlops.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/18_mlops/) · [Colab](https://colab.research.google.com/github/T31SHA/data-ml-handbook/blob/main/notebooks/18_mlops.ipynb) |
| 27 | [ML Monitoring and Reliability](content/18b_monitoring.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/18b_monitoring/) |
| 28 | [Debugging Playbooks](content/19_debugging.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/19_debugging/) · [Colab](https://colab.research.google.com/github/T31SHA/data-ml-handbook/blob/main/notebooks/19_debugging.ipynb) |
| 29 | [Platform Toolbox: Git, Docker, Kubernetes, Cloud, Linux](content/20_platform.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/20_platform/) |
| 30 | [Decision Flowcharts](content/21_decisions.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/21_decisions/) |
| 31 | [Reference: Formula Sheet, Interview Bank & Glossary](content/22_reference.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/22_reference/) |
| 32 | [Interview Problem Sets with Solutions](content/22b_problem_sets.md) | [Docs](https://t31sha.github.io/data-ml-handbook/handbook/22b_problem_sets/) |

## Quickstart

**Just read it:** [online docs](https://t31sha.github.io/data-ml-handbook/) or the [latest PDF release](https://github.com/T31SHA/data-ml-handbook/releases/latest).

**Run the examples locally:**

```bash
git clone https://github.com/T31SHA/data-ml-handbook.git && cd data-ml-handbook
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test            # executes every snippet in code/ (≈4 min on a laptop CPU)
```

Or open any notebook in [`notebooks/`](notebooks/) with one click in Google Colab.

## Build it yourself

| Command | What it does |
|---|---|
| `make pdf` | Renders math with MathJax (Node) → HTML → PDF via WeasyPrint → `dist/Data-ML-Handbook.pdf` |
| `make docs` / `make serve` | Builds / live-serves the MkDocs Material site |
| `make test` | Runs all snippets with pytest |
| `make outputs` | Re-executes snippets and refreshes the outputs printed in the book |
| `make figures` | Regenerates all figures from `figures/src/` |
| `make notebooks` | Regenerates Colab notebooks from `code/` |

Requirements: Python ≥ 3.10, Node ≥ 18, Graphviz (for flowcharts) and the WeasyPrint system libraries (Pango). On Ubuntu:
`sudo apt install graphviz libpango-1.0-0 libpangoft2-1.0-0 fonts-dejavu`. Source Serif 4, Fira Sans and IBM Plex Mono are used if installed (`make fonts`).

## Repository layout

```
content/      numbered Markdown sections — the single source of truth
figures/      SVG/PNG figures  ·  figures/src/ = the scripts that draw them
code/         runnable snippets (+ sql/, out/ = captured outputs shown in the book)
notebooks/    Colab-ready notebooks, one per section (generated)
build/        PDF pipeline: build_pdf.py, style.css, cover.html, MathJax renderer
docs/         MkDocs Material site (handbook pages are generated from content/)
tests/        pytest suite that executes every snippet and checks every include
```

## Contributing

Found an error, a clearer derivation, or a missing topic? Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md),
pick a [`good first issue`](https://github.com/T31SHA/data-ml-handbook/labels/good%20first%20issue), and check the [ROADMAP](ROADMAP.md).
Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Citation

If this handbook helps your work or teaching, please cite it (see [CITATION.cff](CITATION.cff)):

```bibtex
@misc{abdi2026datamlhandbook,
  author = {Abdi, Sharahbil Mohamed},
  title  = {Data/ML Handbook: The Elite Field Guide},
  year   = {2026},
  url    = {https://github.com/T31SHA/data-ml-handbook}
}
```

## License

Code (`code/`, `build/`, `figures/src/`, `tests/`) is [MIT](LICENSE). Written content and figures are [CC BY 4.0](LICENSE-CONTENT).

---

Sharahbil Mohamed Abdi · Data/ML Engineer · [@T31SHA](https://github.com/T31SHA)
