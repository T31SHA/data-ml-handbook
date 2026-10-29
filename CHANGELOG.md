# Changelog
All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com), versioning: [SemVer](https://semver.org).

## [2.0.0] - 2026-10-29
### Added
- Ten new chapters: Bayesian Inference in Practice; Experimentation and A/B Testing at Scale; Causal Inference in Practice; LLM Fine-Tuning, Evaluation and Retrieval in Depth; Time Series Forecasting in Depth; Recommender Systems in Depth; Data Modeling for Analytics; Distributed Data Systems; ML Monitoring and Reliability; Interview Problem Sets with Solutions.
- 11 new executed snippets (`code/n01`–`n11`), 6 new figures (`figures/src/figs_v2.py`), 113 new typeset formulas.
### Changed
- Book typesetting redesigned: serif text face (Source Serif), near-monochrome palette with one ink-blue accent, plain title page, quiet running heads and folios, rule-based callouts, booktabs-style tables, greyscale code highlighting.
- Orientation chapter and decision-flowchart cross-references renumbered for 33 chapters.
- README simplified: no emoji or banner, one row of flat neutral badges.

## [1.0.0] - 2026-10-29
### Added
- 23 sections (00–22): Python/pandas/Polars/DuckDB, CSV→model workflow, end-to-end California Housing project, visualisation, linear algebra, calculus & optimisation, probability & information theory, statistics & causal inference, classical ML, evaluation & tuning, deep learning (PyTorch + Keras), CNN/RNN/Transformer architectures, LLM engineering (KV cache, LoRA, RAG, guardrails), applied AI, SQL, data engineering, Spark/Kafka/Airflow/dbt, MLOps, debugging playbooks, platform toolbox, decision flowcharts, reference (formula sheet, interview bank, glossary).
- 89 figures with generator scripts; 768 MathJax-typeset formulas; 50 executed snippets; 18 Colab notebooks.
- PDF pipeline (MathJax → WeasyPrint), MkDocs Material site, CI, Pages deploy and release workflows.
