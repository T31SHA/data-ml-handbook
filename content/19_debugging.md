# Debugging Playbooks

tagline: Systematic, symptom-first procedures for the failures that eat weeks: a model that won't learn, NaN losses, leakage that looks like genius, drift and production decay, plus data-pipeline incidents.

## The master triage flow

![](figures/flow_debug.svg)

!!! rule "Rule of thumb: debug from data outward"
    Most ML bugs are data bugs. Before touching the architecture: look at raw examples, check labels, check the pipeline output (after transforms), check splits for leakage, and compare against a trivial baseline. Change one thing at a time and keep a log of what you tried.

## Playbook 1: the model is not learning

| Step | Check | What it reveals |
|---|---|---|
| 1 | Initial loss ≈ theory? ($\ln K$ for $K$ classes, variance of $y$ for MSE) | wrong loss/labels/output layer |
| 2 | Overfit 1 batch (10–100 examples) to ~0 loss | if impossible: bug in model, loss, labels, or gradients |
| 3 | Inputs after the pipeline: plot / print stats | all zeros, unnormalised, shuffled labels, wrong channel order |
| 4 | Labels aligned with inputs? | a shuffle applied to X but not y |
| 5 | Gradients flowing? per-layer grad norms; parameters changing? | frozen layers, `requires_grad=False`, detached tensors, dead ReLUs |
| 6 | Learning rate sweep (LR range test) | LR too low (flat) or too high (noisy/diverging) |
| 7 | `model.train()` vs `model.eval()`; `optimizer.zero_grad()` | dropout/BN in wrong mode; accumulated gradients |
| 8 | Output layer and loss pairing | softmax + CrossEntropyLoss (double softmax), sigmoid + BCE vs logits |
| 9 | Data volume / signal | maybe there is no signal: compare with a GBM and a dummy baseline |

!!! pitfall "Pitfall: the silent killers"
    Forgetting `zero_grad()` (gradients accumulate), evaluating in train mode, normalising validation data with validation statistics, label encoders fitted separately on train and test (category → different integers), and `DataLoader(shuffle=False)` on sorted data (each batch is one class).

## Playbook 2: NaN or Inf loss

@@code code/s19_nan.py@@
@@out code/out/s19_nan.txt@@

| Cause | Signature | Fix |
|---|---|---|
| $\log(0)$, $0\cdot\infty$ | NaN on the first confident wrong prediction | losses on logits (`BCEWithLogitsLoss`, `CrossEntropyLoss`); clamp `eps` |
| Exploding gradients | loss grows then inf/NaN | lower LR, gradient clipping (norm 1.0), warmup, normalisation |
| Unscaled inputs / targets | huge initial loss | standardise X; scale or log-transform y |
| Division by zero / sqrt of negative | NaN in custom ops | add `eps`, check domains; `set_detect_anomaly(True)` |
| fp16 overflow | NaN only with mixed precision | bf16, or GradScaler (dynamic loss scaling) |
| Bad data | NaN in inputs | `assert torch.isfinite(x).all()` in the data pipeline |
| Softmax overflow | inf in exp | subtract max (built-ins do) |

!!! tip "Pro tip: catch NaNs early"
    Add finite-checks on inputs, loss and gradient norm every step and stop with the batch index; save the offending batch for replay. In PyTorch, anomaly detection (above) names the backward op that produced NaN; it is slow, so enable it only while debugging.

## Playbook 3: leakage (too good to be true)

| Symptom | Check |
|---|---|
| Validation score suspiciously high (AUC 0.99 on a hard problem) | top features by importance: any that encode the target or the future? |
| Big gap between offline and online performance | feature availability at prediction time (point-in-time) |
| One feature dominates | is it created after the outcome (e.g. "refund_issued" for churn)? |
| Random split great, time split bad | temporal leakage; entities shared across splits |

| Leak type | Example | Fix |
|---|---|---|
| Target leakage | "account_closed_date" used to predict churn | drop features not available at decision time |
| Preprocessing leakage | scaler/imputer/encoder/feature selection fitted on all data | fit inside a `Pipeline` within CV |
| Temporal leakage | random split on time series; rolling features including the current day | time-based splits; lag features shifted |
| Group leakage | same patient/user in train and test | `GroupKFold` |
| Duplicate leakage | near-duplicate rows across splits | dedup before splitting |
| Test-set reuse | tuning on test | nested CV, untouched holdout |

The workflow section's executed demo: feature selection on all data before CV reported **0.795** accuracy on pure noise; doing it inside the pipeline gave the honest **0.51**.

## Playbook 4: performance decays in production

| Step | Question | Tool |
|---|---|---|
| 1 | Is it the data pipeline? schema change, nulls, units, late data | data quality monitors, lineage |
| 2 | Is training/serving skew present? | compare logged online features with offline recomputation for the same entities |
| 3 | Covariate drift? which features moved | PSI/KS per feature, segment breakdown |
| 4 | Label/prior shift? | positive rate over time; recalibrate |
| 5 | Concept drift? | performance by time slice on matured labels |
| 6 | Feedback loop? | the model's own actions changed the data (recommendations, fraud blocks) |
| 7 | Fix | rollback, recalibrate, retrain on recent data, add features, change the window |

!!! interview "Interview: accuracy dropped 10% last week. What do you do?"
    First rule out the pipeline (schema, nulls, volume, upstream deploys). Then compare feature distributions this week vs training (PSI), segment the drop (country, device, new vs returning), check label delays and base rate. If data drift: retrain/recalibrate. If a broken feature: fix or fall back. Communicate impact and add a monitor that would have caught it earlier.

## Playbook 5: overfitting and underfitting

| | Underfitting (high bias) | Overfitting (high variance) |
|---|---|---|
| Signal | train and val both poor | train ≫ val |
| Data | better features | more data, augmentation |
| Model | more capacity, less regularisation, longer training | regularise (L2, dropout), smaller model, early stopping |
| Trees | deeper trees, more rounds | min_child_samples ↑, max_depth ↓, subsample, early stopping |
| Check | learning curve plateaus high | learning curve gap closes with data |

## Playbook 6: data pipeline incidents

| Incident | First moves | Prevention |
|---|---|---|
| Job failed | read the error, check upstream freshness, rerun idempotently for the date | retries with backoff, alerts with owner |
| Duplicates in a table | find the key, check retries/non-idempotent inserts | MERGE / partition overwrite, unique tests |
| Missing data for a day | source availability, watermark/late data | freshness checks, reprocessing window |
| Numbers don't match the dashboard | grain, filters, time zones, join fan-out | documented metric definitions (semantic layer) |
| Pipeline suddenly slow | data volume, skew, small files, plan change | monitor runtime and bytes scanned |
| Schema change broke it | contract violation upstream | schema registry, contracts in CI |

## Playbook 7: slow training or inference

| Bottleneck | Symptom | Fix |
|---|---|---|
| Input pipeline | GPU utilisation low, CPU busy | more `num_workers`, `pin_memory`, prefetch, pre-processed formats |
| Small batches | GPU underused | larger batch (+ LR scaling), gradient accumulation |
| Precision | fp32 everywhere | bf16 autocast, TF32 |
| Kernel overhead | many small ops | `torch.compile`, fused optimisers |
| Memory | OOM | gradient checkpointing, smaller batch, 8-bit optimisers, FSDP |
| Inference | high latency | batching, quantisation, ONNX/TensorRT, distillation, caching |

<div class="card" markdown="1">

## Cheat card: Debugging

<div class="cols" markdown="1">

#### Not learning
- Loss at init ≈ ln K?
- Overfit one batch
- Inspect data after transforms; labels aligned?
- Grad norms per layer; LR range test
- zero_grad, train/eval modes, logits vs probabilities

#### NaN / Inf
- Loss on logits; eps in log/div/sqrt
- Lower LR, clip gradients, warmup
- Scale inputs and targets
- bf16 instead of fp16; GradScaler
- `set_detect_anomaly(True)` to locate

#### Leakage
- Too good → suspect it
- Features available at decision time only
- Pipeline inside CV; time & group splits; dedup

#### Production decay
- Pipeline → skew → drift → label shift → concept drift → feedback loops
- Segment the drop; PSI per feature

#### Pipelines
- Idempotent reruns by date; unique tests
- Freshness + volume + schema monitors

#### Speed
- GPU idle → input pipeline
- bf16, compile, bigger batch, checkpointing

</div>
</div>
