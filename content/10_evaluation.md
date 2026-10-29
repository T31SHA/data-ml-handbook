# Evaluation, Validation & Tuning

tagline: How to know a model is good before production tells you: split strategies, every metric with its formula and trap, calibration, thresholds, learning curves, hyperparameter search and explainability.

## Splitting: the most consequential decision

| Data | Split | Why |
|---|---|---|
| i.i.d. rows | stratified train/val/test (60/20/20) or K-fold | class balance preserved |
| Time series | train on past, validate on future: `TimeSeriesSplit`, expanding/rolling window, gap | future info must never reach the past |
| Groups (users, patients) | `GroupKFold`, `StratifiedGroupKFold` | the same user in train and test leaks identity |
| Small data | repeated stratified K-fold (5×5) | variance of the estimate |
| Tuning + honest score | nested CV | the selection itself overfits |
| Production-like | out-of-time holdout + shadow traffic | catches drift and pipeline skew |

![](figures/kfold.svg)

@@code code/s08_nested.py@@
@@out code/out/s08_nested.txt@@

!!! pitfall "Pitfall: the test set is not a tuning set"
    Every time you look at test performance and change something, the test set becomes training data for your decisions. With enough iterations you will overfit it. Use validation (or inner CV) for every choice; touch the test set once, at the end.

## Classification metrics

![](figures/confusion.svg)

| Metric | Formula | Question it answers | Trap |
|---|---|---|---|
| Accuracy | $\frac{TP+TN}{N}$ | how often right? | useless when imbalanced |
| Precision | $\frac{TP}{TP+FP}$ | when I flag, am I right? | ignores misses |
| Recall (TPR) | $\frac{TP}{TP+FN}$ | how many positives did I catch? | ignores false alarms |
| Specificity | $\frac{TN}{TN+FP}$ | how many negatives cleared? | |
| F1 | $\frac{2PR}{P+R}$ | balance P and R | ignores TN, threshold-dependent |
| $F_\beta$ | $\frac{(1+\beta^2)PR}{\beta^2P+R}$ | recall $\beta$ times as important | |
| MCC | $\frac{TP\cdot TN - FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | balanced single number | less intuitive |
| ROC-AUC | $P(s_+ > s_-)$ | ranking quality | optimistic under heavy imbalance |
| PR-AUC (AP) | area under P–R | ranking of rare positives | baseline = prevalence, not 0.5 |
| Log-loss | $-\frac1n\sum[y\log p + (1-y)\log(1-p)]$ | probability quality | punishes confident errors hard |
| Brier | $\frac1n\sum(p-y)^2$ | calibration + sharpness | |

@@code code/s08_metrics.py@@
@@out code/out/s08_metrics.txt@@

!!! example "Worked example: the confusion matrix above"
    10 cases, threshold 0.5: TP 3, FP 1, FN 1, TN 5. Precision $= 3/4$, recall $= 3/4$, $F_1 = 0.75$. AUC 0.875: of the $4\times6 = 24$ (positive, negative) pairs, 21 are ranked correctly. The pairwise check in the code reproduces it exactly.

![](figures/roc_pr.svg)

!!! interview "Interview: ROC-AUC or PR-AUC?"
    ROC-AUC uses FPR $= FP/(FP+TN)$; with millions of negatives, thousands of false positives barely move it. PR-AUC uses precision, which is directly hurt by them. For rare-event problems (fraud, defects, rare diseases) report PR-AUC and precision at the operating recall.

### Thresholds are a business decision

![](figures/threshold.svg)

Choose the threshold minimising expected cost: $t^* = \arg\min_t\ c_{FP}\,FP(t) + c_{FN}\,FN(t)$. For calibrated probabilities the Bayes-optimal rule is: flag if $p > \frac{c_{FP}}{c_{FP} + c_{FN}}$. **Example:** a missed fraud costs USD 200 and a manual review USD 5: flag when $p > 5/205 = 2.4\%$.

### Calibration

![](figures/calibration.svg)

A model is calibrated if, among cases scored 0.7, about 70% are positive. Trees/boosting and SVMs are often miscalibrated; class weights and resampling distort probabilities. **Fix:** `CalibratedClassifierCV(method="isotonic")` (≥ 1k samples) or `"sigmoid"` (Platt, small data), fitted on held-out data. Measure with reliability diagrams, Brier score, expected calibration error $\text{ECE} = \sum_b \frac{n_b}{n}\lvert\bar y_b - \bar p_b\rvert$.

## Regression metrics

| Metric | Formula | Robust | Use |
|---|---|---|---|
| MAE | $\frac1n\sum\lvert y-\hat y\rvert$ | yes | interpretable units; median-optimal |
| RMSE | $\sqrt{\frac1n\sum(y-\hat y)^2}$ | no | penalises big errors; mean-optimal |
| MAPE | $\frac{100}{n}\sum\lvert\frac{y-\hat y}{y}\rvert$ | — | undefined at $y=0$; asymmetric |
| sMAPE / WAPE | $\frac{\sum\lvert y-\hat y\rvert}{\sum\lvert y\rvert}$ | yes | forecasting across SKUs |
| $R^2$ | $1 - \frac{SS_{res}}{SS_{tot}}$ | no | variance explained; can be negative on test |
| RMSLE | RMSE on $\log(1+y)$ | partly | relative errors, right-skewed targets |
| Pinball | $\max(q\,e, (q-1)e)$ | | quantile forecasts / prediction intervals |
| MASE | MAE / MAE of naive forecast | yes | time series, scale-free |

!!! rule "Rule of thumb: the loss you train is the statistic you estimate"
    Squared error predicts the conditional **mean**, absolute error the **median**, pinball loss at $q$ the $q$-th **quantile**. If the business cares about the median delivery time, do not train on MSE.

## Ranking & recommendation metrics

| Metric | Formula | Note |
|---|---|---|
| Precision@k | relevant in top-k / k | ignores order inside k |
| Recall@k | relevant in top-k / all relevant | retrieval stage |
| MRR | $\frac1{\lvert Q\rvert}\sum 1/\text{rank}_1$ | first relevant hit |
| MAP | mean of average precision | binary relevance |
| NDCG@k | $\frac{\text{DCG}}{\text{IDCG}}$, $\text{DCG} = \sum \frac{2^{rel_i}-1}{\log_2(i+1)}$ | graded relevance, position-discounted |
| Coverage / diversity | catalogue share recommended | beyond accuracy |

## Diagnosing with curves

![](figures/learning_curves_real.svg)

| Pattern | Diagnosis | Action |
|---|---|---|
| Train and val both poor, close | high bias (underfit) | more features, bigger model, less regularisation |
| Train great, val poor, big gap | high variance (overfit) | more data, regularise, simplify, augment |
| Gap closes as data grows | data-hungry | collect / label more |
| Val curve flat with more data | capacity-bound | change model family |
| Val better than train | dropout/augmentation on train only, or leakage into val | check splits |

![](figures/validation_curve.svg)

## Hyperparameter optimisation

| Method | How | Use when |
|---|---|---|
| Grid search | all combinations | ≤ 2–3 params, cheap model |
| Random search | sample distributions | default; better than grid when few params matter (Bergstra & Bengio) |
| Bayesian (TPE, GP) | model the score surface, pick promising points | expensive models: Optuna, Hyperopt |
| Successive halving / Hyperband | kill bad configs early | many configs, iterative training |
| Population-based training | evolve configs during training | large deep-learning runs |

!!! tip "Pro tip: search spaces"
    Sample learning rates, regularisation strengths and `C` on a **log** scale; integers like `num_leaves` on a quantised log scale. Fix the number of boosting rounds with early stopping rather than searching it. Track every trial (Optuna + MLflow) so the search itself is reproducible.

![](figures/proj_optuna.svg)

## Explainability (XAI)

![](figures/importance.svg)

| Method | Scope | Model-agnostic | Watch out |
|---|---|---|---|
| Coefficients | global | linear only | scale-dependent; correlated features |
| Impurity importance | global | trees | biased to high cardinality; training-set based |
| Permutation importance | global | yes | correlated features share/split importance; use validation data |
| SHAP | local + global | yes (TreeSHAP fast) | explains the model, not causality |
| PDP / ICE | global / local effect curve | yes | assumes feature independence (use ALE if correlated) |
| LIME | local | yes | unstable across runs |
| Counterfactuals | local | yes | "what minimal change flips the decision?" |

!!! derivation "Derivation: Shapley values"
    Feature $j$'s contribution is its average marginal effect across all orderings: $\phi_j = \sum_{S\subseteq F\setminus\{j\}} \frac{\lvert S\rvert!\,(\lvert F\rvert-\lvert S\rvert-1)!}{\lvert F\rvert!}\big[v(S\cup\{j\}) - v(S)\big]$. They satisfy efficiency, $\sum_j\phi_j = f(\mathbf x) - \mathbb E[f]$: the attributions add up exactly to the prediction minus the baseline. For a linear model with independent features, $\phi_j = w_j(x_j - \mathbb E[x_j])$, which is what the beeswarm plot above computes.

<div class="card" markdown="1">

## Cheat card: Evaluation & tuning

<div class="cols" markdown="1">

#### Splits
- i.i.d. → stratified K-fold (5)
- Time → `TimeSeriesSplit` + gap; out-of-time test
- Users/patients → `GroupKFold`
- Tuning + estimate → nested CV
- Fit preprocessing inside the CV pipeline

#### Classification
- Imbalanced: PR-AUC, recall@precision
- Ranking quality: ROC-AUC = P(pos > neg)
- Probabilities: log-loss, Brier, ECE
- Decision: cost-based threshold $p > \frac{c_{FP}}{c_{FP}+c_{FN}}$
- Single balanced number: MCC

#### Regression
- MAE ↔ median, RMSE ↔ mean
- Skewed targets: RMSLE or log-target
- Forecast: MASE, WAPE, pinball for intervals

#### Curves
- Learning curve: bias vs variance
- Validation curve: tune one knob
- Reliability diagram: calibration

#### Tuning
- Random / TPE on log scales
- Early stopping instead of searching rounds
- Report mean ± std across folds

#### Explain
- Permutation importance on validation
- SHAP for local + global
- ALE when features correlate
- Explanations ≠ causation

</div>
</div>
