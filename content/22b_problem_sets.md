# Interview Problem Sets with Solutions

tagline: Worked problems across probability, statistics, ML, SQL and systems, each with a model answer.

## Probability

**P1.** A test for a condition with 1% prevalence has sensitivity 0.95 and specificity 0.90. What is $P(\text{condition}\mid +)$?

*Solution.* $\frac{0.95\times0.01}{0.95\times0.01+0.10\times0.99}=\frac{0.0095}{0.1085}\approx0.088$. Low prevalence dominates: most positives are false.

**P2.** Expected number of fair-coin tosses to see two heads in a row?

*Solution.* Let $E$ be the expected count from the start and $E_H$ after one head. $E=1+\tfrac12E_H+\tfrac12E$ and $E_H=1+\tfrac12\cdot0+\tfrac12E$. Solving gives $E=6$.

**P3.** Sample $k$ items uniformly from a stream of unknown length in $O(k)$ memory.

*Solution.* Reservoir sampling: keep the first $k$; for item $i$ (0-indexed, $i\ge k$) replace a random slot with probability $k/(i+1)$. Each item ends in the reservoir with probability $k/n$.

@@code code/n11_interview.py@@
@@out code/out/n11_interview.txt@@

## Statistics

**S1.** Why does the naive formula $\mathbb E[X^2]-\mathbb E[X]^2$ fail in the listing above?

*Solution.* Catastrophic cancellation: both terms are about $10^{18}$ and their difference (about 1) is below the precision of a double at that magnitude. Welford's update works with deviations from the running mean.

**S2.** An A/B test shows $p=0.04$ on the 9th of 20 metrics checked. Ship?

*Solution.* Not on that evidence. With 20 metrics at $\alpha=0.05$ one false positive is expected. Pre-declare a primary metric, or control the false discovery rate (Benjamini-Hochberg) across secondary metrics.

**S3.** Show that AUC equals the probability that a random positive scores above a random negative.

*Solution.* The ROC curve integrates the true-positive rate over false-positive thresholds; each negative contributes the fraction of positives scored above it. Averaging gives the Mann-Whitney statistic $U/(n_+n_-)$, as the listing confirms numerically.

## Machine learning

**M1.** Training loss falls, validation loss rises after epoch 5. List three fixes in order of cost.

*Solution.* Early stopping at the validation minimum; stronger regularisation (weight decay, dropout, augmentation); more or cleaner data.

**M2.** Your fraud model has 99.5% accuracy. Is it good?

*Solution.* Unknown. With 0.5% fraud, predicting "not fraud" always scores 99.5%. Report precision-recall (PR-AUC) and the cost-weighted operating point.

**M3.** Why scale features for $k$-NN and ridge but not for tree ensembles?

*Solution.* Distances and the $L_2$ penalty depend on units; tree splits depend only on the ordering within a feature.

## SQL

**Q1.** Find each user's second most recent order.

```sql
SELECT user_id, order_id, ordered_at
FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY ordered_at DESC) AS rn
  FROM orders
) t
WHERE rn = 2;
```

**Q2.** Day-over-day retention: share of users active on day $d$ who are also active on day $d+1$.

```sql
SELECT a.day, COUNT(DISTINCT b.user_id) * 1.0 / COUNT(DISTINCT a.user_id) AS next_day_retention
FROM activity a
LEFT JOIN activity b ON b.user_id = a.user_id AND b.day = a.day + INTERVAL 1 DAY
GROUP BY a.day
ORDER BY a.day;
```

## Systems

**D1.** Design a feature pipeline serving a fraud model with p99 under 50 ms.

*Solution outline.* Stream events through Kafka into a stream processor computing windowed aggregates; write them to a low-latency online store (Redis or a key-value store) keyed by entity; write the same definitions to an offline store for training to avoid skew; serve the model behind an API that reads features in one batched lookup; log served features and predictions for monitoring.

**D2.** A nightly Spark job went from 40 to 180 minutes with the same input size. Where do you look?

*Solution outline.* The Spark UI stage view: a straggler task points to skew in a new key value; growing shuffle bytes point to a lost broadcast or a changed join; many small files point to an upstream writer change; executor GC time points to memory pressure.
