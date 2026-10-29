# Recommender Systems in Depth

tagline: Matrix factorisation, implicit feedback, two-tower retrieval, ranking metrics and the production funnel.

## The production funnel

1. **Candidate generation** (millions to hundreds): two-tower embeddings with approximate nearest neighbour search, co-visitation, popularity.
2. **Ranking** (hundreds to tens): a feature-rich model (GBM or deep network) predicting engagement.
3. **Re-ranking**: diversity, freshness, business rules, fairness constraints.

## Matrix factorisation

Approximate the rating matrix by user and item factors, $\hat r_{ui}=\mathbf p_u^\top\mathbf q_i$, by minimising over observed pairs:

$$\min_{P,Q}\sum_{(u,i)\in\Omega}(r_{ui}-\mathbf p_u^\top\mathbf q_i)^2+\lambda\left(\lVert\mathbf p_u\rVert^2+\lVert\mathbf q_i\rVert^2\right).$$

!!! derivation "Derivation: the ALS update"
    With $Q$ fixed the problem is a ridge regression for each user: $\mathbf p_u=(Q_u^\top Q_u+\lambda I)^{-1}Q_u^\top\mathbf r_u$, where $Q_u$ holds the factors of items user $u$ rated. Alternating between users and items decreases the objective at every step, and each step parallelises across rows.

@@code code/n06_mf_als.py@@
@@out code/out/n06_mf_als.txt@@

## Implicit feedback

Clicks and plays are not ratings: a missing entry is not a negative. Weighted MF uses preference $p_{ui}=\mathbb 1[r_{ui}>0]$ with confidence $c_{ui}=1+\alpha r_{ui}$ over *all* pairs; BPR optimises pairwise ranking, $\sum\log\sigma(\hat x_{ui}-\hat x_{uj})$ for an observed item $i$ and a sampled unobserved item $j$.

## Two-tower retrieval

Encode users and items separately, $s(u,i)=f(u)^\top g(i)$, and train with in-batch sampled softmax. Item embeddings are pre-computed and indexed (HNSW, IVF-PQ) so retrieval is a nearest-neighbour lookup.

!!! pitfall "Pitfall: popularity bias in in-batch negatives"
    Popular items appear as negatives more often, so the model under-scores them. Apply the log-Q correction: subtract $\log Q(i)$, the item's sampling probability, from the logit.

## Ranking metrics

$$\text{DCG@}k=\sum_{j=1}^{k}\frac{2^{rel_j}-1}{\log_2(j+1)},\qquad \text{nDCG@}k=\frac{\text{DCG@}k}{\text{IDCG@}k}.$$

| Metric | Measures | Note |
|---|---|---|
| Recall@k | share of relevant items retrieved | candidate-stage metric |
| nDCG@k | graded relevance with position discount | ranking-stage metric |
| MAP | precision averaged over relevant positions | binary relevance |
| Coverage, diversity | catalogue breadth | guards against filter bubbles |

!!! rule "Rule of thumb"
    Offline metrics rank candidate models; only an online experiment measures impact. Split evaluation data by time, never randomly, to avoid training on the future.

## Cheat card: Recommender systems

- Funnel: retrieval, ranking, re-ranking.
- ALS for explicit ratings; weighted MF or BPR for implicit data.
- Two-tower + ANN for retrieval; correct sampling bias.
- Recall@k for retrieval, nDCG@k for ranking, plus coverage.
- Time-based splits; cold start via content features and popularity priors.
