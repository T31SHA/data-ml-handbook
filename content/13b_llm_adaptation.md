# LLM Fine-Tuning, Evaluation and Retrieval in Depth

tagline: When to prompt, retrieve or train; preference optimisation; retrieval metrics; and evaluation that survives production.

## Choosing an adaptation strategy

| Need | First choice | Escalate to |
|---|---|---|
| New facts that change often | RAG over a maintained index | hybrid retrieval + re-ranking |
| Output format or style | prompting with examples, structured output | supervised fine-tuning (LoRA) |
| Domain vocabulary and reasoning patterns | SFT on curated in-domain data | continued pre-training |
| Human preferences (tone, helpfulness, refusals) | preference optimisation (DPO) | RLHF with a reward model |
| Latency or cost | smaller model + distillation | quantisation, speculative decoding |

!!! rule "Rule of thumb"
    Fine-tuning teaches *behaviour*; retrieval supplies *knowledge*. Fine-tuning to inject facts is expensive, hard to update and prone to hallucinating near-miss facts.

## Supervised fine-tuning

SFT minimises next-token cross-entropy on the response tokens only:

$$\mathcal L_{\text{SFT}}=-\sum_{t\in\text{response}}\log p_\theta(y_t\mid x,y_{<t}).$$

Mask the prompt tokens from the loss, pack sequences to reduce padding, and hold out an evaluation set drawn from the *target* distribution, not the training mix.

## Preference optimisation

RLHF fits a reward model $r_\phi$ from pairwise preferences with the Bradley-Terry likelihood $P(y_w\succ y_l)=\sigma(r(x,y_w)-r(x,y_l))$, then maximises reward with a KL penalty to the reference policy. Direct Preference Optimisation removes the explicit reward model:

$$\mathcal L_{\text{DPO}}=-\mathbb E\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right].$$

!!! intuition "Intuition"
    DPO raises the likelihood of the preferred answer relative to the reference model and lowers that of the rejected answer. $\beta$ controls how far the policy may drift from the reference.

!!! pitfall "Pitfall: catastrophic forgetting and over-optimisation"
    Aggressive fine-tuning degrades general ability, and optimising against a fixed reward model or judge exploits its blind spots. Track a general-capability benchmark alongside the target metric and stop when it drops.

## Retrieval in depth

A production retriever is usually **hybrid**: sparse lexical scoring (BM25) catches exact terms, IDs and rare words; dense embeddings catch paraphrases. Fuse ranked lists with reciprocal rank fusion:

$$\text{RRF}(d)=\sum_{r\in\text{rankers}}\frac{1}{k+\text{rank}_r(d)},\qquad k\approx 60.$$

BM25 for a query $q$ and document $d$:

$$\text{BM25}(q,d)=\sum_{w\in q}\text{IDF}(w)\,\frac{f(w,d)\,(k_1+1)}{f(w,d)+k_1\left(1-b+b\,\frac{|d|}{\text{avgdl}}\right)}.$$

@@code code/n07_rag_eval.py@@
@@out code/out/n07_rag_eval.txt@@

## Evaluating a RAG system

Evaluate the stages separately, then end to end.

| Stage | Metric | Needs |
|---|---|---|
| Retrieval | Recall@k, MRR, nDCG@k | labelled query-to-passage relevance |
| Grounding | faithfulness: share of answer claims supported by retrieved context | claim extraction + entailment check |
| Answer | correctness vs reference, task success | references or rubric |
| System | latency p95, cost per query, refusal rate | production logs |

!!! tip "Pro tip"
    When using an LLM as a judge, calibrate it against a few hundred human labels, randomise answer order to counter position bias, and report judge-human agreement (Cohen's $\kappa$) with every score.

## Cheat card: LLM adaptation

- Retrieval for knowledge, fine-tuning for behaviour.
- SFT: loss on response tokens only; evaluate on the target distribution.
- DPO is the simple default for preference data; watch KL drift and general benchmarks.
- Hybrid retrieval with RRF, then a cross-encoder re-ranker on the top 50-100.
- Measure retrieval, grounding and answer quality separately.
- LLM judges need calibration against humans.
