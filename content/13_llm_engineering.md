# LLM Engineering

tagline: Building with large language models: tokenization, context windows, sampling, the KV-cache, quantization, LoRA fine-tuning, RAG and how to evaluate it, agents, guardrails, and the cost arithmetic.

## The LLM stack at a glance

| Layer | Decisions | Typical choices |
|---|---|---|
| Model | size, open vs API, context length, licence | GPT/Claude/Gemini APIs; Llama, Mistral, Qwen, Gemma (open) |
| Adaptation | prompt → RAG → fine-tune → pretrain | escalate only when the cheaper step fails |
| Serving | latency, throughput, cost | vLLM, TGI, TensorRT-LLM, llama.cpp, Ollama |
| Retrieval | chunking, embeddings, index, reranking | pgvector, Qdrant, Weaviate, FAISS; BM25 hybrid |
| Orchestration | tools, agents, memory, workflows | function calling, LangGraph, LlamaIndex, plain code |
| Evaluation | golden sets, LLM-as-judge, online metrics | Ragas, DeepEval, promptfoo, human review |
| Safety | input/output filters, PII, permissions | guardrail models, allow-lists, sandboxing |
| Observability | traces, token usage, feedback | Langfuse, LangSmith, OpenTelemetry |

!!! rule "Rule of thumb: the adaptation ladder"
    Prompt engineering (hours) → few-shot examples → RAG for **knowledge** (days) → fine-tuning for **behaviour, format, style or latency** (weeks) → continued pretraining for a new domain/language (months). Fine-tuning is a poor way to add facts; RAG is a poor way to change behaviour.

## Tokenization

![](figures/tokens.svg)

| Algorithm | Idea | Used by |
|---|---|---|
| BPE | repeatedly merge the most frequent adjacent pair | GPT family (tiktoken), LLaMA-3 |
| Byte-level BPE | BPE over bytes: no unknown tokens | GPT-2+, RoBERTa |
| WordPiece | merge by likelihood gain | BERT |
| Unigram (SentencePiece) | prune a big vocab by likelihood | T5, LLaMA-2, multilingual models |

!!! example "Worked example: BPE merge"
    Corpus "low low lower lowest". Start from characters. The most frequent pair is (l, o), then (lo, w): "low" becomes one token, while "lower" = low + e + r until (e, r) merges. Rare words split into frequent pieces, so there are no out-of-vocabulary words, and the vocabulary (32k–256k) trades sequence length for embedding-table size.

!!! pitfall "Pitfalls caused by tokens"
    English ≈ 0.75 words/token, but many languages (Swahili, Amharic, Arabic) need 1.5–3× more tokens for the same text: more cost, less effective context. Numbers split unpredictably (arithmetic errors); leading spaces create different tokens; character-level tasks ("count the r's") are hard because the model never sees letters.

## Context windows, sampling and decoding

![](figures/temperature.svg)

$$p_i = \frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}$$

**Intuition:** temperature rescales logits. $T\to0$ is greedy (deterministic), $T = 1$ samples the model's distribution, $T > 1$ flattens it. **Example:** logits (2, 1, 0): at $T = 1$ the probabilities are (0.67, 0.24, 0.09); at $T = 0.5$ they are (0.87, 0.12, 0.02).

| Knob | Effect | Default |
|---|---|---|
| temperature | randomness | 0–0.3 extraction/code, 0.7–1.0 creative |
| top-p (nucleus) | sample from smallest set with cumulative prob ≥ p | 0.9–0.95 |
| top-k | sample from k most likely | 40–50 |
| max tokens | output budget | set it; controls cost and latency |
| stop sequences | end generation | for structured outputs |
| structured output | JSON schema / grammar-constrained decoding | always for machine-read outputs |

!!! pitfall "Pitfall: long context is not free memory"
    Models attend less reliably to the middle of long contexts ("lost in the middle"); cost and latency grow with every token; the KV-cache grows linearly. Put instructions at the start and the question at the end, retrieve less but better, and measure answer quality as context grows.

## The KV-cache and inference arithmetic

![](figures/kv_cache.svg)

$$\text{KV bytes} = 2\times L\times h_{kv}\times d_{head}\times n_{tokens}\times\text{batch}\times\text{bytes per value}$$

**Intuition:** during generation each new token attends to all previous ones; caching their keys and values avoids recomputing them, turning $O(n^2)$ recomputation into $O(n)$ per step, but the cache can outgrow the weights.

@@code code/s10_llm_math.py@@
@@out code/out/s10_llm_math.txt@@

| Technique | What it saves | Idea |
|---|---|---|
| GQA / MQA | KV memory (4–8×) | many query heads share few K/V heads |
| PagedAttention (vLLM) | fragmentation | cache in fixed blocks like virtual memory |
| Continuous batching | idle GPU | add/remove requests per decoding step |
| Prefix caching | repeated prompts | reuse KV of a shared system prompt |
| Speculative decoding | latency (2–3×) | small draft model proposes, big model verifies in one pass |
| FlashAttention | memory bandwidth | tiled, fused attention kernel |
| KV quantization | KV memory | int8/fp8 cache |

!!! rule "Rule of thumb: prefill vs decode"
    Prefill (reading the prompt) is compute-bound and parallel; decode (one token at a time) is **memory-bandwidth bound**: each token reads all weights. Rough single-stream decode speed ≈ memory bandwidth / model bytes; e.g. 1 TB/s ÷ 16 GB ≈ 60 tokens/s for an 8B fp16 model. Quantising to 4-bit roughly quadruples it.

## Quantization

$$q = \text{clip}\Big(\text{round}\big(w/s\big),\,-2^{b-1},\,2^{b-1}-1\Big),\qquad \hat w = s\,q,\qquad s = \frac{\max\lvert w\rvert}{2^{b-1}-1}$$

**Example:** int8, group max $\lvert w\rvert = 0.05$: $s = 0.05/127 = 3.9\times10^{-4}$. Weight 0.0123 → $q = \text{round}(31.2) = 31$ → $\hat w = 0.01220$ (error 0.8%).

@@code code/s10_quant.py@@
@@out code/out/s10_quant.txt@@

| Method | Bits | Type | Note |
|---|---|---|---|
| bf16 / fp16 | 16 | baseline | bf16 has fp32's range; preferred for training |
| int8 (LLM.int8, SmoothQuant) | 8 | weights (+activations) | near-lossless |
| fp8 (E4M3) | 8 | weights + activations | H100-class GPUs |
| GPTQ / AWQ | 4 | post-training, weight-only | calibration data; AWQ protects salient channels |
| GGUF (llama.cpp) Q4_K_M etc. | 2–8 | CPU/Mac friendly | k-quants, group-wise |
| NF4 (QLoRA) | 4 | normal-float codebook | fine-tune 4-bit base + LoRA adapters |

!!! pitfall "Pitfall: outliers break naive quantization"
    LLM activations have a few huge-magnitude channels; one absmax scale for a whole tensor wastes resolution on them. Use per-channel or per-group (64–128) scales, outlier handling (LLM.int8, AWQ), and always evaluate the quantized model on your task: 4-bit usually costs a little perplexity, 3-bit often hurts.

## Fine-tuning and LoRA

![](figures/lora.svg)

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r}BA,\qquad B\in\mathbb R^{d\times r},\ A\in\mathbb R^{r\times k},\ r\ll\min(d,k)$$

**Intuition:** the update needed to adapt a pretrained model has low intrinsic rank, so learn it as a thin product. $B$ starts at zero so training starts exactly from the pretrained model. **Example:** $d = k = 4096$, $r = 16$: $16\cdot8192 = 131{,}072$ trainable parameters instead of 16.8M (0.78%). Adapters can be merged into $W_0$ for zero inference overhead, or hot-swapped per customer.

@@code code/s10_lora.py@@
@@out code/out/s10_lora.txt@@

| Method | Trains | Memory | When |
|---|---|---|---|
| Full fine-tune | all weights | ~16 bytes/param (Adam, mixed precision) | big shift, lots of data, big budget |
| LoRA | low-rank adapters on attention/MLP | weights + tiny optimiser state | default PEFT |
| QLoRA | LoRA on a 4-bit frozen base | 7B fits on one 24 GB GPU | limited hardware |
| Prefix / prompt tuning | soft prompt vectors | minimal | many tasks, small changes |
| SFT | instruction → response pairs | | teach format/behaviour |
| DPO / ORPO / KTO | preference pairs (chosen vs rejected) | | align style and preferences without an RL loop |
| RLHF (PPO) | reward model + RL | heavy | frontier alignment |
| Distillation | small student mimics a large teacher | | cost/latency reduction |

!!! tip "Pro tip: LoRA defaults that work"
    `r` 8–16 (64 for harder tasks), `alpha` = 2r, dropout 0.05, target all linear layers (q, k, v, o, gate, up, down), LR 1e-4–2e-4 (10× higher than full fine-tuning), 1–3 epochs. 500–5,000 high-quality examples beat 100k noisy ones. Hold out an eval set and watch for regressions on general ability.

## Retrieval-augmented generation (RAG)

![](figures/rag.svg)

| Stage | Decision | Default / tip |
|---|---|---|
| Parse | PDFs, tables, HTML | keep structure (headings, tables as markdown) |
| Chunk | size and overlap | 300–800 tokens, 10–20% overlap, split on structure; store parent doc ids |
| Embed | model, dimension | a strong open or API embedding model; same model for queries and docs; normalise |
| Index | ANN structure | HNSW (pgvector, Qdrant); add metadata filters |
| Retrieve | dense, sparse or hybrid | **hybrid** BM25 + dense with reciprocal rank fusion |
| Rerank | cross-encoder | retrieve 20–50, rerank to 3–8 |
| Generate | prompt with citations | "answer only from context; say if missing" |
| Evaluate | retrieval + generation metrics | golden question set from real traffic |

@@code code/s10_rag.py@@
@@out code/out/s10_rag.txt@@

!!! pitfall "Pitfall: visible in the run above"
    Query "how long do refunds take" missed d4 ("...a full refund...") because pure lexical TF-IDF treats "refunds" and "refund" as different terms, and the second slot was filled with an irrelevant document (score 0.0). Fixes: stemming/BM25, dense embeddings, hybrid retrieval, and a minimum-score threshold so irrelevant chunks never reach the prompt.

### Evaluating RAG

![](figures/rag_triad.svg)

| Metric | Measures | How |
|---|---|---|
| Recall@k / hit rate | did the right chunk get retrieved? | labelled (question, relevant chunk ids) |
| MRR / NDCG | is it ranked high? | same labels |
| Context precision | share of retrieved context that is relevant | LLM judge or labels |
| Faithfulness / groundedness | every claim supported by the context | claim extraction + entailment (LLM judge) |
| Answer relevance | answers the question asked | LLM judge, embedding similarity |
| Answer correctness | matches reference answer | exact/F1 for short answers, judge for long |
| Citation accuracy | cited source contains the claim | automatic check |
| Online | thumbs up/down, escalation rate, deflection | product analytics |

!!! interview "Interview: the RAG system hallucinates. How do you debug it?"
    Split the problem: 1) Was the answer in the corpus? 2) Was it retrieved (recall@k on a golden set)? 3) Did it survive reranking and fit in context? 4) Did the model use it (faithfulness)? Fix the first failing stage: parsing/chunking, hybrid retrieval, reranking, or prompting/constrained answers with citations and abstention.

## Agents and tool use

![](figures/agent.svg)

| Pattern | Description | Use |
|---|---|---|
| Function calling | model emits JSON matching a tool schema; your code executes | APIs, DB queries |
| ReAct | think → act → observe loop | open-ended tasks |
| Plan-and-execute | plan first, then run steps | long tasks, fewer LLM calls |
| Router | classify the request → specialised prompt/model | cost and quality |
| Workflow (DAG) | fixed steps with LLM calls inside | most production "agents" should be this |
| Multi-agent | specialised agents hand off | complex, but harder to debug |
| Reflection / critic | second pass checks the first | quality-critical outputs |

!!! rule "Rule of thumb: agents"
    Prefer deterministic workflows; add autonomy only where the path genuinely varies. Give tools narrow, well-described schemas; validate arguments; set step and cost budgets; log full traces; require human confirmation for irreversible actions (payments, emails, deletes).

## Guardrails and safety

| Risk | Example | Mitigation |
|---|---|---|
| Prompt injection | a retrieved web page says "ignore previous instructions" | treat retrieved/tool content as data; privilege separation; allow-listed tools; output checks |
| Jailbreaks | role-play to bypass policy | safety classifiers (input + output), refusal tuning |
| Data leakage / PII | model echoes a customer's card number | PII detection and redaction, access control at retrieval time |
| Hallucination | invented citations or numbers | grounding, citations, abstention, verification step |
| Over-permissioned agents | agent deletes production data | least privilege, sandbox, human approval |
| Toxicity / bias | harmful content | moderation models, evals per demographic |
| Cost/DoS | huge prompts, loops | rate limits, token budgets, max steps |

## Evaluating LLM applications

| Level | Method |
|---|---|
| Unit | assertions on format (valid JSON, schema, regex), deterministic checks |
| Golden set | 100–500 real questions with references; run on every change (CI) |
| LLM-as-judge | rubric-based grading with a strong model; calibrate against human labels; randomise answer order |
| Pairwise | A/B preference between versions, Elo/Bradley–Terry |
| Online | user feedback, task success, latency p95, cost per task |

!!! pitfall "Pitfall: judge bias"
    LLM judges prefer longer answers, the first position, and their own model family's style. Use clear rubrics, swap positions, use reference answers, and spot-check agreement with humans (Cohen's κ).

<div class="card" markdown="1">

## Cheat card: LLM engineering

<div class="cols" markdown="1">

#### Decide
- Knowledge → RAG; behaviour/format → fine-tune (LoRA)
- Start with the best API model to prove value, then optimise cost
- Workflow first, agent only when needed

#### Numbers to know
- ~0.75 English words per token
- Weights: params × bytes (8B fp16 = 16 GB, 4-bit ≈ 4–5 GB)
- KV: $2 L h_{kv} d_{head}$ × tokens × bytes
- Decode tok/s ≈ bandwidth / model bytes
- Train FLOPs ≈ 6 N D

#### Sampling
- Extraction/code: T 0–0.2; creative: 0.7–1.0, top-p 0.9
- Structured output with JSON schema

#### LoRA
- r 8–16, alpha 2r, LR 1e-4–2e-4
- All linear layers; QLoRA when GPU-poor

#### RAG
- Chunks 300–800 tokens, overlap 10–20%
- Hybrid BM25 + dense, rerank top 20–50
- Eval: recall@k, faithfulness, relevance, correctness

#### Safety
- Untrusted content is data, never instructions
- Least-privilege tools; human approval for irreversible actions
- PII redaction; budgets for tokens and steps

</div>
</div>
