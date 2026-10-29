import math, re
from collections import Counter
docs = {
 "d1": "LoRA freezes the base weights and trains low rank adapter matrices A and B",
 "d2": "The KV cache stores keys and values so decoding avoids recomputing attention",
 "d3": "Quantization maps weights to int8 or int4 to cut memory and latency",
 "d4": "Retrieval augmented generation grounds answers in retrieved passages",
 "d5": "Direct preference optimization fine tunes a model on preference pairs without a reward model",
}
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
D = {k: tok(v) for k, v in docs.items()}; N = len(D); avg = sum(map(len, D.values())) / N
df = Counter(w for d in D.values() for w in set(d))
def bm25(q, d, k1=1.5, b=0.75):
    tf = Counter(d); s = 0
    for w in tok(q):
        if w not in tf: continue
        idf = math.log(1 + (N - df[w] + .5) / (df[w] + .5))
        s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
    return s
qrels = {"how does lora fine tuning work": {"d1"}, "why is decoding fast with cache": {"d2"},
         "preference tuning without reward model": {"d5"}, "reduce memory of weights": {"d3"}}
mrr, hits = 0, 0
for q, rel in qrels.items():
    ranked = sorted(D, key=lambda k: -bm25(q, D[k]))
    r = next(i for i, k in enumerate(ranked, 1) if k in rel)
    mrr += 1 / r; hits += r <= 3
    print(f"{q:42s} top={ranked[:3]} rank_of_relevant={r}")
print(f"MRR={mrr/len(qrels):.3f}  Recall@3={hits/len(qrels):.2f}")
