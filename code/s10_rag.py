import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
docs = {"d1": "Refunds are issued within 14 days of purchase if the item is unused.",
        "d2": "Shipping to Nairobi takes 3-5 business days; express is next day.",
        "d3": "Our API rate limit is 100 requests per minute per key.",
        "d4": "Unused items can be returned for a full refund; opened items get store credit.",
        "d5": "Passwords must be rotated every 90 days for admin accounts."}
ids, texts = list(docs), list(docs.values())
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(texts); E = vec.transform(texts)
def retrieve(q, k=2):
    s = (E @ vec.transform([q]).T).toarray().ravel(); top = np.argsort(-s)[:k]; return [(ids[i], round(float(s[i]), 3)) for i in top]
gold = {"how long do refunds take": {"d1", "d4"}, "delivery time to nairobi": {"d2"}, "api limits": {"d3"}}
recall, rr = [], []
for q, rel in gold.items():
    got = [d for d, _ in retrieve(q, 2)]; recall.append(len(rel & set(got)) / len(rel))
    rr.append(next((1 / (i + 1) for i, d in enumerate(got) if d in rel), 0)); print(f"{q!r:30s} -> {retrieve(q, 2)}")
print(f"recall@2={np.mean(recall):.2f}  MRR={np.mean(rr):.2f}")
ctx = "\n".join(f"[{d}] {docs[d]}" for d, _ in retrieve("how long do refunds take"))
prompt = f"Answer ONLY from the context and cite [ids]. If missing, say so.\n<context>\n{ctx}\n</context>\nQ: how long do refunds take?"
print(prompt)
