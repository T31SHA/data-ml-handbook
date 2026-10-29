import numpy as np, hashlib
from collections import Counter
rng = np.random.default_rng(6)
keys = rng.zipf(1.3, 200_000) % 5000                  # heavy-tailed key popularity
def h(k, n): return int(hashlib.md5(str(k).encode()).hexdigest(), 16) % n
P = 16
loads = Counter(h(k, P) for k in keys)
mx = max(loads.values()); mean = len(keys) / P
print(f"hash partitioning: max/mean load = {mx/mean:.2f}  (job time is set by the slowest partition)")
# salting: split the hottest keys across S sub-keys, aggregate twice
hot = {k for k, c in Counter(keys).most_common(5)}
S = 8
loads2 = Counter(h((k, rng.integers(S)) if k in hot else k, P) for k in keys)
print(f"salted hot keys  : max/mean load = {max(loads2.values())/mean:.2f}")
# consistent hashing: fraction of keys that move when adding a node
def ring(nodes, v=100):
    return sorted((h(f"{n}#{i}", 2**32), n) for n in nodes for i in range(v))
def owner(r, k):
    x = h(k, 2**32)
    for pos, n in r:
        if pos >= x: return n
    return r[0][1]
ks = range(20000)
r1, r2 = ring(range(10)), ring(range(11))
moved = sum(owner(r1, k) != owner(r2, k) for k in ks) / len(ks)
naive = sum(h(k, 10) != h(k, 11) for k in ks) / len(ks)
print(f"keys moved adding 11th node: consistent hashing={moved:.1%}  mod-N={naive:.1%}")
