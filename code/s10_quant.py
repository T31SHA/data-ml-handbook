import numpy as np
w = np.random.default_rng(0).normal(0, 0.02, 4096).astype(np.float32)
def quant(w, bits, group=128):
    qmax = 2 ** (bits - 1) - 1; out = np.empty_like(w)
    for g in range(0, len(w), group):                        # per-group symmetric absmax scaling
        blk = w[g:g + group]; s = np.abs(blk).max() / qmax
        out[g:g + group] = np.clip(np.round(blk / s), -qmax - 1, qmax) * s
    return out
for b in [8, 4, 3]:
    e = quant(w, b) - w; print(f"int{b}: relative RMS error = {np.sqrt((e**2).mean())/w.std():.4f}")
