import numpy as np
rng = np.random.default_rng(0); n = 10_000
g = rng.choice(["A", "B"], n, p=[.7, .3]); y = rng.random(n) < np.where(g == "A", .30, .20)
score = np.clip(y * .35 + rng.normal(.35, .18, n) + (g == "A") * .05, 0, 1); yhat = score > .5
for grp in ["A", "B"]:
    m = g == grp
    print(f"group {grp}: selection rate={yhat[m].mean():.3f}  TPR={yhat[m & y].mean():.3f}  FPR={yhat[m & ~y].mean():.3f}  precision={y[m & yhat].mean():.3f}")
sr = [yhat[g == k].mean() for k in "AB"]; print(f"disparate impact ratio = {min(sr)/max(sr):.2f}  (< 0.8 fails the 80% rule)")
