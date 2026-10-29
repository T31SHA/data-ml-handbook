# HIDE
import sys; sys.path.insert(0, "../figures/src"); from figstyle import *
# ENDHIDE
import pandas as pd, numpy as np, matplotlib.pyplot as plt
df = pd.read_parquet("data/housing.parquet").dropna()
df["income_band"] = pd.qcut(df.MedInc, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)               # small multiples, shared scale
for ax, (band, g) in zip(axes, df.groupby("income_band", observed=True)):
    ax.hexbin(g.HouseAge, g.MedHouseVal, gridsize=18, cmap="viridis", mincnt=1)
    ax.set_title(f"income {band} (n={len(g):,})"); ax.set_xlabel("house age")
axes[0].set_ylabel("median value ($100k)")
best = df.groupby("income_band", observed=True).MedHouseVal.median()
axes[-1].annotate(f"median {best.iloc[-1]:.2f}", xy=(45, best.iloc[-1]), xytext=(5, 4.6),
                  arrowprops=dict(arrowstyle="->", color="#DB2777"), color="#DB2777", weight="bold")
fig.suptitle("Value rises with income in every age bracket", weight="bold")
fig.tight_layout(); fig.savefig(O + "viz_demo.svg", bbox_inches="tight")
print(best.round(2).to_dict())
