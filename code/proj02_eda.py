# HIDE
import warnings; warnings.filterwarnings("ignore"); import sys; sys.path.insert(0, "../figures/src")
from figstyle import *
# ENDHIDE
import pandas as pd, numpy as np
df = pd.read_csv("data/housing_raw.csv")
print("shape:", df.shape, "| duplicates:", df.duplicated().sum())
print("missing %:\n", (df.isna().mean() * 100).round(2)[lambda s: s > 0].to_string())
print("region labels:", sorted(df.region.unique()))
print(df.describe().T[["mean", "std", "min", "50%", "max"]].round(2).to_string())
print("target capped at 5.0 (= $500k):", (df.MedHouseVal >= 5.0).mean().round(3))
print("corr with target:\n", df.corr(numeric_only=True)["MedHouseVal"].drop("MedHouseVal").sort_values().round(3).to_string())
# HIDE
f, axs = plt.subplots(1, 3, figsize=(12, 3.6))
sc = axs[0].scatter(df.Longitude, df.Latitude, c=df.MedHouseVal, s=2, cmap="viridis"); axs[0].set_title("Geography: price by location"); f.colorbar(sc, ax=axs[0], label="MedHouseVal ($100k)")
axs[0].set_xlabel("longitude"); axs[0].set_ylabel("latitude")
axs[1].hist(df.MedHouseVal, bins=60, color=T); axs[1].axvline(5, color=R, ls="--"); axs[1].text(4.1, 900, "cap at 5.0\n(censored!)", color=R, fontsize=8, weight="bold"); axs[1].set_title("Target distribution")
axs[2].scatter(df.MedInc, df.MedHouseVal, s=2, alpha=.25, color=I); axs[2].set_title("Strongest driver: median income"); axs[2].set_xlabel("MedInc ($10k)"); axs[2].set_ylabel("MedHouseVal")
save(f, "proj_eda")
# ENDHIDE
