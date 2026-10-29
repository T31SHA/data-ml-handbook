# HIDE
import warnings; warnings.filterwarnings("ignore")
# ENDHIDE
# Build a realistic "raw" CSV from the California Housing census data (20,640 block groups).
# We add the kinds of defects real exports have, so the cleaning steps are meaningful.
import numpy as np, pandas as pd
from sklearn.datasets import fetch_california_housing
rng = np.random.default_rng(42)
df = fetch_california_housing(as_frame=True).frame            # 8 features + MedHouseVal (in $100k)
df["region"] = np.select([df.Latitude > 37.2, df.Latitude > 35.5, df.Longitude > -118.3],
                         ["NorCal", "Central", "SoCal-Inland"], default="SoCal-Coast")
df.loc[rng.random(len(df)) < 0.03, "MedInc"] = np.nan          # 3% missing income
df.loc[rng.random(len(df)) < 0.01, "HouseAge"] = np.nan        # 1% missing age
df["region"] = df["region"].where(rng.random(len(df)) > 0.02, df["region"].str.lower() + " ")  # messy labels
df = pd.concat([df, df.sample(150, random_state=1)])           # 150 duplicate rows
df.to_csv("data/housing_raw.csv", index=False)
print(df.shape); print(df.head(3).round(2).to_string())
