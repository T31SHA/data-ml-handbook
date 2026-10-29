import pandas as pd, numpy as np, os
p = "data/housing_raw.csv"
raw = pd.read_csv(p)
typed = pd.read_csv(p, dtype={"region": "category", "HouseAge": "float32", "Population": "float32"},
                    engine="pyarrow")
mb = lambda d: d.memory_usage(deep=True).sum() / 1e6
print(f"default dtypes: {mb(raw):.2f} MB | tuned dtypes: {mb(typed):.2f} MB")
total = sum(chunk.loc[chunk.MedHouseVal > 3, "Population"].sum() for chunk in pd.read_csv(p, chunksize=5_000))
print(f"chunked aggregate over {os.path.getsize(p)/1e6:.1f} MB file: {total:,.0f}")
raw.to_parquet("data/housing.parquet", index=False)
print(f"CSV {os.path.getsize(p)/1e6:.2f} MB -> Parquet {os.path.getsize('data/housing.parquet')/1e6:.2f} MB")
print(pd.read_parquet("data/housing.parquet", columns=["MedInc", "region"]).dtypes.to_dict())
