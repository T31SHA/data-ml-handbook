import pandas as pd, numpy as np
rng = np.random.default_rng(0)
df = pd.DataFrame({"user": rng.integers(1, 5, 12), "city": rng.choice(["Nairobi", "Mombasa"], 12),
                   "amount": rng.gamma(2, 50, 12).round(1),
                   "ts": pd.date_range("2026-01-01", periods=12, freq="5D")})
out = (df.assign(month=df.ts.dt.to_period("M"))
         .groupby(["city", "month"], observed=True)
         .agg(n=("amount", "size"), revenue=("amount", "sum"), p90=("amount", lambda s: s.quantile(.9)))
         .round(1))
print(out.head(4))
df["user_avg"] = df.groupby("user")["amount"].transform("mean")          # broadcast group stat to rows
df["rank_in_city"] = df.groupby("city")["amount"].rank(ascending=False)   # window-function equivalent
df["prev_amount"] = df.sort_values("ts").groupby("user")["amount"].shift(1)  # LAG
print(df[["user", "city", "amount", "user_avg", "rank_in_city", "prev_amount"]].head(4).round(1).to_string())
