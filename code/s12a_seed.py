import duckdb, numpy as np, pandas as pd, os
rng = np.random.default_rng(7)
if os.path.exists("data/shop.duckdb"): os.remove("data/shop.duckdb")
con = duckdb.connect("data/shop.duckdb")
cust = pd.DataFrame({"id": range(1, 201), "name": [f"cust_{i}" for i in range(1, 201)],
                     "region": rng.choice(["Nairobi", "Mombasa", "Kisumu", "Eldoret"], 200), "signup": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, 200), "D")})
o = pd.DataFrame({"order_id": range(1, 3001), "customer_id": rng.integers(1, 201, 3000),
                  "amount": rng.gamma(2, 1500, 3000).round(2), "status": rng.choice(["paid", "refund", "new"], 3000, p=[.85, .05, .10]),
                  "created_at": pd.to_datetime("2025-06-01") + pd.to_timedelta(rng.integers(0, 365 * 24 * 60, 3000), "min")})
logins = pd.DataFrame({"user_id": rng.integers(1, 11, 400), "d": (pd.to_datetime("2026-01-01") + pd.to_timedelta(rng.integers(0, 60, 400), "D")).date}).drop_duplicates()
emp = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "name": ["Amina", "Brian", "Chao", "Dalia", "Eli", "Fatma"], "manager_id": [None, 1, 1, 2, 2, 4], "salary": [9000, 7000, 7200, 5000, 5100, 4000]})
for n, d in [("customers", cust), ("orders", o), ("logins", logins), ("employees", emp)]: con.execute(f"CREATE TABLE {n} AS SELECT * FROM d")
print({t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ["customers", "orders", "logins", "employees"]})
