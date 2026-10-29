import polars as pl, duckdb, numpy as np
n = 1_000_000
df = pl.DataFrame({"city": np.random.default_rng(0).choice(["A", "B", "C"], n),
                   "sales": np.random.default_rng(1).exponential(100, n)})
df.write_parquet("data/sales.parquet")
lazy = (pl.scan_parquet("data/sales.parquet")              # lazy: nothing read yet
          .filter(pl.col("sales") > 50)
          .group_by("city").agg(pl.len().alias("n"), pl.col("sales").mean().round(2).alias("avg"))
          .sort("city"))
print(lazy.collect())                                       # optimiser pushes the filter into the scan
print(duckdb.sql("SELECT city, count(*) n, round(avg(sales),2) avg FROM 'data/sales.parquet' "
                 "WHERE sales > 50 GROUP BY 1 ORDER BY 1").fetchall())
