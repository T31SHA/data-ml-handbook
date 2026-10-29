import pandas as pd, pyarrow as pa, pyarrow.parquet as pq, shutil, os, glob
root = "data/lake/orders"; shutil.rmtree(root, ignore_errors=True)
def load_partition(ds: str, df: pd.DataFrame):
    """Idempotent: overwrite exactly one date partition; re-runs/backfills produce the same result."""
    pq.write_to_dataset(pa.Table.from_pandas(df.assign(dt=ds)), root, partition_cols=["dt"],
                        existing_data_behavior="delete_matching")
day = pd.DataFrame({"order_id": [1, 2, 3], "amount": [10.0, 20.0, 5.0]})
load_partition("2026-10-23", day); load_partition("2026-10-23", day); load_partition("2026-10-24", day.head(1))
t = pq.read_table(root).to_pandas()
print(t.groupby("dt", observed=True).size().to_string(), "\nfiles:", sorted(os.path.relpath(p, root) for p in glob.glob(root + "/*/*")))
