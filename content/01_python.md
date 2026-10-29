# Python, NumPy, pandas, Polars & DuckDB

tagline: The working toolkit: idiomatic Python, vectorised NumPy, expressive pandas, fast Polars and SQL-on-files with DuckDB.

## Python essentials for data work

| Concept | Idiom | Why it matters |
|---|---|---|
| Comprehensions | `[f(x) for x in xs if p(x)]`, `{k: v for k, v in items}` | clearer and faster than append loops |
| Generators | `(x for x in big)`, `yield` | constant memory streaming over large files |
| Unpacking | `a, *rest = seq`, `f(*args, **kwargs)` | flexible function APIs |
| f-strings | `f"{x:.3f} {n:,} {p:.1%} {d:%Y-%m-%d}"` | readable formatting |
| Dataclasses | `@dataclass(frozen=True) class Cfg: lr: float = 1e-3` | typed, immutable configs |
| Type hints | `def f(x: pd.DataFrame) -> np.ndarray:` | catch bugs early with `mypy`/`pyright` |
| Context managers | `with open(p) as f:` / `contextlib.contextmanager` | guaranteed cleanup (files, DB, timers) |
| `pathlib` | `Path("data").glob("*.parquet")` | OS-independent paths |
| `functools` | `lru_cache`, `partial`, `reduce` | memoisation and composition |
| `collections` | `Counter`, `defaultdict`, `deque`, `namedtuple` | the right container for the job |
| `itertools` | `groupby`, `chain`, `product`, `combinations`, `islice` | lazy combinatorics |
| Logging | `logging.getLogger(__name__)` | never `print` in production code |

| Operation | list | set / dict | deque | numpy array |
|---|---|---|---|---|
| index `x[i]` | O(1) | n/a / O(1) by key | O(n) | O(1) |
| membership `in` | O(n) | O(1) average | O(n) | O(n) |
| append / pop end | O(1) | O(1) | O(1) both ends | O(n) copy |
| insert / pop front | O(n) | n/a | O(1) | O(n) |
| sort | O(n log n) | n/a | n/a | O(n log n) |

!!! pitfall "Pitfall: mutable default arguments"
    `def f(x, acc=[])` shares one list across calls. Use `acc=None` then `acc = acc or []`. The same trap appears with `dict` and with DataFrames passed into functions that mutate them; return new objects instead.

!!! tip "Pro tip: environments"
    Use one environment per project: `uv venv && uv pip install -r requirements.txt` (fast), or `poetry`/`conda`. Pin exact versions with a lock file; ML artefacts (pickles) often break across library versions.

## NumPy: vectorise everything

@@code code/s01_numpy.py@@
@@out code/out/s01_numpy.txt@@

**Broadcasting rule.** Align shapes from the right; two dimensions are compatible if they are equal or one of them is 1. `(n, d) - (d,)` subtracts a row vector from every row; `(n, 1) * (1, m)` builds an outer product.

| Task | Idiom |
|---|---|
| Create | `np.zeros((n, d))`, `np.arange`, `np.linspace`, `rng = np.random.default_rng(42)` |
| Reshape | `a.reshape(-1, 1)`, `a[:, None]`, `a.T`, `np.squeeze`, `np.stack(axis=…)` |
| Reduce | `a.sum(axis=0)` (down columns), `a.mean(axis=1, keepdims=True)` |
| Select | boolean mask `a[a > 0]`, fancy index `a[[0, 2]]`, `np.where(cond, x, y)` |
| Linear algebra | `A @ B`, `np.linalg.solve`, `lstsq`, `eigh`, `svd`, `norm` |
| Einstein sum | `np.einsum("bij,bjk->bik", A, B)` (batched matmul) |
| Numerics | `np.log1p`, `np.expm1`, `scipy.special.logsumexp` (stable), `np.isclose` |

!!! pitfall "Pitfall: views vs copies"
    Slicing returns a **view**: `b = a[:10]; b[:] = 0` modifies `a`. Use `.copy()` when you need independence. Fancy indexing and boolean masks return copies.

## pandas: the core verbs

@@code code/s01_pandas.py@@
@@out code/out/s01_pandas.txt@@

| Task | pandas | SQL |
|---|---|---|
| Filter | `df.query("x > 5 and city == 'A'")` / `df[mask]` | `WHERE` |
| Project / derive | `df[["a", "b"]]`, `df.assign(r=lambda d: d.a / d.b)` | `SELECT a, b, a/b AS r` |
| Aggregate | `df.groupby("g").agg(n=("id", "size"), s=("x", "sum"))` | `GROUP BY g` |
| Window | `groupby().transform`, `.rank()`, `.shift()`, `.rolling()` | `OVER (PARTITION BY … ORDER BY …)` |
| Join | `a.merge(b, on="k", how="left", validate="m:1")` | `LEFT JOIN … USING (k)` |
| Reshape | `pivot_table`, `melt`, `stack`, `unstack`, `explode` | `PIVOT` / `UNPIVOT` |
| Time | `pd.to_datetime`, `.dt.*`, `resample("W")`, `asof` merge | `DATE_TRUNC`, range joins |

!!! pitfall "Pitfall: silent join explosions"
    A many-to-many merge multiplies rows without any error. Always pass `validate="one_to_one"` or `"many_to_one"` and compare `len()` before and after. Use `indicator=True` to audit unmatched keys.

!!! tip "Pro tip: speed and memory"
    `category` dtype for low-cardinality strings (10–50× less memory), `pd.read_csv(..., engine="pyarrow", dtype_backend="pyarrow")`, avoid `apply(axis=1)` and `iterrows` (100× slower than vectorised ops), use `.loc[mask, col] = v` to avoid `SettingWithCopyWarning`, and enable Copy-on-Write (`pd.options.mode.copy_on_write = True`), which is the default from pandas 3.0.

## Polars and DuckDB: when pandas is too slow

@@code code/s01_polars_duckdb.py@@
@@out code/out/s01_polars_duckdb.txt@@

| | pandas | Polars | DuckDB |
|---|---|---|---|
| Interface | DataFrame API | expression API (lazy + eager) | SQL (+ relational Python API) |
| Engine | mostly single-threaded | Rust, multi-threaded, query optimiser | C++, vectorised, multi-threaded, out-of-core |
| Sweet spot | small/medium data, ecosystem glue | 1–100 GB single machine, pipelines | ad-hoc SQL on Parquet/CSV, local warehouse |
| Interop | Arrow ↔ all three zero-copy | `to_pandas()`, `from_arrow` | `.df()`, `.pl()`, queries pandas frames by name |

!!! rule "Rule of thumb"
    Under ~1 GB: pandas is fine. 1–100 GB on one machine: Polars or DuckDB on Parquet. Beyond one machine, or many concurrent users: Spark, a warehouse, or a lakehouse engine.

<div class="card" markdown="1">

## Cheat card: Python data stack

<div class="cols" markdown="1">

#### Load
- `pd.read_csv(p, usecols=, dtype=, parse_dates=)`
- `pd.read_parquet(p, columns=)` / `pl.scan_parquet`
- `duckdb.sql("SELECT … FROM 'x/*.parquet'")`

#### Inspect
- `df.info(memory_usage="deep")`, `df.describe().T`
- `df.isna().mean()`, `df.nunique()`, `value_counts(normalize=True)`

#### Transform
- `assign`, `query`, `loc[mask, col]`
- `groupby().agg(named=…)`, `transform`, `rank`, `shift`, `rolling`
- `merge(validate=…)`, `pivot_table`, `melt`

#### NumPy
- broadcast instead of loops; `keepdims=True`
- `solve`/`lstsq`, never `inv`
- `default_rng(seed)` for reproducibility

#### Performance
- `category` dtype; Arrow backend
- Parquet over CSV; select columns early
- Polars lazy / DuckDB for > 1 GB

#### Hygiene
- one env per project, locked versions
- type hints + `ruff` + `pytest`
- no mutable defaults; return new frames

</div>
</div>
