# SQL for Analytics & Engineering

tagline: The language every data role is judged on: execution order, joins, window functions, the classic interview patterns, NULL traps, slowly changing dimensions and query performance. Every query here was executed in DuckDB against a seeded shop database.

## The seeded database

@@code code/s12a_seed.py@@
@@out code/out/s12a_seed.txt@@

## Logical execution order

| Step | Clause | Consequence |
|---|---|---|
| 1 | `FROM` / `JOIN` | build the row set |
| 2 | `WHERE` | filter rows (no aggregates, no SELECT aliases) |
| 3 | `GROUP BY` | form groups |
| 4 | `HAVING` | filter groups (aggregates allowed) |
| 5 | window functions | computed over the filtered, grouped result |
| 6 | `QUALIFY` | filter on window results (DuckDB, Snowflake, BigQuery) |
| 7 | `SELECT` / `DISTINCT` | project and name columns |
| 8 | `ORDER BY` | aliases usable here |
| 9 | `LIMIT` / `OFFSET` | trim |

@@code code/sql/q01_anatomy.sql@@
@@out code/out/q01_anatomy.txt@@

!!! interview "Interview: WHERE vs HAVING"
    `WHERE` filters rows before grouping; `HAVING` filters groups after aggregation. Filter as early as possible (`WHERE`) so less data is grouped; use `HAVING` only for conditions on aggregates like `SUM(amount) > 100000`.

## Joins

| Join | Returns | Typical bug |
|---|---|---|
| `INNER` | matching rows only | silently drops unmatched rows |
| `LEFT` | all left rows, NULLs if no match | filter on right table in `WHERE` turns it into an INNER join: put it in `ON` |
| `FULL OUTER` | everything from both | reconciliation; coalesce keys |
| `CROSS` | Cartesian product | accidental when the join condition is missing |
| Semi (`EXISTS`, `IN`) | left rows with a match, no duplication | |
| Anti (`NOT EXISTS`) | left rows without a match | `NOT IN` + NULL returns nothing |
| `ASOF` | nearest earlier match by time | point-in-time features, prices at trade time |

!!! pitfall "Pitfall: join fan-out"
    Joining orders (many) to payments (many) per customer multiplies rows, so `SUM(amount)` double counts. Always know each table's grain; aggregate to the join grain **before** joining, and assert row counts (`COUNT(*)` before vs after).

## Window functions

$$\text{func}(\ldots)\ \texttt{OVER}\ (\texttt{PARTITION BY}\ g\ \texttt{ORDER BY}\ t\ \texttt{ROWS BETWEEN}\ \ldots)$$

| Family | Functions | Use |
|---|---|---|
| Ranking | `ROW_NUMBER`, `RANK` (gaps), `DENSE_RANK` (no gaps), `NTILE(n)`, `PERCENT_RANK` | top-N, dedup, quantile buckets |
| Offset | `LAG`, `LEAD`, `FIRST_VALUE`, `LAST_VALUE`, `NTH_VALUE` | period-over-period, sessionisation |
| Aggregate | `SUM`, `AVG`, `COUNT`, `MIN`, `MAX` over a frame | running totals, moving averages, share of total |
| Frame | `ROWS` (physical) vs `RANGE` (logical values) | `RANGE` includes ties; default frame with ORDER BY is RANGE UNBOUNDED PRECEDING |

### Latest record per key (dedup)

@@code code/sql/q02_dedup_latest.sql@@
@@out code/out/q02_dedup_latest.txt@@

### Month-over-month growth and moving average

@@code code/sql/q03_mom.sql@@
@@out code/out/q03_mom.txt@@

### Top-N per group and share of total

@@code code/sql/q04_topn_rank.sql@@
@@out code/out/q04_topn_rank.txt@@

## Classic interview patterns

### Gaps and islands (streaks)

**Intuition:** for consecutive dates, date minus row number is constant, so it identifies each island. **Example:** dates 3, 4, 5, 8 with row numbers 1, 2, 3, 4 give 2, 2, 2, 4: two islands.

@@code code/sql/q05_islands.sql@@
@@out code/out/q05_islands.txt@@

### Recursive CTE (hierarchies, graphs)

@@code code/sql/q06_recursive.sql@@
@@out code/out/q06_recursive.txt@@

### Cohort retention

@@code code/sql/q07_cohort.sql@@
@@out code/out/q07_cohort.txt@@

| Pattern | Technique |
|---|---|
| Second highest salary | `DENSE_RANK() = 2`, or `MAX` where `< MAX` |
| Running total | `SUM(x) OVER (ORDER BY t ROWS UNBOUNDED PRECEDING)` |
| Sessionisation | `LAG(ts)`; new session when gap > 30 min; `SUM(new_flag) OVER (ORDER BY ts)` |
| Funnel | `COUNT(DISTINCT user) FILTER (WHERE step = ...)` or conditional aggregation |
| Median | `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x)` / `MEDIAN(x)` |
| Pivot | `SUM(CASE WHEN k = 'a' THEN v END)`, or `PIVOT` |
| Users who did A but not B | anti-join with `NOT EXISTS` |
| Year-over-year | self-join on period − 1 or `LAG(x, 12)` |
| Point-in-time join | `ASOF JOIN` or `ROW_NUMBER` over events ≤ timestamp |

## NULL semantics and other traps

@@code code/sql/q08_pitfall_notin.sql@@
@@out code/out/q08_pitfall_notin.txt@@

| Expression | Result | Why |
|---|---|---|
| `NULL = NULL` | NULL (not TRUE) | unknown; use `IS NULL` / `IS NOT DISTINCT FROM` |
| `x NOT IN (1, NULL)` | NULL for every x | `x <> NULL` is unknown |
| `COUNT(col)` vs `COUNT(*)` | excludes vs includes NULLs | |
| `AVG(col)` | ignores NULLs | treat missing as 0? use `COALESCE` explicitly |
| `SUM` of no rows | NULL, not 0 | `COALESCE(SUM(x), 0)` |
| Integer division `5/2` | 2 in Postgres/SQL Server | cast to decimal |
| `BETWEEN` on timestamps | inclusive at both ends | use `>= start AND < end` |
| `DISTINCT` on many columns | hides fan-out bugs | find the grain instead |

## Slowly changing dimensions

| Type | Behaviour | Use |
|---|---|---|
| 0 | never changes | birth date |
| 1 | overwrite | typo fixes; history not needed |
| 2 | new row with `valid_from`, `valid_to`, `is_current` | full history for point-in-time analysis |
| 3 | previous-value column | only last change matters |
| 4 / 6 | history table / hybrid 1+2+3 | |

@@code code/sql/q09_scd2.sql@@
@@out code/out/q09_scd2.txt@@

!!! tip "Pro tip: SCD2 in practice"
    Use `MERGE` (Snowflake, Databricks, BigQuery, Postgres 15+) or dbt snapshots instead of hand-written update + insert. Hash the tracked attributes to detect changes cheaply. Join facts to the dimension with `fact.ts >= valid_from AND fact.ts < valid_to` to get the attributes as they were at event time.

## Query performance

| Symptom | Likely cause | Fix |
|---|---|---|
| Full scan on a big table | no partition pruning / index | filter on the partition/cluster key; avoid functions on the filtered column |
| Join explodes | many-to-many, skewed key | pre-aggregate; dedup; salt skewed keys |
| Spill to disk | huge sort/hash | reduce columns early; increase memory; approximate (`approx_count_distinct`) |
| Slow `COUNT(DISTINCT)` | exact distinct on billions | HyperLogLog sketches |
| Repeated heavy subquery | recomputation | materialise (temp table, materialised view, dbt model) |
| `SELECT *` on columnar store | reads all columns | select only needed columns |

!!! rule "Rule of thumb: read the plan"
    `EXPLAIN ANALYZE` before optimising. In row stores (Postgres) indexes matter: B-tree on equality/range filters and join keys, composite index order = equality columns first, then range. In columnar warehouses (BigQuery, Snowflake, Redshift, DuckDB) partitioning, clustering/sort keys and column pruning matter; indexes mostly don't exist.

<div class="card" markdown="1">

## Cheat card: SQL

<div class="cols" markdown="1">

#### Order of evaluation
- FROM → WHERE → GROUP BY → HAVING → WINDOW → QUALIFY → SELECT → ORDER BY → LIMIT

#### Window recipes
- Dedup: `ROW_NUMBER() OVER (PARTITION BY k ORDER BY ts DESC) = 1`
- Running total: `SUM(x) OVER (ORDER BY t)`
- Moving avg: `AVG(x) OVER (ORDER BY t ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)`
- MoM: `x / LAG(x) OVER (ORDER BY m) - 1`
- Share: `x / SUM(x) OVER (PARTITION BY g)`

#### Patterns
- Islands: date − ROW_NUMBER
- Sessions: LAG gap > threshold, cumulative SUM of flags
- Hierarchy: WITH RECURSIVE
- Anti-join: NOT EXISTS (never NOT IN with NULLs)

#### NULLs
- `IS NULL`, `COALESCE`, `IS NOT DISTINCT FROM`
- COUNT(col) skips NULLs; SUM of nothing is NULL

#### Grain discipline
- State the grain of every table
- Aggregate before joining many-to-many
- Assert row counts after joins

#### Performance
- Filter on partition keys; select few columns
- EXPLAIN ANALYZE; avoid functions on filtered columns
- Materialise reused heavy CTEs

</div>
</div>
