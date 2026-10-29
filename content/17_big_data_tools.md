# Spark, Kafka, Airflow & dbt

tagline: The four tools behind most data platforms: how Spark really executes, how to tune and debug it, Kafka's guarantees, orchestration with Airflow, and analytics engineering with dbt. Code in this section is reference syntax (it needs a cluster) and is labelled as not executed.

## Spark execution model

![](figures/spark_dag.svg)

![](figures/spark.svg)

| Concept | Meaning |
|---|---|
| Driver / executors | driver plans and schedules; executors run tasks on partitions |
| Transformation (lazy) | `select`, `filter`, `join`, `groupBy`: builds a logical plan |
| Action | `count`, `collect`, `write`: triggers a **job** |
| Narrow vs wide | narrow (map, filter) stays in a partition; wide (groupBy, join, distinct) needs a **shuffle** → new **stage** |
| Task | one partition processed by one core in one stage |
| Catalyst | optimises the logical plan (predicate pushdown, column pruning, join reordering) |
| Tungsten / Photon | code generation, off-heap memory, vectorised execution |
| AQE | adaptive query execution: coalesce shuffle partitions, switch join strategy, split skewed partitions at runtime |

**Rule:** jobs = actions; stages = shuffles + 1; tasks = partitions per stage. **Example:** read 50 GB of Parquet with 128 MB splits → ~400 tasks in stage 1; a `groupBy` creates a shuffle → stage 2 with `spark.sql.shuffle.partitions` tasks (default 200; AQE coalesces).

### PySpark idioms (reference, not executed)

```python
from pyspark.sql import SparkSession, functions as F, Window
spark = SparkSession.builder.appName("orders").getOrCreate()
orders = spark.read.parquet("s3://lake/silver/orders/")            # lazy
customers = spark.read.table("silver.customers")
w = Window.partitionBy("customer_id").orderBy(F.col("created_at").desc())
latest = (orders.filter(F.col("status") == "paid")
          .withColumn("rn", F.row_number().over(w)).filter("rn = 1").drop("rn"))
daily = (orders.groupBy(F.to_date("created_at").alias("dt"), "region")
         .agg(F.sum("amount").alias("revenue"), F.countDistinct("customer_id").alias("buyers")))
enriched = daily.join(F.broadcast(customers.select("region", "region_name").distinct()), "region")
(enriched.repartition("dt").write.mode("overwrite")
 .option("partitionOverwriteMode", "dynamic").partitionBy("dt").saveAsTable("gold.daily_revenue"))
enriched.explain("formatted")                                       # read the physical plan
```

### Join strategies

| Strategy | When | Cost |
|---|---|---|
| Broadcast hash join | one side small (< `autoBroadcastJoinThreshold`, 10 MB default; raise to ~100 MB–1 GB) | no shuffle of the big side |
| Sort-merge join | both sides large | shuffle + sort both |
| Shuffle hash join | medium sides, no sort needed | shuffle both |
| Bucketed join | both tables bucketed on the key, same bucket count | no shuffle at read |

### Performance tuning and debugging

| Symptom (Spark UI) | Cause | Fix |
|---|---|---|
| One task runs 10× longer | data skew on a key | AQE skew join, salting, broadcast the small side, filter hot keys separately |
| Executor OOM | big partitions, `collect()`, wide rows | more partitions, avoid `collect`, increase memory overhead |
| Spill (memory/disk) in shuffle | too few shuffle partitions | target ~100–200 MB per partition; AQE |
| Thousands of tiny tasks | too many small files / partitions | compact files, `coalesce` before write |
| Slow Python UDF | row-by-row serialisation | built-in functions; pandas UDFs (Arrow) |
| Recomputing a DataFrame | reused lineage | `cache()`/`persist()` then unpersist; or write a checkpoint |
| Full scan despite filter | filter on derived column / no partition column | filter on partition columns directly |

!!! pitfall "Pitfall: collect() and toPandas()"
    They bring the whole distributed dataset to the driver's memory. Aggregate or sample first, or write results to storage. Also: `count()` to "check" a DataFrame triggers a full job each time.

!!! interview "Interview: repartition vs coalesce"
    `repartition(n)` does a full shuffle and can increase or decrease partitions (and balance them, or partition by a column). `coalesce(n)` only merges existing partitions without a shuffle, so it is cheap but can leave unbalanced partitions; use it to reduce file count before writing.

## Kafka

| Concept | Meaning |
|---|---|
| Topic → partitions | ordered, append-only logs; parallelism unit |
| Offset | position in a partition; consumers commit offsets |
| Key | determines partition (`hash(key) % n`): per-key ordering |
| Consumer group | partitions are split among consumers; max useful consumers = partitions |
| Replication | `replication.factor=3`, `min.insync.replicas=2`, `acks=all` for durability |
| Retention | by time/size, or log compaction (latest per key) |
| Idempotent producer | `enable.idempotence=true`: no duplicates on retry |
| Transactions | exactly-once read-process-write within Kafka (`isolation.level=read_committed`) |
| Schema Registry | Avro/Protobuf/JSON schemas with compatibility rules |

```python
# reference, not executed (confluent-kafka)
from confluent_kafka import Producer, Consumer
p = Producer({"bootstrap.servers": "broker:9092", "acks": "all", "enable.idempotence": True})
p.produce("orders", key=str(order_id), value=json.dumps(event)); p.flush()
c = Consumer({"bootstrap.servers": "broker:9092", "group.id": "order-loader",
              "enable.auto.commit": False, "auto.offset.reset": "earliest"})
c.subscribe(["orders"])
while True:
    msg = c.poll(1.0)
    if msg is None or msg.error(): continue
    upsert(json.loads(msg.value()))          # idempotent sink keyed by order_id
    c.commit(msg)                            # commit AFTER processing => at-least-once
```

!!! rule "Rule of thumb: partitions"
    Partitions = max(target throughput / per-partition throughput, number of consumers you need). Adding partitions later breaks key → partition mapping (ordering), so over-provision moderately up front (e.g. 12–48 for a busy topic).

## Airflow (orchestration)

```python
# reference, not executed (Airflow 2.x TaskFlow)
from airflow.decorators import dag, task
from pendulum import datetime
@dag(schedule="@daily", start_date=datetime(2026, 1, 1, tz="Africa/Nairobi"), catchup=False,
     default_args={"retries": 3, "retry_exponential_backoff": True}, max_active_runs=1, tags=["orders"])
def orders_daily():
    @task
    def extract(ds=None) -> str:                 # ds = logical date: deterministic, backfillable
        return land_raw_orders(ds)               # writes s3://lake/bronze/orders/dt={ds}/
    @task
    def transform(path: str, ds=None):
        run_spark_job("silver_orders", ds=ds)    # partition overwrite => idempotent
    @task
    def quality(ds=None):
        assert_checks("silver.orders", ds)       # fail the run => downstream blocked
    transform(extract()) >> quality()
orders_daily()
```

| Practice | Why |
|---|---|
| Tasks are idempotent and keyed on the logical date | retries and backfills are safe |
| Airflow orchestrates; heavy compute runs elsewhere (Spark, warehouse, K8s pods) | scheduler stays healthy |
| Pass small metadata via XCom, never data | XCom lives in the metadata DB |
| Sensors in `reschedule` mode or deferrable operators | don't occupy worker slots |
| `catchup=False` unless you intend backfills | avoid surprise mass runs |
| SLAs / alerts on failure and lateness | ownership |

**Alternatives:** Dagster (asset-centric, typed, great local dev), Prefect (Pythonic, dynamic flows), cloud-native (Step Functions, Cloud Composer, Databricks Workflows).

## dbt (analytics engineering)

```sql
-- models/marts/fct_daily_revenue.sql   (reference, not executed)
{{ config(materialized='incremental', unique_key=['dt', 'region'], incremental_strategy='merge') }}
select date_trunc('day', o.created_at) as dt, c.region,
       sum(o.amount) as revenue, count(distinct o.customer_id) as buyers
from {{ ref('stg_orders') }} o
join {{ ref('dim_customers') }} c on c.customer_id = o.customer_id
where o.status = 'paid'
{% if is_incremental() %}
  and o.created_at >= (select max(dt) - interval 3 day from {{ this }})   -- reprocess late data
{% endif %}
group by 1, 2
```

```yaml
# models/marts/schema.yml
models:
  - name: fct_daily_revenue
    columns:
      - name: dt
        tests: [not_null]
      - name: revenue
        tests:
          - dbt_utils.accepted_range: {min_value: 0}
    tests:
      - dbt_utils.unique_combination_of_columns: {combination_of_columns: [dt, region]}
```

| Concept | Meaning |
|---|---|
| `ref()` / `source()` | dependencies → DAG and lineage |
| Materialisations | view, table, incremental, ephemeral, snapshot (SCD2) |
| Layers | staging (1:1 with sources, rename/cast) → intermediate → marts (facts/dims) |
| Tests | generic (unique, not_null, relationships, accepted_values) and singular (SQL returning failing rows) |
| Docs | `dbt docs generate`: catalog + lineage graph |
| CI | `dbt build --select state:modified+` on PRs (slim CI) |

<div class="card" markdown="1">

## Cheat card: Spark, Kafka, Airflow, dbt

<div class="cols" markdown="1">

#### Spark
- Lazy transformations; actions trigger jobs; shuffles cut stages
- Broadcast small tables; enable AQE
- Partitions ~100–200 MB; files 128 MB–1 GB
- Skew → salting / AQE skew join
- No `collect()` on big data; prefer built-ins over Python UDFs
- `explain()` and the Spark UI before tuning

#### Kafka
- Order per partition; key = entity id
- `acks=all`, idempotent producer, RF 3, min ISR 2
- Commit offsets after processing; idempotent sink
- Schema registry with compatibility rules

#### Airflow
- Idempotent tasks on logical date `ds`
- Orchestrate, don't compute; XCom = metadata only
- Retries with backoff; deferrable sensors; alerts

#### dbt
- staging → intermediate → marts
- `ref()` everything; tests on keys and ranges
- Incremental + merge + lookback window for late data
- Snapshots for SCD2; slim CI

</div>
</div>
