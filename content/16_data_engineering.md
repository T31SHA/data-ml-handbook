# Data Engineering Foundations

tagline: Architectures, storage formats, modelling, ingestion and change data capture, batch vs streaming, data quality and contracts, and a design checklist for pipelines that survive production.

## Reference architectures

![](figures/data_arch.svg)

| Architecture | Storage | Strength | Weakness | Examples |
|---|---|---|---|---|
| Data warehouse | managed columnar, schema-on-write | SQL performance, governance | cost for raw/unstructured data | Snowflake, BigQuery, Redshift |
| Data lake | object storage files, schema-on-read | cheap, any format | "data swamp" without governance; no ACID | S3/ADLS/GCS + Parquet |
| Lakehouse | open table format on object storage | ACID + cheap storage + one copy for BI and ML | more moving parts | Delta Lake, Apache Iceberg, Hudi on Databricks/Spark/Trino |
| Data mesh | domain-owned data products | scales organisationally | needs a platform and strong standards | federated governance |

### Medallion layering

![](figures/medallion.svg)

| Layer | Contents | Rules |
|---|---|---|
| Bronze (raw) | data as received, append-only, with ingestion metadata | never edit; replayable; keep the source schema |
| Silver (clean) | typed, deduplicated, conformed, joined to reference data | quality checks gate promotion; SCD2 where needed |
| Gold (serving) | business aggregates, marts, features | modelled for consumers; documented; SLAs |

## Storage formats and table formats

| Format | Layout | Best for | Notes |
|---|---|---|---|
| CSV / JSON | row, text | interchange, small data | no types, slow, ambiguous; JSON for nested events |
| Avro | row, binary, schema embedded | Kafka messages, write-heavy | schema evolution with a registry |
| Parquet | columnar, compressed, statistics per row group | analytics (the default) | column pruning + predicate pushdown |
| ORC | columnar | Hive ecosystems | similar to Parquet |
| Arrow | columnar **in memory** | zero-copy between pandas/Polars/DuckDB/Spark | not a storage format per se |
| Delta / Iceberg / Hudi | Parquet + transaction log / metadata | ACID, time travel, schema evolution, MERGE | Iceberg: engine-neutral; Delta: Databricks-native |

!!! intuition "Why columnar is fast for analytics"
    A query like `SELECT AVG(amount) WHERE country = 'KE'` touches 2 of maybe 100 columns. Columnar files read only those 2, compress each column well (similar values together, dictionary/run-length encoding), and skip whole row groups whose min/max statistics exclude 'KE'. Reading 2% of the bytes is often a 50× speedup.

!!! rule "Rule of thumb: file sizing and partitioning"
    Target 128 MB–1 GB Parquet files. Partition by a low-cardinality column used in most filters (usually date); avoid partitions under ~1 GB or more than a few thousand partitions. Thousands of tiny files ("small file problem") kill performance: compact them (`OPTIMIZE`, Iceberg rewrite). Use clustering / Z-order for secondary filter columns.

## Data modelling

![](figures/star.svg)

| Approach | Shape | Use |
|---|---|---|
| Normalised (3NF) | many narrow tables, no redundancy | OLTP applications: integrity, writes |
| Dimensional (Kimball star) | facts (events, measures) + dimensions (who/what/where) | BI and analytics: simple joins, fast |
| Snowflake schema | normalised dimensions | rarely worth the extra joins |
| Data Vault | hubs, links, satellites | auditable enterprise integration, many sources |
| One Big Table (OBT) | wide denormalised table | columnar warehouses, dashboards, ML features |

| Fact table type | Grain | Example |
|---|---|---|
| Transaction | one row per event | order line |
| Periodic snapshot | one row per entity per period | daily account balance |
| Accumulating snapshot | one row per process instance, updated at milestones | order: placed → paid → shipped → delivered |
| Factless | events with no measure | student attended class |

!!! pitfall "Pitfall: undefined grain"
    The number-one modelling bug. Write the grain in one sentence ("one row per order line per day") before designing columns. Mixing grains in one table causes double counting that no query can fix.

## Ingestion: ETL, ELT and change data capture

![](figures/etl.svg)

| Pattern | How | When |
|---|---|---|
| Full load | copy everything each run | small tables, reference data |
| Incremental (watermark) | `WHERE updated_at > last_max` | append-mostly tables with reliable timestamps; misses hard deletes |
| CDC (log-based) | read the database transaction log (binlog/WAL) | inserts, updates **and deletes**, low source impact (Debezium, Fivetran, DMS) |
| Event streaming | producers publish events to Kafka/Kinesis/Pub/Sub | real-time, decoupled services |
| API pulls | paginated REST/GraphQL with rate limits | SaaS sources (Airbyte, Fivetran, custom) |
| File drops | SFTP/S3 landing zone + manifest | partners, legacy systems |

![](figures/cdc.svg)

!!! tip "Pro tip: CDC to a lakehouse"
    Land raw change events (op, before, after, LSN/ts) in bronze; in silver apply them with `MERGE` ordered by the log sequence number, deduplicating by key so the latest event wins, and handle deletes as tombstones (soft-delete flags). Snapshot + CDC from the snapshot's log position gives an exact starting point.

## Batch, streaming, lambda and kappa

![](figures/lambda_kappa.svg)

![](figures/kafka.svg)

| Concept | Meaning |
|---|---|
| Event time vs processing time | when it happened vs when it arrived; aggregate by event time |
| Watermark | "no events older than T are still expected"; closes windows; late data policy after it |
| Windows | tumbling (fixed, non-overlapping), sliding/hopping (overlapping), session (gap-based) |
| Delivery | at-most-once, at-least-once (+ idempotent sink = effectively once), exactly-once (transactions) |
| Backpressure | slow consumer signals producers / buffers grow; monitor consumer lag |
| Partitioning | key-based: per-key order guaranteed only within a partition |
| Compacted topic | keeps the latest value per key: a changelog / table |

!!! rule "Rule of thumb: do you need streaming?"
    If decisions are made hourly or daily, micro-batch is simpler and cheaper. Stream when the value decays in seconds to minutes: fraud, alerting, real-time features, operational dashboards. A good compromise: stream into the lake (bronze) and process in frequent small batches.

## Idempotency, backfills and late data

@@code code/s13_idempotent.py@@
@@out code/out/s13_idempotent.txt@@

| Principle | Implementation |
|---|---|
| Idempotent writes | overwrite a partition, or `MERGE` on a natural/business key; never blind `INSERT` on retries |
| Deterministic runs | parameterise by logical date (`ds`), not `now()` |
| Backfills | same code path as daily runs, over a date range; bounded concurrency |
| Late data | reprocess a trailing window (e.g. last 3 days) each run, or event-time windows with watermarks |
| Exactly-once illusion | at-least-once delivery + idempotent sink + dedup on event id |

## Data quality and contracts

@@code code/s13_quality.py@@
@@out code/out/s13_quality.txt@@

| Dimension | Check | Tool |
|---|---|---|
| Completeness | null rate, row count vs expected | dbt tests, Great Expectations, Soda, pandera |
| Uniqueness | primary key unique | `unique` test |
| Validity | ranges, enums, regex, types | schema checks |
| Consistency | referential integrity, totals reconcile across systems | `relationships` test, reconciliation queries |
| Timeliness | freshness: max(event_time) lag | dbt source freshness, SLAs |
| Distribution | drift in mean/quantiles/category mix | PSI, anomaly detection on metrics |

!!! tip "Pro tip: data contracts"
    A data contract is an agreed, versioned schema + semantics + quality SLAs between a producer team and consumers, enforced in CI (breaking changes fail the producer's build) and at runtime (schema registry compatibility: backward/forward/full). It moves quality upstream to where the data is created.

## Governance, security and cost

| Area | Practice |
|---|---|
| Catalog and lineage | Unity Catalog, DataHub, OpenMetadata; column-level lineage from dbt/Spark |
| Access control | RBAC/ABAC, row-level and column-level security, masking of PII |
| Privacy | data minimisation, retention policies, right-to-erasure workflows (deletes in lakehouse tables + vacuum) |
| Encryption | at rest (KMS) and in transit (TLS); secrets in a vault, never in code |
| Cost | partition pruning, auto-suspend warehouses, lifecycle rules to cold storage, monitor per-query cost |

## The data-engineering design checklist

| # | Question | Good answer looks like |
|---|---|---|
| 1 | Who consumes this and what decision does it drive? | named consumers, SLA (freshness, availability) |
| 2 | What is the grain of each table? | one sentence per table |
| 3 | Batch or streaming? | justified by how fast value decays |
| 4 | How do we ingest changes, including deletes? | CDC or watermark + periodic full reconciliation |
| 5 | Is every step idempotent and re-runnable for any date? | partition overwrite / MERGE; logical dates |
| 6 | How are late and duplicate events handled? | event-time windows, dedup keys, reprocessing window |
| 7 | Schema evolution? | contract, registry, additive changes only without a version bump |
| 8 | Data quality gates? | tests at bronze→silver→gold; failing checks block promotion |
| 9 | Observability? | freshness, volume, schema, distribution monitors; alerts with owners |
| 10 | Backfill plan? | documented, bounded concurrency, cost estimate |
| 11 | Storage layout? | format, partitioning, file size, compaction |
| 12 | Security and privacy? | PII classified, masked, access-controlled, retention set |
| 13 | Cost? | estimate per run and per month; pruning verified |
| 14 | Failure modes? | retries with backoff, dead-letter queue, runbook |
| 15 | Ownership? | team, on-call, documentation in the catalog |

<div class="card" markdown="1">

## Cheat card: Data engineering foundations

<div class="cols" markdown="1">

#### Architecture
- Lakehouse (Iceberg/Delta on object storage) for BI + ML on one copy
- Bronze raw → Silver clean → Gold marts/features
- Mesh = organisational, not a technology

#### Formats
- Parquet for analytics, Avro for Kafka, Arrow in memory
- Files 128 MB–1 GB; partition by date; compact small files

#### Modelling
- State the grain; star schema for BI; OBT for dashboards/ML
- SCD2 for history; surrogate keys for dimensions

#### Ingestion
- CDC (Debezium) catches deletes; watermarks don't
- MERGE ordered by log position; tombstones for deletes

#### Streaming
- Event time + watermarks; tumbling/sliding/session windows
- At-least-once + idempotent sink ≈ exactly-once
- Order only within a partition (by key)

#### Reliability
- Idempotent, parameterised by logical date
- Reprocess a trailing window for late data
- Quality gates between layers; contracts with producers

#### Observe
- Freshness, volume, schema, distribution, lineage

</div>
</div>
