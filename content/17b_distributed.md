# Distributed Data Systems

tagline: Partitioning, replication, consistency, the shuffle, skew and exactly-once processing.

## Partitioning

| Scheme | Strength | Weakness |
|---|---|---|
| Hash | even spread for uniform keys | range scans touch every partition |
| Range | efficient range scans | hot spots on sequential keys (timestamps) |
| Consistent hashing | adding a node moves about $1/N$ of keys | needs virtual nodes for balance |

Skewed keys break hash partitioning: a stage finishes only when its largest partition does.

![](figures/v2_skew.svg)

@@code code/n09_partition_skew.py@@
@@out code/out/n09_partition_skew.txt@@

!!! tip "Pro tip"
    In Spark, enable adaptive query execution (skew join handling) first. If skew persists, salt the hot keys: append a random suffix in $\{0,\dots,S-1\}$ on the large side, replicate the small side $S$ times, join, then aggregate.

## Replication and consistency

With $N$ replicas, writes acknowledged by $W$ and reads from $R$, quorum overlap $R+W>N$ guarantees a read sees the latest acknowledged write (absent concurrent failures).

| Model | Guarantee | Example systems |
|---|---|---|
| Linearizable | behaves like a single copy | etcd, Spanner |
| Sequential / causal | order respected per client or causal chain | some geo-replicated stores |
| Eventual | replicas converge if writes stop | Dynamo-style stores, DNS |

**CAP**: during a network partition a system must choose between consistency and availability. **PACELC** adds: else (no partition) choose between latency and consistency.

## The shuffle

Wide transformations (`groupBy`, `join`, `distinct`) redistribute data across the network. Cost is roughly proportional to bytes shuffled, so:

- filter and project before wide operations;
- broadcast the small side of a join (below about 10-100 MB);
- pre-aggregate with map-side combiners (`reduceByKey` rather than `groupByKey`);
- partition tables on disk by common join or filter keys.

## Delivery semantics

| Semantics | Meaning | How |
|---|---|---|
| At most once | may lose messages | commit offset before processing |
| At least once | may duplicate | commit after processing; retries |
| Exactly once (effectively) | each effect applied once | idempotent writes or transactional offset + output commits |

!!! pitfall "Pitfall: exactly-once at the broker is not end to end"
    Kafka transactions make read-process-write within Kafka atomic. A side effect in an external system (an email, an HTTP call, a non-transactional database write) still needs an idempotency key.

## Cheat card: Distributed data

- Pick a partition key that matches access patterns and has high cardinality.
- Detect skew by max/mean partition size; fix with AQE, salting or broadcast.
- $R+W>N$ for read-your-latest-write quorums.
- Minimise shuffled bytes: filter early, broadcast small tables.
- Design sinks to be idempotent; treat exactly-once as a property of the whole pipeline.
