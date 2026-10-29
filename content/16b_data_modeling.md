# Data Modeling for Analytics

tagline: Normal forms, dimensional modelling, slowly changing dimensions, Data Vault and the grain decision that underpins them all.

## Start with the grain

The **grain** is what one row of a fact table represents ("one order line", "one page view", "one account per day"). Declare it before choosing columns. Every measure must be true at that grain; mixing grains (order totals on order-line rows) causes double counting.

## Normalisation versus denormalisation

| Form | Rule | Removes |
|---|---|---|
| 1NF | atomic values, no repeating groups | lists in cells |
| 2NF | 1NF + no partial dependency on part of a composite key | duplicated attributes per key part |
| 3NF | 2NF + no transitive dependencies | attributes depending on non-key attributes |

OLTP systems normalise (to 3NF) for write integrity. Analytical systems denormalise into star schemas for simple, fast reads.

## Star schema

A central fact table of measures and foreign keys, surrounded by dimension tables of descriptive attributes.

| Fact type | Grain | Example |
|---|---|---|
| Transaction | one event | order line |
| Periodic snapshot | one entity per period | account balance per day |
| Accumulating snapshot | one process instance, updated over milestones | order: placed, shipped, delivered |
| Factless | an event with no measure | student attended class |

!!! rule "Rule of thumb"
    Use surrogate keys in dimensions, keep natural keys as attributes, and store measures that are additive across all dimensions where possible. Ratios should be recomputed from additive components, not summed.

## Slowly changing dimensions

| Type | Behaviour | History |
|---|---|---|
| 0 | never change | original only |
| 1 | overwrite | none |
| 2 | new row with validity dates and a current flag | full |
| 3 | extra column for the previous value | one step |
| 6 | 1 + 2 + 3 combined | full, plus current value on every row |

@@code code/n08_scd2.py@@
@@out code/out/n08_scd2.txt@@

!!! pitfall "Pitfall: joining facts to the current dimension row"
    Reporting historical sales by the customer's *current* city rewrites history. Join on the surrogate key captured at load time, or on the natural key with the event date between `valid_from` and `valid_to`.

## Data Vault and wide tables

**Data Vault** splits data into hubs (business keys), links (relationships) and satellites (attributes with history). It is resilient to source changes and suits integration layers; marts are then built as stars on top. **One big table** (fully denormalised) is acceptable for a single heavily used use-case on columnar engines, but it duplicates logic and makes changes expensive.

## Cheat card: Data modeling

- Declare the grain first.
- Normalise for OLTP, star schema for analytics.
- Surrogate keys, conformed dimensions shared across facts.
- SCD2 with `valid_from`, `valid_to`, `is_current` for history.
- Keep measures additive; derive ratios at query time.
