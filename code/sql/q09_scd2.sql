-- SCD Type 2: close the current row and insert a new version when an attribute changes
CREATE OR REPLACE TEMP TABLE dim_customer AS
SELECT id, region, DATE '2025-01-01' AS valid_from, DATE '9999-12-31' AS valid_to, TRUE AS is_current
FROM customers WHERE id <= 3;
CREATE OR REPLACE TEMP TABLE staging AS SELECT * FROM (VALUES (1, 'Kisumu'), (2, 'Nairobi')) t(id, region);
UPDATE dim_customer d SET valid_to = DATE '2026-10-24', is_current = FALSE
FROM staging s WHERE d.id = s.id AND d.is_current AND d.region <> s.region;
INSERT INTO dim_customer
SELECT s.id, s.region, DATE '2026-10-24', DATE '9999-12-31', TRUE FROM staging s
WHERE NOT EXISTS (SELECT 1 FROM dim_customer d WHERE d.id = s.id AND d.is_current AND d.region = s.region);
SELECT * FROM dim_customer ORDER BY id, valid_from;
