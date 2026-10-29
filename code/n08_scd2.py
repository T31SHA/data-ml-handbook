import duckdb
con = duckdb.connect()
con.execute("""
CREATE TABLE dim_customer(customer_sk INTEGER, customer_id INTEGER, city VARCHAR,
  valid_from DATE, valid_to DATE, is_current BOOLEAN);
INSERT INTO dim_customer VALUES (1,100,'Nairobi','2025-01-01','9999-12-31',TRUE),
                                (2,200,'Mombasa','2025-01-01','9999-12-31',TRUE);
CREATE TABLE stg_customer(customer_id INTEGER, city VARCHAR, load_date DATE);
INSERT INTO stg_customer VALUES (100,'Kisumu','2026-03-01'),(200,'Mombasa','2026-03-01'),(300,'Nakuru','2026-03-01');
""")
# Type-2 merge: close changed rows, then insert new versions and new keys
con.execute("""
UPDATE dim_customer d SET valid_to = s.load_date - INTERVAL 1 DAY, is_current = FALSE
FROM stg_customer s WHERE d.customer_id = s.customer_id AND d.is_current AND d.city <> s.city;
INSERT INTO dim_customer
SELECT (SELECT max(customer_sk) FROM dim_customer) + row_number() OVER (ORDER BY s.customer_id),
       s.customer_id, s.city, s.load_date, DATE '9999-12-31', TRUE
FROM stg_customer s LEFT JOIN dim_customer d ON d.customer_id = s.customer_id AND d.is_current
WHERE d.customer_id IS NULL;
""")
print(con.sql("SELECT * FROM dim_customer ORDER BY customer_id, valid_from"))
# point-in-time lookup: which city was customer 100 in on 2025-06-30?
print(con.sql("SELECT city FROM dim_customer WHERE customer_id=100 AND DATE '2025-06-30' BETWEEN valid_from AND valid_to").fetchall())
