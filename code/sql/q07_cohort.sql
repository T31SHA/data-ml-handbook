WITH first AS (SELECT customer_id, DATE_TRUNC('month', MIN(created_at)) AS cohort FROM orders GROUP BY 1),
act AS (SELECT DISTINCT customer_id, DATE_TRUNC('month', created_at) AS m FROM orders)
SELECT strftime(f.cohort, '%Y-%m') AS cohort,
       COUNT(DISTINCT a.customer_id) FILTER (WHERE date_diff('month', f.cohort, a.m) = 0) AS m0,
       COUNT(DISTINCT a.customer_id) FILTER (WHERE date_diff('month', f.cohort, a.m) = 1) AS m1,
       COUNT(DISTINCT a.customer_id) FILTER (WHERE date_diff('month', f.cohort, a.m) = 2) AS m2
FROM first f JOIN act a USING (customer_id)
GROUP BY 1 ORDER BY 1 LIMIT 4;
