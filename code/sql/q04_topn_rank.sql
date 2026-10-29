WITH spend AS (
  SELECT c.region, c.name, SUM(o.amount) AS total
  FROM orders o JOIN customers c ON c.id = o.customer_id GROUP BY 1, 2)
SELECT region, name, ROUND(total) AS total,
       RANK()       OVER w AS rnk,
       DENSE_RANK() OVER w AS dense_rnk,
       ROUND(100 * total / SUM(total) OVER (PARTITION BY region), 2) AS pct_of_region
FROM spend
WINDOW w AS (PARTITION BY region ORDER BY total DESC)
QUALIFY rnk <= 2
ORDER BY region, rnk;
