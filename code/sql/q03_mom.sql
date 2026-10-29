WITH m AS (
  SELECT DATE_TRUNC('month', created_at) AS month, SUM(amount) AS rev
  FROM orders WHERE status = 'paid' GROUP BY 1)
SELECT month, ROUND(rev) AS rev,
       ROUND(100 * (rev / LAG(rev) OVER (ORDER BY month) - 1), 1)           AS mom_pct,
       ROUND(AVG(rev) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)) AS rev_3m_avg
FROM m ORDER BY month LIMIT 6;
