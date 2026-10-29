SELECT   c.region, COUNT(*) AS n_orders, ROUND(SUM(o.amount)) AS revenue
FROM     orders o
JOIN     customers c ON c.id = o.customer_id
WHERE    o.status = 'paid' AND o.created_at >= DATE '2025-09-01'
GROUP BY c.region
HAVING   SUM(o.amount) > 100000
ORDER BY revenue DESC;
