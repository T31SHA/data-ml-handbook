-- latest order per customer (dedup pattern) - QUALIFY filters on a window function
SELECT customer_id, order_id, amount, created_at
FROM orders
QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at DESC) = 1
ORDER BY customer_id
LIMIT 5;
