-- NOT IN with a NULL in the subquery returns NOTHING, while NOT EXISTS is NULL-safe
SELECT 'NOT IN' AS method, COUNT(*) AS n FROM employees
WHERE id NOT IN (SELECT manager_id FROM employees)              -- manager_id contains NULL
UNION ALL
SELECT 'NOT EXISTS', COUNT(*) FROM employees e
WHERE NOT EXISTS (SELECT 1 FROM employees x WHERE x.manager_id = e.id);
