WITH RECURSIVE chain AS (
  SELECT id, name, manager_id, 1 AS lvl, name AS path FROM employees WHERE manager_id IS NULL
  UNION ALL
  SELECT e.id, e.name, e.manager_id, c.lvl + 1, c.path || ' > ' || e.name
  FROM employees e JOIN chain c ON e.manager_id = c.id)
SELECT lvl, name, path FROM chain ORDER BY path;
