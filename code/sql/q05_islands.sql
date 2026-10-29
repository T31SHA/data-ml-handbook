-- gaps & islands: longest consecutive-day login streak per user
WITH g AS (
  SELECT user_id, d, d - CAST(ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY d) AS INTEGER) AS grp
  FROM logins)
SELECT user_id, MIN(d) AS start_d, MAX(d) AS end_d, COUNT(*) AS streak
FROM g GROUP BY user_id, grp
QUALIFY ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY COUNT(*) DESC) = 1
ORDER BY streak DESC LIMIT 4;
