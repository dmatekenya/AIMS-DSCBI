-- ============================================================
-- basic_sql.sql
-- Basic SQL exploration of ntl_pop database
-- ============================================================

-- 1. Quick sanity checks
\echo '--- Sanity checks: tables and counts ---'
\dt

SELECT 'cells' AS table, COUNT(*) FROM cells
UNION ALL SELECT 'pop', COUNT(*) FROM pop
UNION ALL SELECT 'ntl_annual', COUNT(*) FROM ntl_annual
UNION ALL SELECT 'ntl_monthly', COUNT(*) FROM ntl_monthly;

SELECT * FROM cells LIMIT 5;
SELECT * FROM pop LIMIT 5;
SELECT * FROM ntl_annual LIMIT 5;
SELECT * FROM ntl_monthly LIMIT 5;

-- ============================================================

-- 2. Annual summaries
\echo '--- Annual summaries ---'
SELECT year, AVG(ntl_mean) AS avg_ntl_mean
FROM ntl_annual
GROUP BY year
ORDER BY year;

SELECT cell_id, ntl_mean
FROM ntl_annual
WHERE year = 2023
ORDER BY ntl_mean DESC
LIMIT 5;

-- ============================================================

-- 3. Population filters
\echo '--- Population filters (elderly > 50% of general pop) ---'
SELECT c.cell_name, p.elderly_60, p.general_pop
FROM cells c
JOIN pop p USING (cell_id)
WHERE p.elderly_60 > 0.5 * NULLIF(p.general_pop, 0)
ORDER BY p.elderly_60 DESC;

-- ============================================================

-- 4. Joins and multi-table queries
\echo '--- Annual NTL joined with cell attributes ---'
SELECT
  c.cell_name,
  c.province_name,
  c.district_name,
  a.year,
  a.ntl_mean,
  a.ntl_sum
FROM ntl_annual a
JOIN cells c ON a.cell_id = c.cell_id
WHERE a.year = 2023
ORDER BY a.ntl_mean DESC;

\echo '--- Annual NTL + population: per capita light ---'
SELECT
  c.cell_name,
  c.district_name,
  a.year,
  a.ntl_sum / NULLIF(p.general_pop, 0) AS light_per_capita,
  a.ntl_sum,
  p.general_pop
FROM ntl_annual a
JOIN cells c ON a.cell_id = c.cell_id
JOIN pop   p ON a.cell_id = p.cell_id
WHERE a.year = 2023
ORDER BY light_per_capita DESC NULLS LAST;

\echo '--- District-level aggregate: avg ntl_mean + total ntl_sum ---'
SELECT
  c.province_name,
  c.district_name,
  AVG(a.ntl_mean) AS avg_ntl_mean,
  SUM(a.ntl_sum)  AS total_ntl_sum
FROM ntl_annual a
JOIN cells c ON a.cell_id = c.cell_id
WHERE a.year = 2023
GROUP BY c.province_name, c.district_name
ORDER BY total_ntl_sum DESC;

-- ============================================================

-- 5. Monthly time-series queries
\echo '--- Monthly trend for a specific cell ---'
SELECT year, month, ntl_mean
FROM ntl_monthly
WHERE cell_id = 'RW-001-123'
ORDER BY year, month;

\echo '--- Monthly trend by date ---'
SELECT date, ntl_mean
FROM ntl_monthly
WHERE cell_id = 'RW-001-123'
ORDER BY date;

\echo '--- Average ntl_mean per month across all cells ---'
SELECT year, month, AVG(ntl_mean) AS avg_ntl_mean
FROM ntl_monthly
GROUP BY year, month
ORDER BY year, month;

\echo '--- District-level monthly averages (2023) ---'
SELECT
  m.year,
  m.month,
  c.district_name,
  AVG(m.ntl_mean) AS avg_ntl_mean
FROM ntl_monthly m
JOIN cells c ON m.cell_id = c.cell_id
WHERE m.year = 2023
GROUP BY m.year, m.month, c.district_name
ORDER BY m.year, m.month, c.district_name;

-- ============================================================

-- 6. Integrity checks and indexes
\echo '--- Integrity checks ---'
SELECT p.cell_id
FROM pop p
LEFT JOIN cells c USING (cell_id)
WHERE c.cell_id IS NULL;

\echo '--- Recommended indexes for performance ---'
CREATE INDEX IF NOT EXISTS idx_annual_cell_id_year
  ON ntl_annual (cell_id, year);
CREATE INDEX IF NOT EXISTS idx_monthly_cell_id_date
  ON ntl_monthly (cell_id, date);
CREATE INDEX IF NOT EXISTS idx_cells_cell_id
  ON cells (cell_id);
CREATE INDEX IF NOT EXISTS idx_pop_cell_id
  ON pop (cell_id);

-- ============================================================
-- END OF SCRIPT
-- ============================================================

