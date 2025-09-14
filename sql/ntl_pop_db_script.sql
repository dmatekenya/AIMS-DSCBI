-- Cell information
CREATE TABLE cells (
    cell_id TEXT PRIMARY KEY,
    province_name TEXT,
    district_name TEXT,
    sector_name TEXT,
    cell_name TEXT
);

-- Population 
CREATE TABLE pop (
    cell_id TEXT PRIMARY KEY,
    elderly_60 REAL,
    general_pop REAL,
    children_under5 REAL,
    youth_15_24 REAL,
    men_2020 REAL,
    women_2020 REAL,
    building_count REAL
);

-- Annual NTL stats
CREATE TABLE ntl_annual (
    id SERIAL PRIMARY KEY,
    cell_id TEXT REFERENCES cells(cell_id),
    ntl_min REAL,
    ntl_max REAL,
    ntl_mean REAL,
    ntl_median REAL,
    ntl_sum REAL,
    pixel_count REAL,
    raster_filename TEXT,
    year INT
);


-- Monthly NTL stats
CREATE TABLE ntl_monthly (
    id SERIAL PRIMARY KEY,
    cell_id TEXT REFERENCES cells(cell_id),
    ntl_min REAL,
    ntl_max REAL,
    ntl_mean REAL,
    ntl_median REAL,
    ntl_sum REAL,
    pixel_count REAL,
    raster_filename TEXT,
    year INT,
    month INT,
    date DATE
);


-- LOAD DATA FROM CSV FILES
COPY cells FROM '/Users/dmatekenya/My Drive (dmatekenya@gmail.com)/TEACHING/AIMS-DSCBI/data/tmp-db-data/cells.csv' DELIMITER ',' CSV HEADER;
COPY pop FROM '/Users/dmatekenya/My Drive (dmatekenya@gmail.com)/TEACHING/AIMS-DSCBI/data/tmp-db-data/population.csv' DELIMITER ',' CSV HEADER;
COPY ntl_annual FROM '/Users/dmatekenya/My Drive (dmatekenya@gmail.com)/TEACHING/AIMS-DSCBI/data/tmp-db-data/ntl-annual-2012-2024.csv' DELIMITER ',' CSV HEADER;
\copy ntl_monthly(cell_id, ntl_min, ntl_max, ntl_mean, ntl_median, ntl_sum,
                  pixel_count, raster_filename, year, month, date)
FROM '/Users/dmatekenya/My Drive (dmatekenya@gmail.com)/TEACHING/AIMS-DSCBI/data/tmp-db-data/merged-zonal-stats-2012-2024.csv'
CSV HEADER;


--BASIC QUERIES 
-- Check average light in a given year
SELECT year, AVG(ntl_mean)
FROM ntl_annual
GROUP BY year
ORDER BY year;

-- Get top 5 brightest cells in 2023
SELECT cell_id, ntl_sum
FROM ntl_annual
WHERE year = 2023
ORDER BY ntl_sum DESC
LIMIT 5;


