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
    id SERIAL PRIMARY KEY,
    cell_id TEXT REFERENCES cells(cell_id),
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


