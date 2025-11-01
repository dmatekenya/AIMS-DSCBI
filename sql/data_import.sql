
-- LOAD DATA FROM CSV FILES
COPY cells FROM '/Users/dmatekenya/My Drive (dmatekenya@gmail.com)/TEACHING/AIMS-DSCBI/data/tmp-db-data/cells.csv' DELIMITER ',' CSV HEADER;

COPY pop(cell_id, elderly_60, children_under5, youth_15_24, general_pop, men_2020, women_2020, building_count)
FROM '/Users/dmatekenya/My Drive (dmatekenya@gmail.com)/TEACHING/AIMS-DSCBI/data/tmp-db-data/population.csv'
CSV HEADER;

COPY ntl_annual (cell_id, ntl_min, ntl_max, ntl_mean, ntl_median, ntl_sum,
                  pixel_count, raster_filename, year)
FROM '/Users/dmatekenya/My Drive (dmatekenya@gmail.com)/TEACHING/AIMS-DSCBI/data/tmp-db-data/ntl-annual-2012-2024.csv' 
DELIMITER ',' CSV HEADER;

\COPY ntl_monthly(cell_id, ntl_min, ntl_max, ntl_mean, ntl_median, ntl_sum,
                  pixel_count, raster_filename, year, month, date)
FROM '/Users/dmatekenya/My Drive (dmatekenya@gmail.com)/TEACHING/AIMS-DSCBI/data/tmp-db-data/ntl-monthly-2012-2024.csv'
CSV HEADER;
