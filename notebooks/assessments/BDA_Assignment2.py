"""
GENERAL INSTRUCTIONS
1. WHICH PART TO CHANGE?: Uncomment every line with  [YOUR CODE HERE] and replace it with your code.
Please don't change anything else other than these lines. In other parts, the code isnt commented,
so just replace the parts with YOUR CODE or YOUR CODE HEREE

2. USE OF JUPYTER NOTEBOOK: For those who would like to use Jupyter Notebook. You can copy and paste
each function in the notebook environment, test your code their. However,
remember to paste back your code in a .py file and ensure that its running
okay.

3. INDENTATION: Please make sure that you check your indentation

4. Returning things from function: Please dont forget to use the return statement
to return a value when applicable

5. HINTS: please read my comments for hints and instructions where applicable

6. DEFINING YOUR OWN FUNCTIONS: where I ask you to define your own function
please make sure that you name the function exactly as I said.

7. Please work on the following functions in the order provided:
    - add_date_and_filter
    - get_events_before_jul13
    - preprocess_cdrs_using_spark
    - explore_data_with_spark
    - generate_user_attributes_with_pandas
8. Dataset for this work: all the questions are based on the simulated_cdrs dataset,
please download it and have it on your machine
"""
import random
import seaborn as sns
import numpy as np
from collections import namedtuple
from datetime import datetime
from pathlib import Path

def add_date_and_filter(csv_file, date_format, ref_date):
    """
    Create a dataframe, add date and filter out events based on ref date
    :return:
    """
    # read csv into pandas DataFrame
    df = YOUR CODE                          # POINTS: 1
    str_time_col = 'cdr datetime'

    # convert date string to Python datetime
    # please extract only the Year, month and day from the date string using
    # indexing, please use solutions from first assignment to achieve this
    # then use datetime.strptime() function to convert string to date
    df['cdr_date'] = YOUR CODE                  # POINTS: 1

    # please retrieve all events older than or equal to the reference date
    # please use the query function for pandas DataFrame like below
    df.query('cdr_date <= @ref_date', inplace=True)

    if df.shape[0]:
        return df
    else:
        return 'NA'

def get_events_before_jul13(csv_folder, ref_date, date_format, num_csv_files):
    """
    In this function, we will use the map function to run the
    add_date_and_filter above on a list of CSV_files
    :param csv_folder:
    :return:
    """
    list_csv = [f for f in csv_folder.iterdir() if f.suffix == '.csv']
    # use the function random.choices() to select only a sample of the CSVs  for fast processing
    # set k to num_csv_files in the function
    list_csv = YOUR CODE                        # POINTS: 2
    time_format = "%Y%m%d"

    # Use  the datetime.strptime() to convert the string ref_date to Python datetime object
    # This function requires two args, the other one is date format
    ref_date = datetime.strptime(ref_date,  date_format)

    # Run map with multiple iterables(r.g., two lists)
    # Note that input-1 is the list of CSV we generated above
    # Prepare input-2 and input-3 as below. Please use the  multiplication
    # operator on a single list item to repeat the same element multiple
    # times in a list as we did in the first assignment
    list_date_format = YOUR CODE                                    # POINTS: 1
    list_date_ref_date = YOUR CODE                                  # POINTS: 1

    # now run the map function with the three input lists we have defined
    results = YOUR CODE                                             # POINTS: 3

    # write code to identify elements in the list which arent s 'NA'
    # You can use a list comprehension with if condition. To identify elements
    # which are 'NA', you can use isinstance() function to detect data type
    no_na_results = []                                              # POINTS: 3
    print('I found {} non NA results'.format(len(no_na_results)))

    return no_na_results

def rename_sdf(df, mapper=None):
    ''' Rename column names of a dataframe
        mapper: a dict mapping from the old column names to new names
        Usage:
            df.rename({'old_col_name': 'new_col_name', 'old_col_name2': 'new_col_name2'})
            df.rename(old_col_name=new_col_name)
    '''
    for before, after in mapper.items():
        df = df.withColumnRenamed(before, after)

    return df

def preprocess_cdrs_using_spark(file_or_folder=None, number_of_users_to_sample=None,
                                output_csv=None, date_format='%Y%m%d%H%M%S',
                                debug_mode=True, loc_file=None, save_to_csv=False,
                                userid_col="user_id"):
    """
    In this function, we perfom some basic preprocessing such as below:
    1. rename columns
    2. change some data types
    3. Add location details
    Eventually, we will sample the data to use for our analysis
    :param data_folder:
    :param output_csv_for_sample_users:
    :return:
    """

    # ==============================
    # LOAD DATA
    # ==============================
    # create SparkSession
    spark = YOUR CODE           # POINTS: 1
    # read data with spark and ensure you set header to True
    df = YOUR CODE              # points: 1
    # repartition to speed up
    df = df.repartition(10)

    # if just testing/debugging, pick only a small subset of the dataset
    # for instance, 0.01 percent of the data
    # by using the sample function of spark
    if debug_mode:
        dfs = YOUR CODE             # POINTS: 2
        df = dfs

    # ==================================================
    # RENAME, DROP COLUMNS, ADD DATETIME AND DROP NULLS
    # ===================================================
    # rename columns to remove space the names
    # replace space with underscore
    cols_to_rename = {}
    for c in df.columns:
        if " " in c:
            cols_to_rename[c] = c.replace(" ", "_")

    #  now call the rename_sdf function
    # using df and the dictionary above
    df = YOUR CODE                      # POINTS: 1

    # this approach is also okay but you are hardcoding the column names
    # so, lets avoid hardcoding at all times
    #     df = (df.withColumnRenamed("cdr datetime", "cdrDatetime")
    #         .withColumnRenamed("last calling cellid", "cellId")
    #         .withColumnRenamed("call duration", "cellDuration"))

    # change the column "last_calling_cellid" to just cell_id
    df = YOUR CODE                      # POINTS: 1
    # drop the cdr_type and call_duration because we dont really need them
    df = YOUR CODE                      # POINTS: 1

    # lets make sure we don't have any values in the user_id
    # and cdr_datetime columns
    # use spark filter() function and isNotNull() in a nested
    # fashion to achieve this
    df = YOUR CODE          # POINTS: 2
    df = YOUR CODE          # POINTS: 2

    # Use Spark UDF to add date and datetime
    add_datetime = udf(lambda x: datetime.strptime(x, date_format), TimestampType())
    add_date = udf(lambda x: datetime.strptime(x, date_format), DateType())


    # create timestamp and date column
    df = df.withColumn('datetime', add_datetime(col('cdr_datetime')))
    df = YOUR CODE                                  # POINTS: 2

    # ==================================================
    # ADD LOCATION THROUGH JOIN
    # ===================================================
    # Lets merge with location details using cellId from CDRs and also
    # cellID on the other
    # read pandas dataframe of location details
    dfLoc = YOUR CODE                                   # POINTS: 1
    # remove duplicates from the cell_id column to make sure we only
    # unique cell_ids.Use the drop_duplicates() in pandas
    dfLoc = YOUR CODE                                       # POINTS: 1
    # create spark dataframe from the pandas dataframe
    # using the function createDataFrame from the SparkSession object
    # and the pandas DataFrame created above
    sdfLoc = YOUR CODE                                      # POINTS: 1

    # join the cdrs dataframe with the location dataframe
    # When using the join function, make sure you choose the option which allows
    # to keep all records on the cdrs side which match with records on the right.
    # please use "inner"
    # best option here. Check the docs for details:
    # https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame
    df = YOUR CODE                                          # POINTS: 3

    # Drop all records which didn't find a match in the location file because we cant do this analysis
    # without location. There are several ways to do this. For instance, you can drop all rows where
    # use spark filter and condition set to cell_id isNotNull()
    df = YOUR CODE                  # POINTS: 2

    # ==================================================
    # SAMPLE USERS
    # ===================================================
    # first, create a list of unique user user_ids
    # using the distinct() function on the userid_col
    all_users = YOUR CODE                               # POINTS: 2

    # randomly select the required number of users
    # using the random.choices() function
    random_userid_list = [i[userid_col] for i in random.choices(all_users, k=number_of_users_to_sample)]

    # Now, for each of the selected users, leets get their data
    # select only our random user data using spark filter
    # use df.filter(). Inside it, use the isin() function on the
    # userid column
    dfu = YOUR CODE                         # POINTS: 3

    # save to CSV if necessary
    if save_to_csv:
        dfu.coalesce(1).write.csv(path=output_csv, header=True)
    else:
        return dfu


def explore_data_with_spark(df=None, output_plot_file=None, output_heatmap=None):
    """
    Lets do a quick exploration of the data by generating the following:
    1. Number of days in the data
    2. User call count stats
    3. Weekday and hour calling patterns
    """
    # =====================================
    # CALCULATE THE NUMBER OF DAYS IN DATA
    # =====================================

    # use relevant spark function to generate
    # a list of unique dates, recall that the date
    # column is 'date
    dates_rows = YOUR CODE                      # POINTS: 2
    # sort the dates using sorted() function
    sorted_dates = YOUR CODE                    # POINTS: 1
    # use list indexing to get the first element and last
    # element from the sorted list, substract them to get
    # time difference
    diff = YOUR CODE                                # POINTS: 1
    # use days function to get the number of days
    num_days = YOUR CODE                            # POINTS: 1

    # =====================================
    # GENERATE WEEKDAY AND HOUR CALL COUNT
    # =====================================

    # define UDF to calculate hour and weekday
    # for weekday use weekday() function while
    # for hour, use hour()
    add_hr = YOUR CODE              # Define Spark udf to get hour from the datetime column         # POINTS: 2
    add_wkday = YOUR CODE           # Define Spark udf to get hour from the datetime column         # POINTS: 2
    # create a dictionary with keys as the weekday integers while the values
    # are the weekday name
    day_dict = YOUR CODE                                # POINTS: 1

    # add hour column, lets call it 'hr
    # also add weekday column, we call it 'wkday'
    df = YOUR CODE                                  # POINTS: 1
    df = YOUR CODE                                  # POINTS: 1
    # Group by 'wkday', 'hr' using spark group by function
    # and convert from Spark DataFrame above using toPandas()
    pdf = YOUR CODE                                 # POINTS: 1
    # use the Pandas.Series.map() function on the "wkday" column to convert
    # wkday as numbers to week day names and create  new column for weekday name
    # Lets call that column 'weekDay'
    YOUR CODE                           # POINTS: 3
    # drop the "wkday" column now
    YOUR CODE                                   # POINTS: 1
    pdf = pdf.pivot(index='weekDay', columns='hr', values='count')

    # create and save heatmap
    # use plt.figure() to create figure with your chosen size
    YOUR CODE                                       # POINTS: 1
    # use sns.heatmap() function to create heatmap using the Pandas DataFrame, pdf
    YOUR CODE                                           # POINTS: 1
    # save the figure to file
    YOUR CODE                                           # POINTS: 1

    # =====================================
    # NUMBER OF CALLS FOR EACH USER
    # =====================================
    # Use Spark groupBy function to group user and count number of events
    # convert resulting spark dataframe to pandas in the samee line of code
    df_grp_user =  YOUR CODE                                    # POINTS: 1

    # create a distribution plot of user call count using
    # seaborn distplot() function and save the figure
    # first, use plt.figure() to create new figure environment
    # Next, create the plot and then finally save it
    YOUR CODE                                                       # POINTS: 1
    YOUR CODE                                                       # POINTS: 1
    YOUR CODE                                                       # POINTS: 1

    # report average number calls per day for each user
    # first use spark groupBy on user_id and date, then
    # convert that object to pandas dataframe using toPandas()
    # function
    df_grp_day = YOUR CODE                                          # POINTS: 1

    # get mean and median
    mean = YOUR CODE                                                # POINTS: 1
    median = YOUR CODE                                              # POINTS: 1

    # return results like this mean, median, number of days
    YOUR CODE                                                       # POINTS: 1

if __name__ == '__main__':
    ### FOR THE CODE HERE,I SUGGEST WE SHOULDN'T GIVE ANY POINTS ###

    # ============================================
    # QUESTION 1: USING MAP FUNCTION
    # ============================================
    time_format = "%Y%m%d"
    reference_date = '20180713'
    #  add full path to the CSV folder and the function Path() to convert to
    # a path object
    path_to_csv = YOUR CODE
    get_events_before_jul13(path_to_csv , reference_date, time_format)

    # ============================================
    # QUESTION 2: PREPROCESS SIMULATED CDRS
    # ============================================
    cdrs_dir = ADD PATH TO FOLDER WITH MULTIPLE CSV FILES
    # please start  with a few number of users, e.g., 1000
    number_of_users_to_sample = YOUR CODE
    # Full path to CSV to write outputs
    output_csv = YOUR CODE
    # Download the file called simulated_locs.csv
    # and add its full path below
    loc_file = YOUR CODE
    # finall call the function preprocess_cdrs_using_spark and with other options left
    # as default
    dfu = YOUR CODE

    # ============================================
    # QUESTION 3: EXPLORE USER ACTIVITY PATTERNS
    # ============================================
    # 1. use the df you save from above as input here
    # 2. create a full path to save a plot with "png" extension
    # 3. create a full path to save the heatmap plot
    # Run the function explore_data_with_spark below
    results =  YOUR CODE
