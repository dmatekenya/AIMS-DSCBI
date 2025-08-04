# Complete Pandas Course: From Beginner to Advanced
# Sessions 1-10 Comprehensive Guide

## Course Overview

This notebook covers 10 progressive sessions designed to take you from pandas beginner to advanced user. Each session builds on previous concepts with hands-on practice and real-world examples.

---

# Session 1: Getting Started with Pandas

## Learning Outcomes
By the end of this session, you will be able to:
- **Understand** what pandas is and why it's essential for data analysis
- **Import** pandas and create basic data structures
- **Distinguish** between Series and DataFrame objects
- **Create** DataFrames from dictionaries and lists
- **Use** basic inspection methods to understand your data
- **Navigate** DataFrame structure including index and columns

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Pandas version:", pd.__version__)
```

### What is Pandas?
Pandas is a powerful Python library for data manipulation and analysis. Think of it as Excel for Python, but much more powerful!

```python
# Creating a simple example to show pandas power
sales_data = {
    'Product': ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard'],
    'Price': [999, 699, 399, 299, 49],
    'Units_Sold': [50, 120, 80, 30, 200],
    'Category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Accessories']
}

# This is what we'll learn to do easily with pandas:
df = pd.DataFrame(sales_data)
total_revenue = (df['Price'] * df['Units_Sold']).sum()
print(f"Total Revenue: ${total_revenue:,}")
print(f"Best selling product: {df.loc[df['Units_Sold'].idxmax(), 'Product']}")
```

### Core Data Structures

#### 1. Series (1-dimensional)
```python
# Series - like a single column in Excel
prices = pd.Series([999, 699, 399, 299, 49], name='Prices')
print("Series example:")
print(prices)
print(f"\nSeries type: {type(prices)}")
print(f"Series name: {prices.name}")
```

#### 2. DataFrame (2-dimensional)
```python
# DataFrame - like an Excel spreadsheet
products_df = pd.DataFrame(sales_data)
print("\nDataFrame example:")
print(products_df)
print(f"\nDataFrame type: {type(products_df)}")
print(f"DataFrame shape: {products_df.shape}")
```

### Creating DataFrames from Different Sources

```python
# Method 1: From dictionary
students_dict = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [23, 22, 24, 23],
    'Grade': ['A', 'B', 'A', 'C'],
    'Score': [92, 85, 94, 78]
}
students_df = pd.DataFrame(students_dict)

# Method 2: From lists
data_lists = [
    ['Alice', 23, 'A', 92],
    ['Bob', 22, 'B', 85],
    ['Charlie', 24, 'A', 94],
    ['Diana', 23, 'C', 78]
]
students_df2 = pd.DataFrame(data_lists, columns=['Name', 'Age', 'Grade', 'Score'])

print("From dictionary:")
print(students_df)
print("\nFrom lists:")
print(students_df2)
```

### Basic DataFrame Inspection
```python
# Essential methods for understanding your data
print("Shape (rows, columns):", students_df.shape)
print("\nColumn names:", students_df.columns.tolist())
print("\nIndex:", students_df.index.tolist())
print("\nFirst 2 rows:")
print(students_df.head(2))
print("\nLast 2 rows:")
print(students_df.tail(2))
print("\nDataFrame info:")
print(students_df.info())
print("\nBasic statistics:")
print(students_df.describe())
```

### Understanding Index and Columns
```python
# Index and columns are fundamental concepts
print("Index values:", students_df.index.values)
print("Column values:", students_df.columns.values)

# You can set custom index
students_custom_index = students_df.set_index('Name')
print("\nWith Name as index:")
print(students_custom_index)
```

---

# Session 2: Data Loading and Basic Exploration

## Learning Outcomes
By the end of this session, you will be able to:
- **Load** data from CSV and Excel files into pandas DataFrames
- **Handle** basic file reading parameters and options
- **Perform** initial data exploration on real datasets
- **Select** single and multiple columns from DataFrames
- **Use** basic row selection with .loc[] and .iloc[]
- **Understand** DataFrame dimensions and structure

```python
# Let's create sample files to work with
def create_sample_datasets():
    """Create sample CSV and Excel files for practice"""
    
    # Sample employee data
    employee_data = {
        'employee_id': range(101, 121),
        'name': ['John Smith', 'Jane Doe', 'Mike Johnson', 'Sarah Wilson', 'David Brown',
                'Lisa Davis', 'Tom Miller', 'Amy Garcia', 'Chris Lee', 'Maria Rodriguez',
                'James Taylor', 'Linda Martinez', 'Robert Anderson', 'Patricia Thomas',
                'Michael Jackson', 'Jennifer White', 'William Harris', 'Elizabeth Clark',
                'Richard Lewis', 'Susan Robinson'],
        'department': ['IT', 'HR', 'Finance', 'IT', 'Marketing', 'Finance', 'IT', 'HR',
                      'Marketing', 'Finance', 'IT', 'HR', 'Finance', 'Marketing', 'IT',
                      'HR', 'Finance', 'Marketing', 'IT', 'HR'],
        'salary': [75000, 65000, 70000, 80000, 60000, 72000, 78000, 63000, 58000, 74000,
                  82000, 64000, 71000, 59000, 79000, 66000, 73000, 61000, 81000, 67000],
        'years_experience': [5, 3, 7, 6, 2, 8, 4, 3, 1, 9, 7, 4, 6, 2, 5, 3, 8, 2, 6, 4],
        'hire_date': pd.date_range('2018-01-01', periods=20, freq='90D')
    }
    
    # Create CSV file
    employee_df = pd.DataFrame(employee_data)
    employee_df.to_csv('employees.csv', index=False)
    
    # Create Excel file with multiple sheets
    with pd.ExcelWriter('company_data.xlsx', engine='openpyxl') as writer:
        employee_df.to_excel(writer, sheet_name='Employees', index=False)
        
        # Sales data for second sheet
        sales_data = pd.DataFrame({
            'month': pd.date_range('2023-01-01', periods=12, freq='M'),
            'revenue': np.random.randint(100000, 200000, 12),
            'expenses': np.random.randint(70000, 120000, 12)
        })
        sales_data.to_excel(writer, sheet_name='Sales', index=False)
    
    print("Sample files created: employees.csv and company_data.xlsx")
    return employee_df

# Create our sample data
sample_df = create_sample_datasets()
```

### Loading Data from Files
```python
# Reading CSV files
employees = pd.read_csv('employees.csv')
print("Loaded from CSV:")
print(employees.head())

# Reading Excel files
employees_excel = pd.read_excel('company_data.xlsx', sheet_name='Employees')
sales_excel = pd.read_excel('company_data.xlsx', sheet_name='Sales')

print("\nLoaded employees from Excel:")
print(employees_excel.head(3))
print("\nLoaded sales from Excel:")
print(sales_excel.head(3))

# Common parameters for read_csv
employees_custom = pd.read_csv('employees.csv', 
                              parse_dates=['hire_date'],  # Convert to datetime
                              index_col='employee_id')    # Use as index
print("\nWith custom parameters:")
print(employees_custom.head(3))
```

### Initial Data Exploration
```python
# Always start with these exploration steps
print("Dataset shape:", employees.shape)
print("\nColumn data types:")
print(employees.dtypes)
print("\nBasic statistics:")
print(employees.describe())
print("\nFirst few rows:")
print(employees.head())
print("\nDataset info:")
employees.info()
```

### Basic Column Selection
```python
# Selecting single columns
names = employees['name']
print("Single column (Series):")
print(type(names))
print(names.head())

# Selecting multiple columns
basic_info = employees[['name', 'department', 'salary']]
print("\nMultiple columns (DataFrame):")
print(type(basic_info))
print(basic_info.head())

# Different ways to select columns
print("\nDifferent selection methods:")
print("Dot notation:", type(employees.name))  # Works for valid Python names
print("Bracket notation:", type(employees['name']))
```

### Basic Row Selection
```python
# Using .loc[] - label-based selection
print("Using .loc[] - first 3 rows:")
print(employees.loc[0:2])  # Includes end index

print("\nUsing .loc[] - specific rows and columns:")
print(employees.loc[0:2, ['name', 'salary']])

# Using .iloc[] - position-based selection
print("\nUsing .iloc[] - first 3 rows:")
print(employees.iloc[0:3])  # Excludes end index

print("\nUsing .iloc[] - specific positions:")
print(employees.iloc[0:3, [1, 3]])  # First 3 rows, columns 1 and 3
```

---

# Session 3: Data Selection and Filtering

## Learning Outcomes
By the end of this session, you will be able to:
- **Create** and apply boolean masks for data filtering
- **Use** comparison operators to filter data based on conditions
- **Combine** multiple conditions using logical operators
- **Apply** string methods for text-based filtering
- **Filter** DataFrames using complex conditional logic
- **Select** specific subsets of data based on multiple criteria

```python
# Let's work with our employee data for filtering examples
employees = pd.read_csv('employees.csv')
print("Working with employee dataset:")
print(employees.head())
print(f"\nDataset shape: {employees.shape}")
```

### Boolean Indexing Basics
```python
# Creating boolean masks
high_salary_mask = employees['salary'] > 70000
print("Boolean mask for high salary:")
print(high_salary_mask.head(10))
print(f"Mask type: {type(high_salary_mask)}")

# Applying the mask to filter data
high_salary_employees = employees[high_salary_mask]
print(f"\nEmployees with salary > $70,000:")
print(high_salary_employees[['name', 'salary', 'department']])
print(f"Count: {len(high_salary_employees)} out of {len(employees)}")
```

### Comparison Operators
```python
# Different comparison operators
print("IT Department employees:")
it_employees = employees[employees['department'] == 'IT']
print(it_employees[['name', 'department', 'salary']])

print("\nExperienced employees (>= 5 years):")
experienced = employees[employees['years_experience'] >= 5]
print(experienced[['name', 'years_experience', 'salary']])

print("\nNot in HR department:")
non_hr = employees[employees['department'] != 'HR']
print(f"Non-HR employees: {len(non_hr)} out of {len(employees)}")

# Using .isin() for multiple values
target_departments = ['IT', 'Finance']
it_finance = employees[employees['department'].isin(target_departments)]
print(f"\nIT and Finance employees: {len(it_finance)}")
print(it_finance[['name', 'department', 'salary']])
```

### Combining Multiple Conditions
```python
# Using & (and) operator
high_paid_it = employees[(employees['salary'] > 75000) & (employees['department'] == 'IT')]
print("High-paid IT employees:")
print(high_paid_it[['name', 'department', 'salary']])

# Using | (or) operator
hr_or_marketing = employees[(employees['department'] == 'HR') | (employees['department'] == 'Marketing')]
print(f"\nHR or Marketing employees: {len(hr_or_marketing)}")

# Complex conditions
senior_well_paid = employees[
    (employees['years_experience'] >= 6) & 
    (employees['salary'] >= 70000) & 
    (employees['department'].isin(['IT', 'Finance']))
]
print(f"\nSenior, well-paid IT/Finance employees:")
print(senior_well_paid[['name', 'department', 'years_experience', 'salary']])

# Using ~ (not) operator
not_entry_level = employees[~(employees['years_experience'] <= 2)]
print(f"\nNon-entry level employees: {len(not_entry_level)}")
```

### String Methods for Filtering
```python
# Working with text data
print("Names containing 'John':")
john_names = employees[employees['name'].str.contains('John', case=False)]
print(john_names[['name', 'department']])

print("\nNames starting with 'M':")
m_names = employees[employees['name'].str.startswith('M')]
print(m_names[['name', 'department']])

print("\nNames ending with 'son':")
son_names = employees[employees['name'].str.endswith('son')]
print(son_names[['name', 'department']])

# String length filtering
print("\nEmployees with short names (< 10 characters):")
short_names = employees[employees['name'].str.len() < 10]
print(short_names[['name', 'department']])

# Case-insensitive filtering
print("\nDepartments containing 'it' (case-insensitive):")
it_like = employees[employees['department'].str.contains('it', case=False)]
print(it_like[['name', 'department']])
```

### Advanced Filtering Techniques
```python
# Using query() method for readable conditions
high_performers = employees.query('salary > 70000 and years_experience >= 5')
print("High performers using query():")
print(high_performers[['name', 'salary', 'years_experience']])

# Filtering with calculated conditions
employees['salary_per_year_exp'] = employees['salary'] / employees['years_experience']
efficient_employees = employees[employees['salary_per_year_exp'] > 12000]
print("\nEmployees with high salary per year of experience:")
print(efficient_employees[['name', 'salary', 'years_experience', 'salary_per_year_exp']])

# Filtering based on rank/percentile
salary_90th_percentile = employees['salary'].quantile(0.9)
top_10_percent_salary = employees[employees['salary'] >= salary_90th_percentile]
print(f"\nTop 10% earners (salary >= ${salary_90th_percentile:,.0f}):")
print(top_10_percent_salary[['name', 'salary', 'department']])
```

---

# Session 4: Data Cleaning Basics

## Learning Outcomes
By the end of this session, you will be able to:
- **Identify** and handle missing data using various strategies
- **Detect** and remove duplicate records from DataFrames
- **Convert** data types appropriately for analysis
- **Rename** columns for better readability and consistency
- **Apply** basic data validation techniques
- **Clean** datasets to prepare them for analysis

```python
# Create a messy dataset to practice cleaning
def create_messy_dataset():
    """Create a dataset with common data quality issues"""
    
    messy_data = {
        'Customer_ID': [1, 2, 3, 4, 5, 5, 7, 8, 9, 10],  # Duplicate ID
        'customer name': ['John Doe', 'jane smith', None, 'Bob Johnson', 'Alice Brown',
                         'Alice Brown', 'Charlie Wilson', '', 'Diana Prince', 'Eve Adams'],
        'age': [25, 30, 35, None, 28, 28, 45, 22, 33, 29],
        'email': ['john@email.com', 'JANE@EMAIL.COM', 'invalid_email', 'bob@email.com',
                 None, 'alice@email.com', 'charlie@email.com', 'diana@email.com', 
                 'diana@email.com', 'eve@email.com'],
        'purchase_amount': ['100.50', '250.75', 'invalid', '75.25', '300.00',
                           '300.00', '150.25', '90.50', '200.75', '125.50'],
        'signup_date': ['2023-01-15', '2023/02/20', '2023-03-25', None, '2023-04-10',
                       '2023-04-10', 'invalid_date', '2023-06-05', '2023-07-12', '2023-08-18'],
        'status': ['active', 'ACTIVE', 'inactive', 'Active', 'INACTIVE', 'INACTIVE', 
                  'active', 'Active', 'inactive', 'active']
    }
    
    return pd.DataFrame(messy_data)

messy_df = create_messy_dataset()
print("Messy dataset for cleaning practice:")
print(messy_df)
print(f"\nDataset info:")
messy_df.info()
```

### Identifying and Handling Missing Data
```python
# Check for missing values
print("Missing values per column:")
print(messy_df.isnull().sum())

print("\nPercentage of missing values:")
missing_percentages = (messy_df.isnull().sum() / len(messy_df) * 100).round(2)
print(missing_percentages)

# Visualize missing data pattern
print("\nMissing data pattern:")
print(messy_df.isnull())

# Different strategies for handling missing values

# Strategy 1: Drop rows with any missing values
df_dropped_rows = messy_df.dropna()
print(f"\nAfter dropping rows with missing values: {len(df_dropped_rows)} rows remain")

# Strategy 2: Drop columns with missing values
df_dropped_cols = messy_df.dropna(axis=1)
print(f"After dropping columns with missing values: {df_dropped_cols.shape[1]} columns remain")

# Strategy 3: Fill missing values
df_filled = messy_df.copy()

# Fill with specific values
df_filled['customer name'] = df_filled['customer name'].fillna('Unknown')
df_filled['age'] = df_filled['age'].fillna(df_filled['age'].median())
df_filled['email'] = df_filled['email'].fillna('no_email@unknown.com')

print("\nAfter filling missing values:")
print(df_filled[['customer name', 'age', 'email']].head(10))

# Forward fill and backward fill
df_filled['signup_date'] = pd.to_datetime(df_filled['signup_date'], errors='coerce')
df_filled['signup_date'] = df_filled['signup_date'].fillna(method='ffill')  # Forward fill

print("\nAfter forward filling dates:")
print(df_filled['signup_date'])
```

### Detecting and Removing Duplicates
```python
# Check for duplicates
print("Duplicate rows:")
duplicates = messy_df.duplicated()
print(f"Number of duplicate rows: {duplicates.sum()}")

# Show duplicate rows
duplicate_rows = messy_df[duplicates]
print("\nDuplicate rows:")
print(duplicate_rows)

# Check duplicates based on specific columns
duplicate_customers = messy_df.duplicated(subset=['Customer_ID'], keep=False)
print(f"\nRows with duplicate Customer_IDs: {duplicate_customers.sum()}")
print(messy_df[duplicate_customers])

# Remove duplicates
df_no_duplicates = messy_df.drop_duplicates()
print(f"\nAfter removing exact duplicates: {len(df_no_duplicates)} rows remain")

# Remove duplicates based on specific columns (keep first occurrence)
df_unique_customers = messy_df.drop_duplicates(subset=['Customer_ID'], keep='first')
print(f"After removing duplicate Customer_IDs: {len(df_unique_customers)} rows remain")
```

### Data Type Conversion
```python
# Check current data types
print("Current data types:")
print(messy_df.dtypes)

# Convert data types
df_typed = messy_df.copy()

# Convert purchase_amount to numeric (handling invalid values)
df_typed['purchase_amount'] = pd.to_numeric(df_typed['purchase_amount'], errors='coerce')

# Convert signup_date to datetime
df_typed['signup_date'] = pd.to_datetime(df_typed['signup_date'], errors='coerce')

# Convert Customer_ID to integer (after handling missing values)
df_typed['Customer_ID'] = df_typed['Customer_ID'].astype('int64')

# Convert status to category for memory efficiency
df_typed['status'] = df_typed['status'].astype('category')

print("\nAfter type conversion:")
print(df_typed.dtypes)

# Show the cleaned numeric column
print("\nCleaned purchase amounts:")
print(df_typed[['Customer_ID', 'purchase_amount']].head(10))
```

### Renaming Columns
```python
# Current column names
print("Current columns:", messy_df.columns.tolist())

# Rename specific columns
df_renamed = messy_df.rename(columns={
    'customer name': 'customer_name',
    'Customer_ID': 'customer_id'
})

print("After renaming specific columns:", df_renamed.columns.tolist())

# Rename all columns to lowercase and replace spaces
df_renamed.columns = df_renamed.columns.str.lower().str.replace(' ', '_')
print("After standardizing all columns:", df_renamed.columns.tolist())

# Rename using a mapping dictionary
column_mapping = {
    'customer_id': 'id',
    'customer_name': 'name',
    'purchase_amount': 'amount',
    'signup_date': 'date_joined'
}

df_final_names = df_renamed.rename(columns=column_mapping)
print("With business-friendly names:", df_final_names.columns.tolist())
```

### Comprehensive Data Cleaning Pipeline
```python
def clean_customer_data(df):
    """Comprehensive cleaning function"""
    
    # Make a copy to avoid modifying original
    cleaned_df = df.copy()
    
    # 1. Standardize column names
    cleaned_df.columns = cleaned_df.columns.str.lower().str.replace(' ', '_')
    
    # 2. Remove exact duplicates
    cleaned_df = cleaned_df.drop_duplicates()
    
    # 3. Handle missing values
    cleaned_df['customer_name'] = cleaned_df['customer_name'].fillna('Unknown')
    cleaned_df['age'] = cleaned_df['age'].fillna(cleaned_df['age'].median())
    cleaned_df['email'] = cleaned_df['email'].fillna('no_email@unknown.com')
    
    # 4. Clean and convert data types
    cleaned_df['purchase_amount'] = pd.to_numeric(cleaned_df['purchase_amount'], errors='coerce')
    cleaned_df['signup_date'] = pd.to_datetime(cleaned_df['signup_date'], errors='coerce')
    
    # 5. Standardize text data
    cleaned_df['customer_name'] = cleaned_df['customer_name'].str.title()
    cleaned_df['email'] = cleaned_df['email'].str.lower()
    cleaned_df['status'] = cleaned_df['status'].str.lower()
    
    # 6. Remove rows with critical missing data
    cleaned_df = cleaned_df.dropna(subset=['customer_id', 'purchase_amount'])
    
    # 7. Remove duplicates based on customer_id
    cleaned_df = cleaned_df.drop_duplicates(subset=['customer_id'], keep='first')
    
    return cleaned_df

# Apply comprehensive cleaning
cleaned_data = clean_customer_data(messy_df)
print("Cleaned dataset:")
print(cleaned_data)
print(f"\nOriginal shape: {messy_df.shape}")
print(f"Cleaned shape: {cleaned_data.shape}")
print(f"\nCleaned data types:")
print(cleaned_data.dtypes)
```

---

# Session 5: Data Transformation

## Learning Outcomes
By the end of this session, you will be able to:
- **Create** new columns using mathematical operations and calculations
- **Apply** custom functions to transform data using .apply()
- **Manipulate** string data using pandas string methods
- **Work** with date/time data for temporal analysis
- **Transform** existing columns to create meaningful derived features
- **Use** vectorized operations for efficient data processing

```python
# Create a sample dataset for transformation practice
def create_sales_dataset():
    """Create a sample sales dataset for transformation examples"""
    
    np.random.seed(42)
    
    sales_data = {
        'order_id': range(1001, 1101),
        'customer_name': ['John Smith', 'Jane Doe', 'Mike Johnson'] * 33 + ['Sarah Wilson'],
        'product_name': ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard'] * 20,
        'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Accessories'] * 20,
        'unit_price': np.random.uniform(50, 1000, 100).round(2),
        'quantity': np.random.randint(1, 10, 100),
        'order_date': pd.date_range('2023-01-01', periods=100, freq='3D'),
        'customer_email': ['john@email.com', 'jane@email.com', 'mike@email.com'] * 33 + ['sarah@email.com'],
        'shipping_address': ['123 Main St, New York, NY', '456 Oak Ave, Los Angeles, CA', 
                           '789 Pine Rd, Chicago, IL'] * 33 + ['321 Elm St, Houston, TX']
    }
    
    return pd.DataFrame(sales_data)

sales_df = create_sales_dataset()
print("Sales dataset for transformation:")
print(sales_df.head())
print(f"Dataset shape: {sales_df.shape}")
```

### Creating New Columns with Calculations
```python
# Basic mathematical operations
sales_df['total_amount'] = sales_df['unit_price'] * sales_df['quantity']
sales_df['discount_5_percent'] = sales_df['total_amount'] * 0.05
sales_df['final_amount'] = sales_df['total_amount'] - sales_df['discount_5_percent']

print("New calculated columns:")
print(sales_df[['unit_price', 'quantity', 'total_amount', 'final_amount']].head())

# Conditional calculations
sales_df['order_size'] = sales_df['quantity'].apply(
    lambda x: 'Large' if x >= 7 else 'Medium' if x >= 4 else 'Small'
)

# Using np.where for conditional logic
sales_df['price_category'] = np.where(
    sales_df['unit_price'] >= 500, 'Premium',
    np.where(sales_df['unit_price'] >= 200, 'Standard', 'Budget')
)

print("\nConditional columns:")
print(sales_df[['quantity', 'order_size', 'unit_price', 'price_category']].head(10))

# Multiple conditions with np.select
conditions = [
    (sales_df['total_amount'] >= 1000) & (sales_df['quantity'] >= 5),
    (sales_df['total_amount'] >= 500) & (sales_df['quantity'] >= 3),
    sales_df['total_amount'] >= 200
]
choices = ['VIP Order', 'Standard Order', 'Regular Order']
sales_df['order_type'] = np.select(conditions, choices, default='Small Order')

print("\nOrder type classification:")
print(sales_df[['total_amount', 'quantity', 'order_type']].value_counts('order_type'))
```

### Using Apply() for Custom Functions
```python
# Simple apply with lambda
sales_df['price_per_letter'] = sales_df.apply(
    lambda row: row['unit_price'] / len(row['product_name']), axis=1
)

# Custom function for complex logic
def calculate_shipping_cost(row):
    """Calculate shipping cost based on order value and quantity"""
    base_cost = 10
    if row['total_amount'] > 500:
        return 0  # Free shipping for orders over $500
    elif row['quantity'] > 5:
        return base_cost * 0.5  # 50% discount for bulk orders
    else:
        return base_cost

sales_df['shipping_cost'] = sales_df.apply(calculate_shipping_cost, axis=1)

print("Custom calculations with apply:")
print(sales_df[['total_amount', 'quantity', 'shipping_cost']].head())

# Apply to specific columns only
def categorize_product(product_name):
    """Categorize products based on name"""
    product_name = product_name.lower()
    if 'laptop' in product_name or 'computer' in product_name:
        return 'Computing'
    elif 'phone' in product_name or 'tablet' in product_name:
        return 'Mobile'
    elif 'monitor' in product_name or 'screen' in product_name:
        return 'Display'
    else:
        return 'Other'

sales_df['product_category'] = sales_df['product_name'].apply(categorize_product)

print("\nProduct categorization:")
print(sales_df[['product_name', 'product_category']].head())
```

### String Manipulation with .str Methods
```python
# String cleaning and transformation
print("Original customer names:")
print(sales_df['customer_name'].head())

# Case conversion
sales_df['customer_name_upper'] = sales_df['customer_name'].str.upper()
sales_df['customer_name_lower'] = sales_df['customer_name'].str.lower()
sales_df['customer_name_title'] = sales_df['customer_name'].str.title()

# String splitting
sales_df[['first_name', 'last_name']] = sales_df['customer_name'].str.split(' ', expand=True)

print("\nString transformations:")
print(sales_df[['customer_name', 'first_name', 'last_name']].head())

# Extract email domain
sales_df['email_domain'] = sales_df['customer_email'].str.split('@').str[1]

# Extract state from address
sales_df['state'] = sales_df['shipping_address'].str.split(', ').str[-1]

print("\nExtracted information:")
print(sales_df[['customer_email', 'email_domain', 'shipping_address', 'state']].head())

# String length and character operations
sales_df['product_