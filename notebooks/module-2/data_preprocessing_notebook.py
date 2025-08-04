# Data Preprocessing with Pandas: A Comprehensive Guide

## Learning Outcomes

By the end of this notebook, participants will be able to:

- **Load large datasets efficiently** using chunking and memory optimization techniques
- **Clean and standardize messy data types** including strings, numbers, and dates
- **Ingest complex data formats** such as JSON, Excel with multiple sheets, and nested structures
- **Perform advanced data transformations** including pivoting, melting, and custom functions
- **Combine multiple datasets** using merging, concatenation, and advanced join operations
- **Generate new variables** through feature engineering and calculated columns
- **Apply best practices** for data preprocessing workflows in real-world scenarios

---

## Section 1: Loading Large Files Efficiently

### Challenge: Working with datasets that are too large for memory

When dealing with large datasets (>1GB), loading everything into memory at once can crash your system or take too long. Here's how to handle it efficiently.

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Simulate creating a large dataset for demonstration
def create_large_dataset(filename='large_sales_data.csv', n_rows=1000000):
    """Create a simulated large sales dataset"""
    np.random.seed(42)
    
    data = {
        'transaction_id': range(1, n_rows + 1),
        'customer_id': np.random.randint(1, 50000, n_rows),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Home'], n_rows),
        'sales_amount': np.random.uniform(10, 1000, n_rows).round(2),
        'transaction_date': pd.date_range('2020-01-01', periods=n_rows, freq='1min'),
        'store_location': np.random.choice(['New York', 'London', 'Tokyo', 'Paris', 'Sydney'], n_rows),
        'payment_method': np.random.choice(['Credit', 'Debit', 'Cash', 'Digital'], 1000),
        'num_purchases': np.random.poisson(5, 1000),
        'satisfaction_score': np.random.uniform(1, 10, 1000).round(1)
    })
    
    print("Base dataset:")
    print(base_data.head())
    print(f"Shape: {base_data.shape}")
    
    return base_data

def create_demographic_features(df):
    """Create demographic-based features"""
    
    df_features = df.copy()
    
    # Age groups
    df_features['age_group'] = pd.cut(
        df_features['age'], 
        bins=[0, 25, 35, 50, 65, 100], 
        labels=['18-25', '26-35', '36-50', '51-65', '65+']
    )
    
    # Income brackets
    df_features['income_bracket'] = pd.cut(
        df_features['income'],
        bins=[0, 30000, 50000, 75000, 100000, float('inf')],
        labels=['Low', 'Lower-Mid', 'Mid', 'Upper-Mid', 'High']
    )
    
    # Tenure categories
    df_features['tenure_category'] = pd.cut(
        df_features['tenure_months'],
        bins=[0, 6, 12, 24, 36, float('inf')],
        labels=['New', 'Recent', 'Established', 'Veteran', 'Loyal']
    )
    
    # Customer life stage (combination of age and income)
    def determine_life_stage(row):
        age = row['age']
        income = row['income']
        
        if age < 30 and income < 40000:
            return 'Young_Starter'
        elif age < 30 and income >= 40000:
            return 'Young_Professional'
        elif 30 <= age < 50 and income < 60000:
            return 'Mid_Moderate'
        elif 30 <= age < 50 and income >= 60000:
            return 'Mid_Affluent'
        elif age >= 50 and income < 50000:
            return 'Senior_Conservative'
        else:
            return 'Senior_Affluent'
    
    df_features['life_stage'] = df_features.apply(determine_life_stage, axis=1)
    
    print("Demographic features created:")
    print(df_features[['age', 'age_group', 'income', 'income_bracket', 'life_stage']].head())
    
    return df_features

def create_behavioral_features(df):
    """Create behavior-based features"""
    
    df_features = df.copy()
    
    # Purchase behavior features
    df_features['avg_purchase_amount'] = df_features['purchase_amount'] / df_features['num_purchases']
    df_features['purchase_frequency'] = df_features['num_purchases'] / df_features['tenure_months']
    df_features['total_spent'] = df_features['purchase_amount'] * df_features['num_purchases']
    
    # Recency features (days since last purchase)
    current_date = pd.Timestamp('2024-01-01')
    df_features['days_since_last_purchase'] = (current_date - df_features['last_purchase_date']).dt.days
    
    # RFM-like features (Recency, Frequency, Monetary)
    df_features['recency_score'] = pd.qcut(df_features['days_since_last_purchase'], 5, labels=[5,4,3,2,1])
    df_features['frequency_score'] = pd.qcut(df_features['num_purchases'].rank(method='first'), 5, labels=[1,2,3,4,5])
    df_features['monetary_score'] = pd.qcut(df_features['total_spent'].rank(method='first'), 5, labels=[1,2,3,4,5])
    
    # Customer value segment
    df_features['customer_segment'] = (
        df_features['recency_score'].astype(str) + 
        df_features['frequency_score'].astype(str) + 
        df_features['monetary_score'].astype(str)
    )
    
    # High-value customer flag
    df_features['is_high_value'] = (
        (df_features['total_spent'] > df_features['total_spent'].quantile(0.8)) &
        (df_features['satisfaction_score'] >= 7)
    )
    
    # Purchase pattern
    def categorize_purchase_pattern(row):
        freq = row['purchase_frequency']
        avg_amount = row['avg_purchase_amount']
        
        if freq > 0.5 and avg_amount < 100:
            return 'Frequent_Small'
        elif freq > 0.5 and avg_amount >= 100:
            return 'Frequent_Large'
        elif freq <= 0.5 and avg_amount < 100:
            return 'Occasional_Small'
        else:
            return 'Occasional_Large'
    
    df_features['purchase_pattern'] = df_features.apply(categorize_purchase_pattern, axis=1)
    
    print("Behavioral features created:")
    print(df_features[['purchase_frequency', 'customer_segment', 'is_high_value', 'purchase_pattern']].head())
    
    return df_features

def create_interaction_features(df):
    """Create interaction and polynomial features"""
    
    df_features = df.copy()
    
    # Interaction features
    df_features['age_income_interaction'] = df_features['age'] * df_features['income'] / 1000000
    df_features['tenure_satisfaction_interaction'] = df_features['tenure_months'] * df_features['satisfaction_score']
    
    # Polynomial features
    df_features['age_squared'] = df_features['age'] ** 2
    df_features['income_log'] = np.log1p(df_features['income'])  # log(1+x) to handle zeros
    
    # Ratio features
    df_features['income_per_age'] = df_features['income'] / df_features['age']
    df_features['spending_rate'] = df_features['total_spent'] / df_features['income']
    
    # Normalization/Standardization examples
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    # Z-score normalization
    scaler = StandardScaler()
    df_features['income_standardized'] = scaler.fit_transform(df_features[['income']])
    
    # Min-Max scaling
    minmax_scaler = MinMaxScaler()
    df_features['age_normalized'] = minmax_scaler.fit_transform(df_features[['age']])
    
    print("Interaction and polynomial features created:")
    print(df_features[['age_income_interaction', 'income_log', 'spending_rate', 'income_standardized']].head())
    
    return df_features

def create_time_based_features(df):
    """Create time-based features"""
    
    df_features = df.copy()
    
    # Extract date components
    df_features['last_purchase_year'] = df_features['last_purchase_date'].dt.year
    df_features['last_purchase_month'] = df_features['last_purchase_date'].dt.month
    df_features['last_purchase_quarter'] = df_features['last_purchase_date'].dt.quarter
    df_features['last_purchase_dayofweek'] = df_features['last_purchase_date'].dt.dayofweek
    df_features['last_purchase_hour'] = df_features['last_purchase_date'].dt.hour
    
    # Cyclical encoding for periodic features
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['last_purchase_month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['last_purchase_month'] / 12)
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['last_purchase_hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['last_purchase_hour'] / 24)
    
    # Season encoding
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df_features['purchase_season'] = df_features['last_purchase_month'].apply(get_season)
    
    # Business vs weekend
    df_features['is_weekend_purchase'] = df_features['last_purchase_dayofweek'].isin([5, 6])
    
    print("Time-based features created:")
    print(df_features[['last_purchase_date', 'last_purchase_quarter', 'purchase_season', 'is_weekend_purchase']].head())
    
    return df_features

def create_categorical_encodings(df):
    """Demonstrate different categorical encoding techniques"""
    
    df_features = df.copy()
    
    # 1. One-Hot Encoding
    category_dummies = pd.get_dummies(df_features['product_category'], prefix='category')
    payment_dummies = pd.get_dummies(df_features['payment_method'], prefix='payment')
    
    # Combine with original dataframe
    df_features = pd.concat([df_features, category_dummies, payment_dummies], axis=1)
    
    # 2. Label Encoding (ordinal)
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    df_features['product_category_encoded'] = le.fit_transform(df_features['product_category'])
    
    # 3. Target Encoding (mean of target variable by category)
    # Using satisfaction_score as pseudo-target
    target_encoding = df_features.groupby('product_category')['satisfaction_score'].mean()
    df_features['category_target_encoded'] = df_features['product_category'].map(target_encoding)
    
    # 4. Frequency Encoding
    category_counts = df_features['product_category'].value_counts()
    df_features['category_frequency'] = df_features['product_category'].map(category_counts)
    
    print("Categorical encoding examples:")
    print(f"One-hot encoded columns: {category_dummies.columns.tolist()}")
    print(df_features[['product_category', 'product_category_encoded', 'category_target_encoded', 'category_frequency']].head())
    
    return df_features

# Execute feature engineering pipeline
base_data = feature_engineering_examples()
demographic_features = create_demographic_features(base_data)
behavioral_features = create_behavioral_features(demographic_features)
interaction_features = create_interaction_features(behavioral_features)
time_features = create_time_based_features(interaction_features)
final_features = create_categorical_encodings(time_features)

print(f"\nFinal dataset shape: {final_features.shape}")
print(f"Original features: {base_data.shape[1]}")
print(f"Features created: {final_features.shape[1] - base_data.shape[1]}")
```

---

## Section 7: Best Practices and Workflow Optimization

### Challenge: Creating efficient, reproducible data preprocessing workflows

```python
def create_preprocessing_pipeline():
    """Create a reusable preprocessing pipeline"""
    
    class DataPreprocessor:
        def __init__(self):
            self.fitted_scalers = {}
            self.fitted_encoders = {}
            self.feature_columns = None
            
        def fit_transform(self, df):
            """Fit preprocessing steps and transform data"""
            return self._fit(df)._transform(df)
        
        def transform(self, df):
            """Transform new data using fitted parameters"""
            if self.feature_columns is None:
                raise ValueError("Pipeline must be fitted first")
            return self._transform(df)
        
        def _fit(self, df):
            """Fit preprocessing parameters"""
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            
            # Identify numeric and categorical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Fit scalers for numeric columns
            for col in numeric_cols:
                if col != 'customer_id':  # Don't scale ID columns
                    scaler = StandardScaler()
                    scaler.fit(df[[col]])
                    self.fitted_scalers[col] = scaler
            
            # Fit encoders for categorical columns
            for col in categorical_cols:
                encoder = LabelEncoder()
                encoder.fit(df[col].astype(str))
                self.fitted_encoders[col] = encoder
            
            self.feature_columns = df.columns.tolist()
            return self
        
        def _transform(self, df):
            """Transform data using fitted parameters"""
            df_transformed = df.copy()
            
            # Apply scaling
            for col, scaler in self.fitted_scalers.items():
                if col in df_transformed.columns:
                    df_transformed[f'{col}_scaled'] = scaler.transform(df_transformed[[col]])
            
            # Apply encoding
            for col, encoder in self.fitted_encoders.items():
                if col in df_transformed.columns:
                    df_transformed[f'{col}_encoded'] = encoder.transform(df_transformed[col].astype(str))
            
            return df_transformed
        
        def get_feature_info(self):
            """Get information about fitted features"""
            return {
                'scaled_features': list(self.fitted_scalers.keys()),
                'encoded_features': list(self.fitted_encoders.keys()),
                'total_features': len(self.feature_columns) if self.feature_columns else 0
            }
    
    # Example usage
    sample_data = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'age': [25, 35, 45, 55, 65],
        'income': [30000, 50000, 70000, 90000, 110000],
        'category': ['A', 'B', 'A', 'C', 'B']
    })
    
    # Fit and transform
    preprocessor = DataPreprocessor()
    transformed_data = preprocessor.fit_transform(sample_data)
    
    print("Original data:")
    print(sample_data)
    print("\nTransformed data:")
    print(transformed_data)
    print("\nFeature info:")
    print(preprocessor.get_feature_info())
    
    return preprocessor

def data_quality_checker():
    """Create comprehensive data quality checking functions"""
    
    def check_data_quality(df, report_name="Data Quality Report"):
        """Comprehensive data quality assessment"""
        
        print(f"\n{'='*50}")
        print(f"{report_name}")
        print(f"{'='*50}")
        
        # Basic info
        print(f"Dataset Shape: {df.shape}")
        print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values analysis
        print(f"\n{'Missing Values Analysis':-<30}")
        missing_stats = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum(),
            'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2),
            'Data_Type': df.dtypes
        })
        missing_stats = missing_stats[missing_stats['Missing_Count'] > 0]
        if not missing_stats.empty:
            print(missing_stats.to_string(index=False))
        else:
            print("No missing values found!")
        
        # Duplicate analysis
        print(f"\n{'Duplicate Analysis':-<30}")
        duplicate_count = df.duplicated().sum()
        print(f"Duplicate Rows: {duplicate_count} ({duplicate_count/len(df)*100:.2f}%)")
        
        # Data type analysis
        print(f"\n{'Data Types Summary':-<30}")
        dtype_summary = df.dtypes.value_counts()
        print(dtype_summary)
        
        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n{'Numeric Columns Statistics':-<30}")
            print(df[numeric_cols].describe())
            
            # Check for outliers using IQR method
            print(f"\n{'Outlier Detection (IQR Method)':-<30}")
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print(f"\n{'Categorical Columns Analysis':-<30}")
            for col in categorical_cols:
                unique_count = df[col].nunique()
                print(f"{col}: {unique_count} unique values")
                if unique_count <= 10:  # Show value counts for low cardinality
                    print(f"  Value counts: {df[col].value_counts().to_dict()}")
        
        return {
            'missing_stats': missing_stats,
            'duplicate_count': duplicate_count,
            'outlier_info': 'See output above'
        }
    
    # Example usage
    sample_data_with_issues = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 5],  # Duplicate
        'age': [25, 35, np.nan, 55, 150],  # Missing value and outlier
        'income': [30000, 50000, 70000, 90000, 110000, 95000],
        'category': ['A', 'B', None, 'C', 'B', 'A'],  # Missing value
        'score': [85, 92, 78, 88, 91, 89]
    })
    
    quality_report = check_data_quality(sample_data_with_issues, "Sample Data Quality Check")
    return quality_report

# Execute pipeline and quality check examples
preprocessor = create_preprocessing_pipeline()
quality_report = data_quality_checker()
```

---

## Summary and Key Takeaways

### What We've Covered

1. **Large File Processing**: Chunking strategies and memory optimization
2. **Data Type Cleaning**: Converting messy strings, numbers, dates, and booleans
3. **Complex Data Ingestion**: JSON flattening and multi-sheet Excel processing
4. **Advanced Transformations**: Pivoting, melting, and window functions
5. **Data Combination**: Various join types and concatenation strategies
6. **Feature Engineering**: Creating demographic, behavioral, and interaction features
7. **Best Practices**: Reusable pipelines and data quality assessment

### Next Steps for Practice

1. **Apply these techniques to your own datasets**
2. **Experiment with different parameters and approaches**
3. **Build your own preprocessing pipeline for your specific use case**
4. **Practice combining multiple techniques in a single workflow**
5. **Focus on reproducibility and code organization**

### Additional Resources

- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **Data Cleaning Best Practices**: Focus on understanding your data first
- **Performance Optimization**: Use appropriate data types and chunking for large datasets
- **Feature Engineering**: Domain knowledge is key to creating meaningful features

---

*This notebook provides a comprehensive foundation for data preprocessing with pandas. Each section can be adapted and extended based on your specific data challenges and requirements.*.random.choice(['Credit Card', 'Cash', 'Debit Card', 'Mobile Pay'], n_rows)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Created large dataset: {filename} with {n_rows:,} rows")
    return filename

# Create our large dataset
large_file = create_large_dataset(n_rows=500000)  # 500K rows for demo
```

#### Strategy 1: Reading in Chunks

```python
def process_large_file_in_chunks(filename, chunk_size=10000):
    """Process large files in manageable chunks"""
    
    print(f"Processing {filename} in chunks of {chunk_size:,} rows...")
    
    # Initialize summary statistics
    total_sales = 0
    row_count = 0
    category_counts = {}
    
    # Process file chunk by chunk
    for chunk_df in pd.read_csv(filename, chunksize=chunk_size):
        # Process each chunk
        total_sales += chunk_df['sales_amount'].sum()
        row_count += len(chunk_df)
        
        # Count categories
        chunk_categories = chunk_df['product_category'].value_counts()
        for category, count in chunk_categories.items():
            category_counts[category] = category_counts.get(category, 0) + count
        
        print(f"Processed {row_count:,} rows so far...")
    
    print(f"\nSummary from {row_count:,} rows:")
    print(f"Total Sales: ${total_sales:,.2f}")
    print(f"Category Distribution: {category_counts}")
    
    return total_sales, category_counts

# Process the large file
total_sales, categories = process_large_file_in_chunks(large_file)
```

#### Strategy 2: Optimizing Data Types

```python
def optimize_datatypes(filename):
    """Optimize data types to reduce memory usage"""
    
    # First, peek at the data to understand structure
    sample_df = pd.read_csv(filename, nrows=1000)
    print("Original data types and memory usage:")
    print(sample_df.dtypes)
    print(f"Memory usage: {sample_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB for 1000 rows")
    
    # Define optimal data types
    dtype_dict = {
        'transaction_id': 'int32',  # Instead of int64
        'customer_id': 'int32',     # Instead of int64  
        'product_category': 'category',  # Instead of object
        'sales_amount': 'float32',  # Instead of float64
        'store_location': 'category',    # Instead of object
        'payment_method': 'category'     # Instead of object
    }
    
    # Load with optimized types
    optimized_df = pd.read_csv(
        filename, 
        dtype=dtype_dict,
        parse_dates=['transaction_date'],  # Parse dates efficiently
        nrows=1000  # Just for comparison
    )
    
    print("\nOptimized data types and memory usage:")
    print(optimized_df.dtypes)
    print(f"Memory usage: {optimized_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB for 1000 rows")
    
    # Calculate memory savings
    original_memory = sample_df.memory_usage(deep=True).sum()
    optimized_memory = optimized_df.memory_usage(deep=True).sum()
    savings = (1 - optimized_memory/original_memory) * 100
    
    print(f"Memory savings: {savings:.1f}%")
    
    return dtype_dict

# Optimize data types
optimal_types = optimize_datatypes(large_file)
```

---

## Section 2: Dealing with Messy Data Types

### Challenge: Real-world data often comes with inconsistent formatting and mixed data types

```python
# Create messy dataset to demonstrate cleaning techniques
def create_messy_dataset():
    """Create a dataset with various data quality issues"""
    
    messy_data = {
        'customer_id': ['CUST001', 'cust-002', 'CUSTOMER_003', '004', 'C_005'],
        'revenue': ['$1,234.56', '2345.67', '$3,456', 'N/A', '4567.89 USD'],
        'signup_date': ['2023-01-15', '15/02/2023', 'March 20, 2023', '2023-04-25', 'invalid_date'],
        'age': ['25', '30.0', 'twenty-five', '35', ''],
        'email': ['john@email.com', 'JANE@EMAIL.COM', 'bob@email', 'alice@email.com', None],
        'phone': ['+1-555-123-4567', '555.234.5678', '(555) 345-6789', '5554567890', 'N/A'],
        'is_premium': ['yes', 'True', '1', 'false', 'No']
    }
    
    return pd.DataFrame(messy_data)

messy_df = create_messy_dataset()
print("Original messy dataset:")
print(messy_df)
print("\nData types:")
print(messy_df.dtypes)
```

#### Cleaning Strategy 1: Standardizing String Data

```python
def clean_string_columns(df):
    """Clean and standardize string columns"""
    
    df_clean = df.copy()
    
    # Clean customer_id: remove non-alphanumeric, standardize format
    df_clean['customer_id_clean'] = (df_clean['customer_id']
                                   .str.replace(r'[^A-Za-z0-9]', '', regex=True)
                                   .str.upper()
                                   .str.pad(7, fillchar='0', side='right'))
    
    # Clean email: lowercase and validate format
    df_clean['email_clean'] = (df_clean['email']
                             .str.lower()
                             .str.strip())
    
    # Validate email format
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    df_clean['email_valid'] = df_clean['email_clean'].str.match(email_pattern, na=False)
    
    # Clean phone numbers: extract digits only
    df_clean['phone_clean'] = (df_clean['phone']
                             .str.replace(r'[^\d]', '', regex=True)
                             .replace('', np.nan))
    
    # Format phone numbers consistently
    def format_phone(phone):
        if pd.isna(phone) or len(str(phone)) != 10:
            return np.nan
        return f"({phone[:3]}) {phone[3:6]}-{phone[6:]}"
    
    df_clean['phone_formatted'] = df_clean['phone_clean'].apply(format_phone)
    
    return df_clean

cleaned_strings = clean_string_columns(messy_df)
print("String columns cleaned:")
print(cleaned_strings[['customer_id', 'customer_id_clean', 'email', 'email_clean', 'phone', 'phone_formatted']])
```

#### Cleaning Strategy 2: Converting Numeric Data

```python
def clean_numeric_columns(df):
    """Clean and convert numeric columns"""
    
    df_clean = df.copy()
    
    # Clean revenue: remove currency symbols, commas, and text
    def clean_revenue(value):
        if pd.isna(value) or value in ['N/A', 'NA', '']:
            return np.nan
        
        # Convert to string and clean
        clean_value = str(value).replace('$', '').replace(',', '').replace(' USD', '')
        
        try:
            return float(clean_value)
        except ValueError:
            return np.nan
    
    df_clean['revenue_clean'] = df_clean['revenue'].apply(clean_revenue)
    
    # Clean age: handle text representations
    def clean_age(value):
        if pd.isna(value) or value == '':
            return np.nan
        
        # Handle text representations
        age_mapping = {
            'twenty-five': 25,
            'thirty': 30,
            'twenty': 20
        }
        
        if str(value).lower() in age_mapping:
            return age_mapping[str(value).lower()]
        
        try:
            return int(float(value))  # Handle "30.0" -> 30
        except ValueError:
            return np.nan
    
    df_clean['age_clean'] = df_clean['age'].apply(clean_age)
    
    return df_clean

cleaned_numeric = clean_numeric_columns(cleaned_strings)
print("Numeric columns cleaned:")
print(cleaned_numeric[['revenue', 'revenue_clean', 'age', 'age_clean']])
```

#### Cleaning Strategy 3: Date and Boolean Conversion

```python
def clean_dates_and_booleans(df):
    """Clean date and boolean columns"""
    
    df_clean = df.copy()
    
    # Clean dates: handle multiple formats
    def parse_flexible_date(date_str):
        if pd.isna(date_str) or date_str == 'invalid_date':
            return pd.NaT
        
        # Try multiple date formats
        formats = ['%Y-%m-%d', '%d/%m/%Y', '%B %d, %Y']
        
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                continue
        
        # If none work, try pandas' flexible parser
        try:
            return pd.to_datetime(date_str)
        except:
            return pd.NaT
    
    df_clean['signup_date_clean'] = df_clean['signup_date'].apply(parse_flexible_date)
    
    # Clean boolean column
    def clean_boolean(value):
        if pd.isna(value):
            return np.nan
        
        value_str = str(value).lower().strip()
        
        if value_str in ['yes', 'true', '1', 'y', 't']:
            return True
        elif value_str in ['no', 'false', '0', 'n', 'f']:
            return False
        else:
            return np.nan
    
    df_clean['is_premium_clean'] = df_clean['is_premium'].apply(clean_boolean)
    
    return df_clean

final_cleaned = clean_dates_and_booleans(cleaned_numeric)
print("Final cleaned dataset:")
print(final_cleaned[['signup_date', 'signup_date_clean', 'is_premium', 'is_premium_clean']])
```

---

## Section 3: Ingesting Complex Data Formats

### Challenge: Working with nested JSON, multiple Excel sheets, and hierarchical data

#### Working with JSON Data

```python
import json

# Create complex JSON data
def create_complex_json():
    """Create nested JSON data structure"""
    
    json_data = {
        "customers": [
            {
                "id": 1,
                "name": "John Doe",
                "contact": {
                    "email": "john@email.com",
                    "phone": "555-1234",
                    "address": {
                        "street": "123 Main St",
                        "city": "New York",
                        "zip": "10001"
                    }
                },
                "orders": [
                    {"order_id": "ORD001", "amount": 150.50, "date": "2023-01-15"},
                    {"order_id": "ORD002", "amount": 75.25, "date": "2023-02-20"}
                ],
                "preferences": ["electronics", "books"]
            },
            {
                "id": 2,
                "name": "Jane Smith",
                "contact": {
                    "email": "jane@email.com", 
                    "phone": "555-5678",
                    "address": {
                        "street": "456 Oak Ave",
                        "city": "Los Angeles", 
                        "zip": "90210"
                    }
                },
                "orders": [
                    {"order_id": "ORD003", "amount": 200.00, "date": "2023-01-10"}
                ],
                "preferences": ["clothing", "home"]
            }
        ]
    }
    
    # Save to file
    with open('complex_data.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    
    return json_data

complex_json = create_complex_json()
print("Created complex JSON structure")
```

```python
def flatten_json_data(filename='complex_data.json'):
    """Flatten nested JSON into pandas DataFrame"""
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Method 1: Use pd.json_normalize for automatic flattening
    customers_flat = pd.json_normalize(data['customers'])
    print("Flattened customer data:")
    print(customers_flat.columns.tolist())
    print(customers_flat.head())
    
    # Method 2: Manual flattening for more control
    flattened_records = []
    
    for customer in data['customers']:
        base_record = {
            'customer_id': customer['id'],
            'customer_name': customer['name'],
            'email': customer['contact']['email'],
            'phone': customer['contact']['phone'],
            'street': customer['contact']['address']['street'],
            'city': customer['contact']['address']['city'],
            'zip': customer['contact']['address']['zip'],
            'preferences': ', '.join(customer['preferences'])
        }
        
        # Flatten orders (one row per order)
        for order in customer['orders']:
            record = base_record.copy()
            record.update({
                'order_id': order['order_id'],
                'order_amount': order['amount'],
                'order_date': order['date']
            })
            flattened_records.append(record)
    
    flattened_df = pd.DataFrame(flattened_records)
    print("\nManually flattened data:")
    print(flattened_df)
    
    return flattened_df

flattened_data = flatten_json_data()
```

#### Working with Excel Files (Multiple Sheets)

```python
def create_multi_sheet_excel():
    """Create Excel file with multiple sheets"""
    
    # Sheet 1: Customer Info
    customers = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
        'segment': ['Premium', 'Standard', 'Premium', 'Standard', 'Premium'],
        'join_date': pd.date_range('2020-01-01', periods=5, freq='30D')
    })
    
    # Sheet 2: Transaction Data
    transactions = pd.DataFrame({
        'transaction_id': range(1, 16),
        'customer_id': [1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
        'amount': np.random.uniform(50, 500, 15).round(2),
        'transaction_date': pd.date_range('2023-01-01', periods=15, freq='7D'),
        'product_category': np.random.choice(['A', 'B', 'C'], 15)
    })
    
    # Sheet 3: Product Catalog
    products = pd.DataFrame({
        'category': ['A', 'B', 'C'],
        'category_name': ['Electronics', 'Clothing', 'Books'],
        'margin_percent': [15.5, 45.2, 30.0]
    })
    
    # Write to Excel
    with pd.ExcelWriter('multi_sheet_data.xlsx', engine='openpyxl') as writer:
        customers.to_excel(writer, sheet_name='Customers', index=False)
        transactions.to_excel(writer, sheet_name='Transactions', index=False)
        products.to_excel(writer, sheet_name='Products', index=False)
    
    print("Created multi-sheet Excel file")
    return customers, transactions, products

customers, transactions, products = create_multi_sheet_excel()
```

```python
def process_multi_sheet_excel(filename='multi_sheet_data.xlsx'):
    """Process Excel file with multiple sheets"""
    
    # Method 1: Read all sheets at once
    all_sheets = pd.read_excel(filename, sheet_name=None)  # None reads all sheets
    
    print("Available sheets:", list(all_sheets.keys()))
    
    # Method 2: Read specific sheets
    customers_df = pd.read_excel(filename, sheet_name='Customers')
    transactions_df = pd.read_excel(filename, sheet_name='Transactions')
    products_df = pd.read_excel(filename, sheet_name='Products')
    
    print("\nCustomers data:")
    print(customers_df.head())
    
    print("\nTransactions data:")
    print(transactions_df.head())
    
    print("\nProducts data:")
    print(products_df.head())
    
    return customers_df, transactions_df, products_df

customers_df, transactions_df, products_df = process_multi_sheet_excel()
```

---

## Section 4: Advanced Data Transformations

### Challenge: Reshaping data and performing complex transformations

#### Pivoting and Melting Data

```python
def demonstrate_pivot_operations():
    """Show pivot, melt, and other reshaping operations"""
    
    # Create sample sales data
    sales_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=12, freq='M'),
        'product_A': np.random.randint(100, 200, 12),
        'product_B': np.random.randint(150, 250, 12),
        'product_C': np.random.randint(80, 180, 12),
        'region': np.random.choice(['North', 'South'], 12)
    })
    
    print("Original wide format data:")
    print(sales_data.head())
    
    # Melt: Wide to Long format
    melted_data = pd.melt(
        sales_data, 
        id_vars=['date', 'region'],
        value_vars=['product_A', 'product_B', 'product_C'],
        var_name='product',
        value_name='sales'
    )
    
    print("\nMelted to long format:")
    print(melted_data.head(10))
    
    # Pivot: Long to Wide format
    pivoted_data = melted_data.pivot_table(
        index=['date'],
        columns=['product', 'region'],
        values='sales',
        aggfunc='mean'
    )
    
    print("\nPivoted back to wide format:")
    print(pivoted_data.head())
    
    # Advanced pivot with multiple aggregations
    advanced_pivot = melted_data.pivot_table(
        index='date',
        columns='product',
        values='sales',
        aggfunc=['sum', 'mean', 'std']
    )
    
    print("\nAdvanced pivot with multiple aggregations:")
    print(advanced_pivot.head())
    
    return melted_data, pivoted_data

melted, pivoted = demonstrate_pivot_operations()
```

#### Custom Transformations and Window Functions

```python
def advanced_transformations():
    """Demonstrate advanced pandas transformations"""
    
    # Create time series data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sales_ts = pd.DataFrame({
        'date': dates,
        'sales': np.random.normal(1000, 200, 100).cumsum(),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'product': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Sort by date for time series operations
    sales_ts = sales_ts.sort_values('date').reset_index(drop=True)
    
    print("Time series data:")
    print(sales_ts.head())
    
    # Rolling calculations
    sales_ts['sales_7day_avg'] = sales_ts['sales'].rolling(window=7).mean()
    sales_ts['sales_7day_std'] = sales_ts['sales'].rolling(window=7).std()
    
    # Expanding calculations (cumulative)
    sales_ts['sales_cumsum'] = sales_ts['sales'].expanding().sum()
    sales_ts['sales_cumavg'] = sales_ts['sales'].expanding().mean()
    
    # Lag and lead features
    sales_ts['sales_lag1'] = sales_ts['sales'].shift(1)
    sales_ts['sales_lead1'] = sales_ts['sales'].shift(-1)
    
    # Percentage change
    sales_ts['sales_pct_change'] = sales_ts['sales'].pct_change()
    
    # Group-wise transformations
    sales_ts['sales_region_rank'] = sales_ts.groupby('region')['sales'].rank(ascending=False)
    sales_ts['sales_region_pct'] = sales_ts.groupby('region')['sales'].transform(
        lambda x: x / x.sum() * 100
    )
    
    print("\nWith advanced transformations:")
    print(sales_ts[['date', 'sales', 'sales_7day_avg', 'sales_lag1', 'sales_region_rank']].head(10))
    
    return sales_ts

transformed_data = advanced_transformations()
```

---

## Section 5: Combining Multiple Datasets

### Challenge: Merging datasets with different structures and handling complex joins

```python
def demonstrate_data_combining():
    """Show various ways to combine datasets"""
    
    # Create related datasets
    customers = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'city': ['NYC', 'LA', 'Chicago', 'Houston', 'Miami'],
        'signup_date': pd.date_range('2020-01-01', periods=5, freq='60D')
    })
    
    orders = pd.DataFrame({
        'order_id': ['O001', 'O002', 'O003', 'O004', 'O005', 'O006'],
        'customer_id': [1, 1, 2, 3, 4, 6],  # Note: customer 6 doesn't exist in customers
        'amount': [100, 150, 200, 75, 300, 125],
        'order_date': pd.date_range('2023-01-01', periods=6, freq='15D')
    })
    
    products = pd.DataFrame({
        'order_id': ['O001', 'O001', 'O002', 'O003', 'O004', 'O005'],
        'product_name': ['Laptop', 'Mouse', 'Phone', 'Tablet', 'Monitor', 'Keyboard'],
        'quantity': [1, 2, 1, 1, 1, 3],
        'unit_price': [80, 10, 200, 75, 300, 15]
    })
    
    print("Customers:")
    print(customers)
    print("\nOrders:")
    print(orders)
    print("\nProducts:")
    print(products)
    
    return customers, orders, products

customers, orders, products = demonstrate_data_combining()
```

```python
def advanced_merging_techniques(customers, orders, products):
    """Demonstrate different types of joins and merging strategies"""
    
    # 1. Inner Join (only matching records)
    inner_join = pd.merge(customers, orders, on='customer_id', how='inner')
    print("Inner Join (customers with orders):")
    print(inner_join)
    
    # 2. Left Join (all customers, matching orders)
    left_join = pd.merge(customers, orders, on='customer_id', how='left')
    print("\nLeft Join (all customers, with/without orders):")
    print(left_join)
    
    # 3. Right Join (all orders, matching customers)
    right_join = pd.merge(customers, orders, on='customer_id', how='right')
    print("\nRight Join (all orders, with/without customer info):")
    print(right_join)
    
    # 4. Outer Join (all records from both)
    outer_join = pd.merge(customers, orders, on='customer_id', how='outer')
    print("\nOuter Join (all customers and orders):")
    print(outer_join)
    
    # 5. Multiple table join
    # First join customers and orders
    customer_orders = pd.merge(customers, orders, on='customer_id', how='left')
    
    # Then join with products
    full_data = pd.merge(customer_orders, products, on='order_id', how='left')
    print("\nFull dataset (customers + orders + products):")
    print(full_data)
    
    # 6. Aggregated join (summarize before joining)
    order_summary = orders.groupby('customer_id').agg({
        'order_id': 'count',
        'amount': ['sum', 'mean', 'max'],
        'order_date': ['min', 'max']
    }).round(2)
    
    # Flatten column names
    order_summary.columns = ['_'.join(col).strip() for col in order_summary.columns]
    order_summary = order_summary.reset_index()
    
    customers_with_summary = pd.merge(customers, order_summary, on='customer_id', how='left')
    print("\nCustomers with order summary:")
    print(customers_with_summary)
    
    return full_data, customers_with_summary

full_dataset, customer_summary = advanced_merging_techniques(customers, orders, products)
```

#### Handling Concatenation and Complex Combinations

```python
def demonstrate_concatenation():
    """Show concatenation and complex combination techniques"""
    
    # Create data from different time periods
    q1_sales = pd.DataFrame({
        'month': ['Jan', 'Feb', 'Mar'],
        'sales': [1000, 1100, 1200],
        'region': 'North'
    })
    
    q2_sales = pd.DataFrame({
        'month': ['Apr', 'May', 'Jun'],
        'sales': [1150, 1250, 1300],
        'region': 'North'
    })
    
    south_sales = pd.DataFrame({
        'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'sales': [800, 850, 900, 950, 1000, 1050],
        'region': 'South'
    })
    
    # Vertical concatenation (stacking rows)
    yearly_north = pd.concat([q1_sales, q2_sales], ignore_index=True)
    print("North region - full year:")
    print(yearly_north)
    
    # Combine all regions
    all_regions = pd.concat([yearly_north, south_sales], ignore_index=True)
    print("\nAll regions combined:")
    print(all_regions)
    
    # Horizontal concatenation (adding columns)
    monthly_targets = pd.DataFrame({
        'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'target': [900, 950, 1000, 1050, 1100, 1150]
    })
    
    # Add targets to north region data
    north_with_targets = pd.merge(yearly_north, monthly_targets, on='month')
    north_with_targets['vs_target'] = north_with_targets['sales'] - north_with_targets['target']
    north_with_targets['target_achievement'] = (north_with_targets['sales'] / north_with_targets['target'] * 100).round(1)
    
    print("\nNorth region with targets:")
    print(north_with_targets)
    
    return all_regions, north_with_targets

all_regions_data, north_targets = demonstrate_concatenation()
```

---

## Section 6: Feature Engineering and Variable Generation

### Challenge: Creating meaningful variables from existing data

```python
def feature_engineering_examples():
    """Demonstrate various feature engineering techniques"""
    
    # Create base dataset
    np.random.seed(42)
    base_data = pd.DataFrame({
        'customer_id': range(1, 1001),
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 15000, 1000).round(0),
        'tenure_months': np.random.randint(1, 60, 1000),
        'purchase_amount': np.random.exponential(100, 1000).round(2),
        'last_purchase_date': pd.date_range('2022-01-01', periods=1000, freq='12H'),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 1000),
        'payment_method': np