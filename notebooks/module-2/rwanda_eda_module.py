# Module 1: Exploratory Data Analysis (EDA) Fundamentals
# Dataset: Rwanda Cell Population Data
# Focus: Population demographics across administrative divisions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

print("=== MODULE 1: EXPLORATORY DATA ANALYSIS ===")
print("Dataset: Rwanda Cell Population Data\n")

# =====================================
# 1. DATA LOADING AND INITIAL INSPECTION
# =====================================

print("1. DATA LOADING AND INITIAL INSPECTION")
print("-" * 50)

# Load the dataset
df = pd.read_csv('rwacellpop.csv')

print(f"Dataset shape: {df.shape}")
print(f"Total cells analyzed: {df.shape[0]:,}")
print(f"Variables available: {df.shape[1]}")

print("\nColumn names and types:")
print(df.dtypes)

print("\nFirst few rows:")
print(df.head())

print("\nDataset info:")
df.info()

# =====================================
# 2. DATA CLEANING AND PREPROCESSING
# =====================================

print("\n\n2. DATA CLEANING AND PREPROCESSING")
print("-" * 50)

# Check for missing values
print("Missing values per column:")
missing_vals = df.isnull().sum()
missing_pct = (missing_vals / len(df)) * 100
missing_summary = pd.DataFrame({
    'Missing_Count': missing_vals,
    'Missing_Percentage': missing_pct
})
print(missing_summary[missing_summary['Missing_Count'] > 0])

# Check for duplicates
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Examine unique values in categorical columns
print("\nUnique values in administrative levels:")
print(f"Provinces: {df['province_name'].nunique()}")
print(f"Districts: {df['district_name'].nunique()}")
print(f"Sectors: {df['sector_name'].nunique()}")
print(f"Cells: {df['cell_name'].nunique()}")

# Create derived variables for analysis
print("\nCreating derived variables...")

# Calculate total population per cell
df['total_population'] = df['general_2020']

# Create age group proportions
df['elderly_proportion'] = df['elderly_60_plus_2020'] / df['total_population']
df['children_proportion'] = df['children_under_five_2020'] / df['total_population']
df['youth_proportion'] = df['youth_15_24_2020'] / df['total_population']

# Gender ratio (men per 100 women)
df['gender_ratio'] = (df['men_2020'] / df['women_2020']) * 100

# Population density proxy (people per building)
df['people_per_building'] = df['total_population'] / df['building_count']

# Age dependency ratio
df['dependency_ratio'] = (df['children_under_five_2020'] + df['elderly_60_plus_2020']) / (df['total_population'] - df['children_under_five_2020'] - df['elderly_60_plus_2020'])

print("New variables created successfully!")

# =====================================
# 3. UNIVARIATE ANALYSIS
# =====================================

print("\n\n3. UNIVARIATE ANALYSIS")
print("-" * 50)

# Descriptive statistics for population variables
pop_columns = ['total_population', 'elderly_60_plus_2020', 'children_under_five_2020', 
               'youth_15_24_2020', 'men_2020', 'women_2020', 'building_count']

print("Descriptive statistics for population variables:")
desc_stats = df[pop_columns].describe()
print(desc_stats.round(2))

# Distribution analysis
print("\nDistribution characteristics:")
for col in ['total_population', 'gender_ratio', 'people_per_building']:
    if col in df.columns:
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")
        print(f"  Std Dev: {df[col].std():.2f}")
        print(f"  Skewness: {df[col].skew():.2f}")
        print(f"  Kurtosis: {df[col].kurtosis():.2f}")

# Visualization of key distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution of Key Population Variables', fontsize=16)

# Total population distribution
axes[0,0].hist(df['total_population'], bins=50, alpha=0.7, color='skyblue', edgecolors='black')
axes[0,0].set_title('Total Population Distribution')
axes[0,0].set_xlabel('Population')
axes[0,0].set_ylabel('Frequency')

# Log-transformed population (often more normal)
axes[0,1].hist(np.log(df['total_population']), bins=50, alpha=0.7, color='lightgreen', edgecolors='black')
axes[0,1].set_title('Log(Total Population)')
axes[0,1].set_xlabel('Log(Population)')
axes[0,1].set_ylabel('Frequency')

# Gender ratio
axes[0,2].hist(df['gender_ratio'], bins=50, alpha=0.7, color='salmon', edgecolors='black')
axes[0,2].set_title('Gender Ratio (Men per 100 Women)')
axes[0,2].set_xlabel('Ratio')
axes[0,2].set_ylabel('Frequency')

# Age proportions
axes[1,0].hist(df['elderly_proportion'], bins=50, alpha=0.7, color='gold', edgecolors='black')
axes[1,0].set_title('Proportion of Elderly (60+)')
axes[1,0].set_xlabel('Proportion')
axes[1,0].set_ylabel('Frequency')

# Children proportion
axes[1,1].hist(df['children_proportion'], bins=50, alpha=0.7, color='lightcoral', edgecolors='black')
axes[1,1].set_title('Proportion of Children (<5)')
axes[1,1].set_xlabel('Proportion')
axes[1,1].set_ylabel('Frequency')

# People per building
axes[1,2].hist(df['people_per_building'], bins=50, alpha=0.7, color='mediumpurple', edgecolors='black')
axes[1,2].set_title('People per Building')
axes[1,2].set_xlabel('People/Building')
axes[1,2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# =====================================
# 4. BIVARIATE ANALYSIS
# =====================================

print("\n\n4. BIVARIATE ANALYSIS")
print("-" * 50)

# Correlation analysis
print("Correlation matrix for key variables:")
corr_vars = ['total_population', 'elderly_proportion', 'children_proportion', 
             'youth_proportion', 'gender_ratio', 'people_per_building', 'dependency_ratio']
correlation_matrix = df[corr_vars].corr()
print(correlation_matrix.round(3))

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
plt.title('Correlation Matrix: Population Demographics')
plt.tight_layout()
plt.show()

# Population by administrative level
print("\nPopulation distribution by administrative levels:")

# By Province
province_pop = df.groupby('province_name')['total_population'].agg(['sum', 'mean', 'count']).round(2)
province_pop.columns = ['Total_Pop', 'Avg_Cell_Pop', 'Cell_Count']
print("By Province:")
print(province_pop.sort_values('Total_Pop', ascending=False))

# Scatter plots for key relationships
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Key Bivariate Relationships', fontsize=16)

# Population vs Building Count
axes[0,0].scatter(df['building_count'], df['total_population'], alpha=0.6, color='blue')
axes[0,0].set_xlabel('Building Count')
axes[0,0].set_ylabel('Total Population')
axes[0,0].set_title('Population vs Building Count')

# Gender ratio vs Total population
axes[0,1].scatter(df['total_population'], df['gender_ratio'], alpha=0.6, color='red')
axes[0,1].set_xlabel('Total Population')
axes[0,1].set_ylabel('Gender Ratio')
axes[0,1].set_title('Gender Ratio vs Population Size')

# Children vs Elderly proportions
axes[1,0].scatter(df['children_proportion'], df['elderly_proportion'], alpha=0.6, color='green')
axes[1,0].set_xlabel('Children Proportion')
axes[1,0].set_ylabel('Elderly Proportion')
axes[1,0].set_title('Age Structure Relationship')

# Dependency ratio vs Population
axes[1,1].scatter(df['total_population'], df['dependency_ratio'], alpha=0.6, color='purple')
axes[1,1].set_xlabel('Total Population')
axes[1,1].set_ylabel('Dependency Ratio')
axes[1,1].set_title('Dependency Ratio vs Population')

plt.tight_layout()
plt.show()

# =====================================
# 5. MULTIVARIATE EXPLORATION
# =====================================

print("\n\n5. MULTIVARIATE EXPLORATION")
print("-" * 50)

# Box plots by province
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Population Characteristics by Province', fontsize=16)

# Total population by province
df.boxplot(column='total_population', by='province_name', ax=axes[0,0])
axes[0,0].set_title('Total Population by Province')
axes[0,0].set_xlabel('Province')
axes[0,0].tick_params(axis='x', rotation=45)

# Gender ratio by province
df.boxplot(column='gender_ratio', by='province_name', ax=axes[0,1])
axes[0,1].set_title('Gender Ratio by Province')
axes[0,1].set_xlabel('Province')
axes[0,1].tick_params(axis='x', rotation=45)

# Children proportion by province
df.boxplot(column='children_proportion', by='province_name', ax=axes[1,0])
axes[1,0].set_title('Children Proportion by Province')
axes[1,0].set_xlabel('Province')
axes[1,0].tick_params(axis='x', rotation=45)

# Elderly proportion by province
df.boxplot(column='elderly_proportion', by='province_name', ax=axes[1,1])
axes[1,1].set_title('Elderly Proportion by Province')
axes[1,1].set_xlabel('Province')
axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Summary statistics by province
print("Detailed statistics by province:")
province_summary = df.groupby('province_name').agg({
    'total_population': ['mean', 'median', 'std'],
    'gender_ratio': ['mean', 'std'],
    'children_proportion': ['mean', 'std'],
    'elderly_proportion': ['mean', 'std'],
    'people_per_building': ['mean', 'std']
}).round(3)

print(province_summary)

# =====================================
# 6. OUTLIER DETECTION AND TREATMENT
# =====================================

print("\n\n6. OUTLIER DETECTION AND TREATMENT")
print("-" * 50)

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(data, column, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(data[column]))
    outliers = data[z_scores > threshold]
    return outliers

# Analyze outliers for key variables
outlier_vars = ['total_population', 'gender_ratio', 'people_per_building']

for var in outlier_vars:
    print(f"\nOutlier analysis for {var}:")
    
    # IQR method
    iqr_outliers, lower_bound, upper_bound = detect_outliers_iqr(df, var)
    print(f"  IQR method: {len(iqr_outliers)} outliers ({len(iqr_outliers)/len(df)*100:.1f}%)")
    print(f"  Normal range: {lower_bound:.2f} to {upper_bound:.2f}")
    
    # Z-score method
    zscore_outliers = detect_outliers_zscore(df, var)
    print(f"  Z-score method: {len(zscore_outliers)} outliers ({len(zscore_outliers)/len(df)*100:.1f}%)")
    
    if len(iqr_outliers) > 0:
        print(f"  Extreme values: {df[var].nsmallest(3).values} ... {df[var].nlargest(3).values}")

# Visualize outliers
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Outlier Detection Visualization', fontsize=16)

for i, var in enumerate(outlier_vars):
    # Box plot
    axes[i].boxplot(df[var])
    axes[i].set_title(f'{var}')
    axes[i].set_ylabel('Value')

plt.tight_layout()
plt.show()

# =====================================
# 7. KEY INSIGHTS SUMMARY
# =====================================

print("\n\n7. KEY INSIGHTS FROM EDA")
print("-" * 50)

print("Dataset Overview:")
print(f"• {len(df):,} administrative cells across {df['province_name'].nunique()} provinces")
print(f"• Total population covered: {df['total_population'].sum():,.0f}")
print(f"• Average cell population: {df['total_population'].mean():.0f}")

print(f"\nDemographic Patterns:")
print(f"• Gender ratio: {df['gender_ratio'].mean():.1f} men per 100 women")
print(f"• Children (<5): {df['children_proportion'].mean()*100:.1f}% of population")
print(f"• Elderly (60+): {df['elderly_proportion'].mean()*100:.1f}% of population")
print(f"• Youth (15-24): {df['youth_proportion'].mean()*100:.1f}% of population")

print(f"\nInfrastructure:")
print(f"• Average people per building: {df['people_per_building'].mean():.1f}")
print(f"• Dependency ratio: {df['dependency_ratio'].mean():.2f}")

print(f"\nVariability:")
print(f"• Population ranges from {df['total_population'].min():.0f} to {df['total_population'].max():,.0f}")
print(f"• High variation in population density and age structure across cells")

print("\n" + "="*60)
print("EDA COMPLETE - Ready for visualization and inference modules!")
print("="*60)