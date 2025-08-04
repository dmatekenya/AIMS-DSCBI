# 🐼 Pandas Tutorial: 2-Hour Crash Course

Welcome to this hands-on tutorial on **pandas** for data analysis! In this notebook, we will explore the core functionality of pandas using examples and exercises.

---

## 🕒 0:10 – Loading & Exploring Data

```python
import pandas as pd
import numpy as np

# Simulated dataset
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'age': [25, 30, 35, 40, np.nan],
    'income': [50000, 60000, 55000, 70000, 65000],
    'gender': ['F', 'M', 'M', 'M', 'F']
}
df = pd.DataFrame(data)

# View first few rows
df.head()
```

---

## 🕒 0:30 – Subsetting & Filtering

```python
# Select a single column
df['age']

# Filter rows
df[df['age'] > 30]

# Use query
df.query("income > 55000 and gender == 'M'")
```

---

## 🕒 0:55 – Cleaning & Transforming Data

```python
# Check for missing values
df.isnull().sum()

# Fill missing age
df['age'] = df['age'].fillna(df['age'].mean())

# Create a new column
df['income_k'] = df['income'] / 1000

# Rename columns
df.rename(columns={'income': 'annual_income'}, inplace=True)

df
```

---

## 🕒 1:15 – Combining & Reshaping Data

```python
# Another DataFrame to merge
df_region = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'region': ['North', 'South', 'East', 'West', 'Central']
})

# Merge
df_merged = pd.merge(df, df_region, on='name')
df_merged
```

---

## 🕒 1:35 – Working with Dates & Time

```python
# Add simulated date
df_merged['join_date'] = pd.date_range(start='2023-01-01', periods=5, freq='M')

# Extract year and month
df_merged['year'] = df_merged['join_date'].dt.year
df_merged['month'] = df_merged['join_date'].dt.month

df_merged
```

---

## 🕒 1:50 – Wrap-Up and Resources

- **pandas Documentation**: https://pandas.pydata.org/docs/
- **Cheat Sheet**: https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
- **Practice**: https://www.kaggle.com/learn/pandas

Thanks for participating!
