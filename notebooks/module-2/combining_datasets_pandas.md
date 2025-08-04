
# Combining Datasets in Pandas

In real-world data analysis, it's common to work with data spread across multiple tables or files. Pandas provides powerful tools to combine these datasets efficiently:

## 🔹 Concatenation (`pd.concat`)
- **What it does**: Stacks DataFrames either vertically (row-wise) or horizontally (column-wise).
- **When to use**: When the datasets have the same structure (e.g., same columns for vertical stacking).
- **Example**: Combining monthly sales reports stored as separate DataFrames into one long DataFrame.

```python
combined_df = pd.concat([df_jan, df_feb, df_mar], axis=0)
```

## 🔹 Merge (`pd.merge`)
- **What it does**: Combines DataFrames based on one or more common columns (similar to SQL joins).
- **When to use**: When you need to enrich one dataset with columns from another based on shared keys (e.g., customer ID, district).
- **Types of joins**:
  - `inner` – Only rows with matching keys in both DataFrames
  - `left` – All rows from the left DataFrame, with matches from the right
  - `right` – All rows from the right DataFrame, with matches from the left
  - `outer` – All rows from both, matching where possible

```python
merged_df = pd.merge(df_customers, df_orders, on='customer_id', how='left')
```

## ✅ Summary
- Use **`concat()`** to stack data together.
- Use **`merge()`** to combine related data based on key columns.

Understanding when and how to use these functions is key to building clean and meaningful datasets for analysis.
