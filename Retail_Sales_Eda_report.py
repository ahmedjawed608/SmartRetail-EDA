import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('eda_plots', exist_ok=True)

# Step 1: Load data
try:
    df = pd.read_csv('retail_sales.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'retail_sales.csv' not found.")
    exit()

print("\n--- Initial Inspection ---")
print(df.head())
df.info()
print(df.describe())
print("Category unique values:", df['Category'].unique())
print("Region unique values:", df['Region'].unique())

# Step 2: Type conversion — do this before anything else
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce')

# Step 3: Clean invalid string nulls, then drop only where Category is missing
invalid_strings = ["nan?", "nan", "null", "null?", ""]
for col in ['Category', 'Region']:
    df[col] = df[col].apply(
        lambda x: np.nan if isinstance(x, str) and x.strip().lower() in invalid_strings else x
    )
df = df.dropna(subset=['Category'])
df.dropna(subset=['Date'], inplace=True)

# Step 4: Feature engineering from Date
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek

print("\n--- Missing Values After Cleaning ---")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values.")

if missing_values.sum() > 0:
    plt.figure(figsize=(12, 7))
    sns.heatmap(df.isnull(), cbar=True, cmap='viridis')
    plt.title('Heatmap of Missing Values')
    plt.savefig('eda_plots/missing_values_heatmap.png')
    plt.close()

print("\n--- Descriptive Statistics ---")
print(df.describe(include=np.number))

# Step 5: Numerical distributions (before outlier treatment)
print("\n--- Distribution of Numerical Features ---")
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
# Exclude engineered time columns from distribution plots
plot_num_cols = [c for c in numerical_cols if c not in ['Year', 'Month', 'DayOfWeek']]

for col in plot_num_cols:
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col)
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[col].dropna())
    plt.title(f'Box Plot of {col}', fontsize=14)
    plt.ylabel(col)

    plt.tight_layout()
    plt.savefig(f'eda_plots/{col}_distribution.png')
    plt.close()
    print(f"Distribution plot saved: '{col}'")

# Step 6: Outlier treatment using IQR clipping
print("\n--- Outlier Detection and Treatment ---")
for col in plot_num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[f'{col}_original'] = df[col]
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    print(f"  {col}: original range [{df[f'{col}_original'].min():.2f}, {df[f'{col}_original'].max():.2f}]"
          f" → clipped [{df[col].min():.2f}, {df[col].max():.2f}]")

    plt.figure(figsize=(10, 6))
    sns.histplot(df[col].dropna(), kde=True, bins=30)
    plt.title(f'Distribution of {col} (Outliers Clipped)', fontsize=14)
    plt.xlabel(f'{col} (Clipped)')
    plt.ylabel('Frequency')
    plt.savefig(f'eda_plots/{col}_distribution_clipped.png')
    plt.close()

# Step 7: Categorical distributions — Date is datetime now so won't appear here
print("\n--- Distribution of Categorical Features ---")
categorical_cols = df.select_dtypes(include='str').columns.tolist()

for col in categorical_cols:
    plt.figure(figsize=(12, 7))
    sns.countplot(y=df[col].dropna(), order=df[col].value_counts().index,
                  hue=df[col].dropna(), palette='viridis', legend=False)
    plt.title(f'Count Plot of {col}', fontsize=14)
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f'eda_plots/{col}_countplot.png')
    plt.close()
    print(f"Value counts for '{col}':\n{df[col].value_counts()}\n")

# Step 8: Time series
print("\n--- Time Series Analysis ---")
if 'Sales' in df.columns:
    daily_sales = df.groupby('Date')['Sales'].sum().reset_index()
    plt.figure(figsize=(16, 8))
    plt.plot(daily_sales['Date'], daily_sales['Sales'], marker='o', linestyle='-')
    plt.title('Total Sales Over Time', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.grid(True)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('eda_plots/sales_over_time.png')
    plt.close()
    print("Time series plot saved.")

# Step 9: Correlation matrix
print("\n--- Correlation Analysis ---")
df_numeric = df.select_dtypes(include=np.number)
correlation_matrix = df_numeric.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig('eda_plots/correlation_matrix.png')
plt.close()
print("Correlation matrix saved.")

# Step 10: Bivariate analysis
print("\n--- Bivariate Analysis ---")
if 'Category' in df.columns and 'Sales' in df.columns:
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Category', y='Sales', data=df,
                hue='Category', palette='pastel', estimator=np.mean, legend=False)
    plt.title('Average Sales by Category', fontsize=16)
    plt.xlabel('Product Category')
    plt.ylabel('Average Sales')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('eda_plots/sales_by_category.png')
    plt.close()
    print("Average sales by category plot saved.")

if 'Sales' in df.columns and 'Profit' in df.columns:
    plt.figure(figsize=(12, 8))
    hue_col = 'Category' if 'Category' in df.columns else None
    sns.scatterplot(x='Sales', y='Profit', data=df, hue=hue_col, palette='deep')
    plt.title('Sales vs. Profit', fontsize=16)
    plt.xlabel('Sales')
    plt.ylabel('Profit')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('eda_plots/sales_vs_profit_scatterplot.png')
    plt.close()
    print("Sales vs Profit scatter plot saved.")

# Step 11: Key Insights and Business Recommendations
df['MonthName'] = df['Date'].dt.strftime('%B')
df['DayName'] = df['Date'].dt.day_name()

category_performance = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
cat_profit_performance = df.groupby('Category')['Profit'].sum().sort_values(ascending=False)
regional_data = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
monthly_sales = df.groupby('MonthName')['Sales'].sum().reset_index()
daily_performance = df.groupby('DayName')['Sales'].sum().sort_values(ascending=False)
corr_sales_profit = df[['Sales', 'Profit']].corr().loc['Sales', 'Profit']

print("\n=== KEY INSIGHTS AND RECOMMENDATIONS ===")

print("\n KEY FINDINGS:")
print("1. Top Performing Category (Sales):", category_performance.index[0],
      f"— ${category_performance.iloc[0]:,.2f} total sales")
print("2. Top Performing Category (Profit):", cat_profit_performance.index[0],
      f"— ${cat_profit_performance.iloc[0]:,.2f} total profit")
print("3. Best Performing Region:", regional_data.index[0],
      f"— ${regional_data.iloc[0]:,.2f} total sales")
print("4. Weakest Region:", regional_data.index[-1],
      f"— ${regional_data.iloc[-1]:,.2f} total sales")
print("5. Highest Sales Month:", monthly_sales.loc[monthly_sales['Sales'].idxmax(), 'MonthName'],
      f"— ${monthly_sales['Sales'].max():,.2f}")
print("6. Best Day of Week:", daily_performance.index[0],
      f"— ${daily_performance.iloc[0]:,.2f} total sales")
print("7. Total Annual Sales:  ${:,.2f}".format(df['Sales'].sum()))
print("8. Total Annual Profit: ${:,.2f}".format(df['Profit'].sum()))

print("\n BUSINESS RECOMMENDATIONS:")
print(f"1. Focus marketing budget on '{category_performance.index[0]}' — it drives the highest revenue across all categories.")
print(f"2. '{regional_data.index[-1]}' region is underperforming by ${regional_data.iloc[0] - regional_data.iloc[-1]:,.2f} vs top region — investigate pricing or distribution issues there.")
print(f"3. Run promotions and stock up inventory before {monthly_sales.loc[monthly_sales['Sales'].idxmax(), 'MonthName']} — it is the peak sales month.")
print(f"4. Schedule flash sales or discounts on {daily_performance.index[0]}s — highest sales day of the week.")
print(f"5. Sales and Profit have a {corr_sales_profit:.2f} correlation — volume-driving strategies are effective since higher sales lead to higher profit.")
print(f"6. '{cat_profit_performance.index[-1]}' has the lowest profit — review its pricing margin or cost structure.")

print("\n PROFITABILITY ANALYSIS:")
profit_margin = (df['Profit'].sum() / df['Sales'].sum()) * 100
print(f"Overall Profit Margin: {profit_margin:.2f}%")
print(f"Best profit category:  '{cat_profit_performance.index[0]}' — ${cat_profit_performance.iloc[0]:,.2f}")
print(f"Worst profit category: '{cat_profit_performance.index[-1]}' — ${cat_profit_performance.iloc[-1]:,.2f}")

print("\nEDA complete. All plots saved in 'eda_plots/'.")