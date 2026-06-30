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
#Step 9: Regional Profit

if 'Region' in df.columns and 'Profit' in df.columns:
    region_profit = df.groupby('Region')['Profit'].sum()
    region_profit = region_profit.clip(lower=0)
    
    plt.figure(figsize=(10, 8))
    plt.pie(region_profit, labels=region_profit.index, autopct='%1.1f%%', 
            startangle=90, colors=sns.color_palette('pastel'))
    plt.title('Profit Share by Region', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('eda_plots/profit_share_by_region_pie.png')
    plt.close()
    print("Pie chart for 'Profit Share by Region' saved to 'eda_plots/profit_share_by_region_pie.png'")

# Step 10: Correlation matrix
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

# Step 11: Summary report with real computed values
print("\n--- EDA Summary ---")
category_sales = df.groupby('Category')['Sales'].sum()
top_category =category_sales.idxmax()
top_category_value = category_sales.sum().max()
sales_skew = df['Sales_original'].skew() if 'Sales_original' in df.columns else df['Sales'].skew()
most_common_category = df['Category'].value_counts().idxmax()
most_common_region = df['Region'].value_counts().idxmax() if df['Region'].notna().any() else 'N/A'
corr_sales_profit = df_numeric.corr().loc['Sales', 'Profit'] if 'Sales' in df_numeric and 'Profit' in df_numeric else None

print("\n### Key Observations:")
print(f"- Cleaned invalid null strings ('NaN?', 'Null', 'Nan') from Category and Region.")
print(f"- Sales skewness: {sales_skew:.2f} ({'right-skewed' if sales_skew > 0.5 else 'left-skewed' if sales_skew < -0.5 else 'roughly symmetric'}).")
print(f"- Outliers clipped using IQR method on Sales, Profit, and Quantity.")
print(f"- Most frequent category: '{most_common_category}'. Most frequent region: '{most_common_region}'.")
print(f"- Highest avg sales category: '{top_category}' ({top_category_value:.2f}).")
if corr_sales_profit is not None:
    print(f"- Sales vs Profit correlation: {corr_sales_profit:.2f}.")
print(f"- Extracted Year, Month, DayOfWeek from Date for time-based analysis.")

print("\n### Recommendations:")
print(f"- Fix null markers at data source level to avoid manual cleaning.")
print(f"- Prioritize '{top_category}' in marketing — highest average sales.")
print(f"- Use Month and DayOfWeek features to analyze seasonal and weekly sales patterns.")
print(f"- Review pre-clip outlier values — some may be genuine high-value transactions.")
print(f"- Next step: build a sales forecasting model using time-based features.")

print("\nEDA complete. All plots saved in 'eda_plots/'.")