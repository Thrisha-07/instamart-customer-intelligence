
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ORDERS_PATH = ROOT / "data" / "clean" / "orders.csv"
CUSTOMERS_PATH = ROOT / "data" / "clean" / "customers.csv"
PRODUCTS_PATH = ROOT / "data" / "clean" / "products.csv"

# Load cleaned datasets
orders = pd.read_csv(ORDERS_PATH)
customers = pd.read_csv(CUSTOMERS_PATH)
products = pd.read_csv(PRODUCTS_PATH)

# Keep completed orders only
completed = orders[orders["status"] == "completed"].copy()

# Ensure numeric columns
numeric_columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes"
]

for column in numeric_columns:
    completed[column] = pd.to_numeric(
        completed[column],
        errors="coerce"
    )

print("=" * 60)
print("PART A - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 1. Order amount descriptive statistics
amount = completed["order_amount"]

print("\n1. ORDER AMOUNT DISTRIBUTION")
print("Completed orders:", len(completed))
print(f"Mean: INR {amount.mean():,.2f}")
print(f"Median: INR {amount.median():,.2f}")
print(f"Skewness: {amount.skew():.2f}")

# 2. Quartiles and IQR outlier detection
q1 = amount.quantile(0.25)
q2 = amount.quantile(0.50)
q3 = amount.quantile(0.75)

iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

lower_outliers = (amount < lower_fence).sum()
upper_outliers = (amount > upper_fence).sum()

print("\n2. QUARTILES AND OUTLIERS")
print(f"Q1: INR {q1:,.2f}")
print(f"Median (Q2): INR {q2:,.2f}")
print(f"Q3: INR {q3:,.2f}")
print(f"IQR: INR {iqr:,.2f}")
print(f"Lower fence: INR {lower_fence:,.2f}")
print(f"Upper fence: INR {upper_fence:,.2f}")
print("Outliers below lower fence:", lower_outliers)
print("Outliers above upper fence:", upper_outliers)
print("Total outliers:", lower_outliers + upper_outliers)

# 3. SQL-style city and category revenue analysis using Pandas
analysis = (
    completed
    .merge(
        customers[["customer_id", "city"]],
        on="customer_id",
        how="left"
    )
    .merge(
        products[["product_id", "category"]],
        on="product_id",
        how="left"
    )
)

pivot = pd.pivot_table(
    analysis,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    fill_value=0,
    margins=True
).round(2)

print("\n3. CITY AND CATEGORY REVENUE PIVOT")
print(pivot)
print(f"\nPivot grand total: INR {pivot.loc['All', 'All']:,.2f}")

# 4. City-level sum, mean and count using named aggregation
city_stats = (
    analysis.groupby("city")
    .agg(
        total_revenue_inr=("order_amount", "sum"),
        average_order_value_inr=("order_amount", "mean"),
        completed_order_count=("order_amount", "count")
    )
    .round(2)
)

print("\n4. CITY-LEVEL REVENUE STATISTICS")
print(city_stats)

# 5. Correlation matrix
correlation = completed[numeric_columns].corr().round(3)

print("\n5. CORRELATION MATRIX")
print(correlation)

# Find the strongest absolute correlation between different variables
absolute_correlation = correlation.abs()
for i in range(len(numeric_columns)):
    absolute_correlation.iloc[i, i] = 0

strongest_pair = absolute_correlation.stack().idxmax()
strongest_value = correlation.loc[
    strongest_pair[0],
    strongest_pair[1]
]

print("\nStrongest absolute correlation:")
print(f"{strongest_pair[0]} and {strongest_pair[1]}: {strongest_value:.3f}")

print("\nExploratory analysis completed.")
