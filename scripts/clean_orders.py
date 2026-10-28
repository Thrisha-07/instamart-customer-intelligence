
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

# Load raw files
orders = pd.read_csv(RAW / "orders.csv")
products = pd.read_csv(RAW / "products.csv")

# 1. Convert timestamps to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")

# 2. Remove exact duplicate rows
orders = orders.drop_duplicates().copy()

# 3. Keep the latest record for each order ID
orders = (
    orders.sort_values("order_ts")
    .drop_duplicates(subset="order_id", keep="last")
    .copy()
)

# 4. Convert amounts to numeric
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

# Attach product prices and categories
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one",
)

# Fill invalid amounts using the specified formula
missing_amount = orders["order_amount"].isna()
orders.loc[missing_amount, "order_amount"] = (
    orders.loc[missing_amount, "price_inr"]
    * orders.loc[missing_amount, "quantity"]
    * (1 - orders.loc[missing_amount, "discount_pct"] / 100)
).round(2)

# 5. Convert delivery time and invalidate negative values
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = float("nan")

# 6. Impute missing delivery times for completed orders only
completed = orders["status"].eq("completed")
orders.loc[completed, "delivery_minutes"] = (
    orders.loc[completed, "delivery_minutes"]
    .fillna(
        orders.loc[completed]
        .groupby("category")["delivery_minutes"]
        .transform("median")
    )
)

# Convert timestamp back to the required SQLite-friendly text format
orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

# Remove helper columns before saving
orders = orders.drop(columns=["price_inr", "category"])

# Save the clean orders dataset
orders.to_csv(CLEAN / "orders.csv", index=False)

# Validation report
print("Clean orders shape:", orders.shape)
print("\nStatus counts:")
print(orders["status"].value_counts().to_string())
print("\nDistinct order IDs:", orders["order_id"].nunique())
print("Duplicate order IDs:", orders["order_id"].duplicated().sum())
print("Missing order amounts:", orders["order_amount"].isna().sum())
print(
    "Missing delivery times for completed orders:",
    orders.loc[orders["status"].eq("completed"), "delivery_minutes"].isna().sum(),
)
print(
    "Missing delivery times for cancelled orders:",
    orders.loc[orders["status"].eq("cancelled"), "delivery_minutes"].isna().sum(),
)
print(
    "Completed order amount total:",
    f"INR {orders.loc[orders['status'].eq('completed'), 'order_amount'].sum():,.2f}",
)
print("\nSaved to:", CLEAN / "orders.csv")
