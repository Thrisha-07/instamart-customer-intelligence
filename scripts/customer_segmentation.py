
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

customers = pd.read_csv(DATA_DIR / "clean" / "customers.csv")
orders = pd.read_csv(DATA_DIR / "clean" / "orders.csv")

# Use completed orders to calculate customer spending and order frequency.
completed = orders[orders["status"].astype(str).str.lower() == "completed"].copy()

completed["order_amount"] = pd.to_numeric(
    completed["order_amount"], errors="coerce"
)

summary = completed.groupby("customer_id").agg(
    total_spend=("order_amount", "sum"),
    order_count=("order_id", "nunique"),
).reset_index()

segments = customers[["customer_id"]].merge(
    summary, on="customer_id", how="left"
)

segments["total_spend"] = segments["total_spend"].fillna(0)
segments["order_count"] = segments["order_count"].fillna(0).astype(int)

# Mark the top 20% of customers by spending as premium.
threshold = segments["total_spend"].quantile(0.80)
segments["is_premium"] = segments["total_spend"] >= threshold

segments["segment_name"] = "regular"
segments.loc[
    (segments["order_count"] >= 5) & segments["is_premium"],
    "segment_name"
] = "loyal_premium"
segments.loc[
    (segments["order_count"] < 5) & segments["is_premium"],
    "segment_name"
] = "high_value"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

segments.to_csv(OUTPUT_DIR / "customer_segments.csv", index=False)

print("Customer segmentation complete.")
print(f"Customers segmented: {len(segments)}")
print(f"Premium customers: {segments['is_premium'].sum()}")
print(segments["segment_name"].value_counts())
print(f"Saved to: {OUTPUT_DIR / 'customer_segments.csv'}")
