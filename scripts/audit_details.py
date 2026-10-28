
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

orders = pd.read_csv(RAW / "orders.csv")
customers = pd.read_csv(RAW / "customers.csv")

amounts = pd.to_numeric(orders["order_amount"], errors="coerce")
delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
age = pd.to_numeric(customers["age"], errors="coerce")

print("=== ORDER AUDIT ===")
print("Raw order rows:", len(orders))
print("Exact duplicate rows:", int(orders.duplicated().sum()))
print("Pending rows:", int(orders["status"].eq("pending").sum()))
print("Distinct order IDs:", orders["order_id"].nunique())
print("Non-numeric order amounts:", int(amounts.isna().sum()))
print("Negative delivery times:", int(delivery.lt(0).sum()))
print("Blank delivery times:", int(orders["delivery_minutes"].isna().sum()))

print("\n=== CUSTOMER AUDIT ===")
print("Blank ages:", int(customers["age"].isna().sum()))
print("Ages outside 10–100:", int((age.lt(10) | age.gt(100)).sum()))
print("Distinct raw city spellings:", customers["city"].nunique())
print(
    "Distinct non-blank gender spellings:",
    customers.loc[
        customers["gender"].fillna("").str.strip().ne(""), "gender"
    ].nunique()
)
print(
    "Blank genders:",
    int(customers["gender"].fillna("").str.strip().eq("").sum())
)
