from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

print("=" * 60)
print("CUSTOMER CLEANING COUNTS")
print("=" * 60)

customers = pd.read_csv(RAW / "customers.csv")

city_clean = customers["city"].str.strip().str.title()
print("City values changed:", (customers["city"] != city_clean).sum())

def normalize_gender(value):
    if pd.isna(value) or not str(value).strip():
        return "unknown"
    value = str(value).strip().lower()
    if value in {"m", "male", "man"}:
        return "male"
    if value in {"f", "female", "woman"}:
        return "female"
    return "unknown"

gender_clean = customers["gender"].apply(normalize_gender)
print("Gender values changed:", (customers["gender"] != gender_clean).sum())
print("Originally missing genders:", customers["gender"].isna().sum())

age_numeric = pd.to_numeric(customers["age"], errors="coerce")
invalid_age = age_numeric.notna() & ~age_numeric.between(10, 100)
age_needing_imputation = age_numeric.isna() | invalid_age
missing_age = age_numeric.isna()

print("Originally missing ages:", missing_age.sum())
print("Ages outside 10-100:", invalid_age.sum())
print("Age values requiring imputation:", age_needing_imputation.sum())

age_valid = age_numeric.copy()
age_valid[invalid_age] = float("nan")
age_filled = age_valid.fillna(
    customers.assign(city=city_clean)
    .groupby("city")["age"]
    .transform(lambda x: pd.to_numeric(x, errors="coerce").where(
        pd.to_numeric(x, errors="coerce").between(10, 100)
    ).median())
)
print("Age values still missing after imputation:", age_filled.isna().sum())

signup_dates = pd.to_datetime(customers["signup_date"], errors="coerce")
print("Invalid signup dates:", signup_dates.isna().sum())

print("\n" + "=" * 60)
print("ORDER CLEANING COUNTS")
print("=" * 60)

orders = pd.read_csv(RAW / "orders.csv")
products = pd.read_csv(RAW / "products.csv")

print("Raw order rows:", len(orders))

parsed_ts = pd.to_datetime(orders["order_ts"], errors="coerce")
print("Invalid order timestamps:", parsed_ts.isna().sum())

before_duplicates = len(orders)
orders = orders.drop_duplicates().copy()
print("Exact duplicate rows removed:", before_duplicates - len(orders))

before_order_id_dedup = len(orders)
orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
orders = (
    orders.sort_values("order_ts")
    .drop_duplicates(subset="order_id", keep="last")
    .copy()
)
print("Older/repeated order-ID records removed:", before_order_id_dedup - len(orders))

print("Rows remaining after order-ID deduplication:", len(orders))
print("Pending statuses remaining:", orders["status"].eq("pending").sum())

amount_numeric = pd.to_numeric(orders["order_amount"], errors="coerce")
print("Order amounts requiring repair:", amount_numeric.isna().sum())

orders["order_amount"] = amount_numeric
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left",
    validate="many_to_one"
)

missing_amount = orders["order_amount"].isna()
orders.loc[missing_amount, "order_amount"] = (
    orders.loc[missing_amount, "price_inr"]
    * orders.loc[missing_amount, "quantity"]
    * (1 - orders.loc[missing_amount, "discount_pct"] / 100)
).round(2)
print("Order amounts imputed:", missing_amount.sum())

orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

negative_delivery = orders["delivery_minutes"] < 0
print("Negative delivery times invalidated:", negative_delivery.sum())
orders.loc[negative_delivery, "delivery_minutes"] = float("nan")

missing_delivery_before = orders["delivery_minutes"].isna()
print("Missing delivery times before imputation:", missing_delivery_before.sum())

completed = orders["status"].eq("completed")
completed_missing = (
    completed & orders["delivery_minutes"].isna()
)
print("Completed delivery times imputed:", completed_missing.sum())

orders.loc[completed, "delivery_minutes"] = (
    orders.loc[completed]
    .groupby("category")["delivery_minutes"]
    .transform(lambda x: x.fillna(x.median()))
)

print(
    "Cancelled delivery times left missing:",
    orders.loc[orders["status"].eq("cancelled"), "delivery_minutes"].isna().sum()
)

print("\n" + "=" * 60)
print("PRODUCT CLEANING COUNTS")
print("=" * 60)

products = pd.read_csv(RAW / "products.csv")
print("Raw product rows:", len(products))

before_exact = len(products)
products = products.drop_duplicates().copy()
print("Exact duplicate product rows removed:", before_exact - len(products))

for column in ["product_id", "product_name", "category"]:
    cleaned = products[column].astype("string").str.strip()
    print(f"{column} values trimmed/changed:", (products[column].astype("string") != cleaned).sum())
    products[column] = cleaned

price_numeric = pd.to_numeric(products["price_inr"], errors="coerce")
print("Invalid/missing product prices:", price_numeric.isna().sum())
products["price_inr"] = price_numeric

missing_essential = products[
    ["product_id", "product_name", "category", "price_inr"]
].isna().any(axis=1)
print("Rows missing essential product information:", missing_essential.sum())
products = products.dropna(
    subset=["product_id", "product_name", "category", "price_inr"]
)

invalid_price = products["price_inr"] <= 0
print("Rows with non-positive prices removed:", invalid_price.sum())
products = products[~invalid_price].copy()

before_id_dedup = len(products)
products = products.drop_duplicates(
    subset="product_id", keep="first"
).copy()
print("Duplicate product IDs removed:", before_id_dedup - len(products))

print("Final cleaned product rows:", len(products))
print("\nAUDIT COMPLETE")