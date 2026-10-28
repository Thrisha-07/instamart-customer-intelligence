
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

products = pd.read_csv(RAW / "products.csv")

# Remove exact duplicate rows
products = products.drop_duplicates().copy()

# Clean text columns
for column in ["product_id", "product_name", "category"]:
    products[column] = products[column].astype("string").str.strip()

# Convert prices to numeric
products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="coerce"
)

# Remove rows missing essential product information
products = products.dropna(
    subset=["product_id", "product_name", "category", "price_inr"]
)

# Keep only valid positive prices
products = products[products["price_inr"] > 0].copy()

# Ensure each product ID appears only once
products = products.drop_duplicates(
    subset="product_id", keep="first"
).copy()

products = products.sort_values("product_id").reset_index(drop=True)

products.to_csv(CLEAN / "products.csv", index=False)

print("Clean products shape:", products.shape)
print("Duplicate product IDs:", products["product_id"].duplicated().sum())
print("Missing values:")
print(products.isna().sum().to_string())
print("\nSaved to:", CLEAN / "products.csv")
