
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

files = {
    "customers": RAW / "customers.csv",
    "products": RAW / "products.csv",
    "orders": RAW / "orders.csv",
    "tickets": RAW / "tickets.csv",
}

for name, path in files.items():
    print("\n" + "=" * 60)
    print(f"DATASET: {name.upper()}")
    print("=" * 60)

    if name == "tickets":
        df = pd.read_csv(path, keep_default_na=False)
    else:
        df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values by column:")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
    })
    print(missing.to_string())

print("\nAudit inspection completed.")
