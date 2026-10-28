
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"

datasets = {
    "customers": CLEAN / "customers.csv",
    "products": CLEAN / "products.csv",
    "orders": CLEAN / "orders.csv",
}

with sqlite3.connect(DB_PATH) as connection:
    for table_name, csv_path in datasets.items():
        df = pd.read_csv(csv_path)
        df.to_sql(table_name, connection, if_exists="replace", index=False)

        row_count = pd.read_sql_query(
            f"SELECT COUNT(*) AS count FROM {table_name}",
            connection
        )["count"].iloc[0]

        print(f"{table_name}: {row_count} rows")

    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        connection
    )

    print("\nTables in database:")
    print(tables.to_string(index=False))

print(f"\nDatabase saved to: {DB_PATH}")
