
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"

with open(SQL_PATH, "r", encoding="utf-8") as file:
    sql_text = file.read()

queries = [
    query.strip()
    for query in sql_text.split(";")
    if query.strip()
]

with sqlite3.connect(DB_PATH) as connection:
    for number, query in enumerate(queries, start=1):
        print(f"\n{'=' * 55}")
        print(f"QUERY {number}")
        print("=" * 55)

        result = pd.read_sql_query(query, connection)
        print(result.to_string(index=False))
