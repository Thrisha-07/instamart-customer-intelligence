
from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "instamart.db"

with sqlite3.connect(DB_PATH) as connection:
    # Customer count: SQL vs Pandas
    sql_customers = pd.read_sql_query(
        "SELECT COUNT(*) AS total FROM customers",
        connection
    )["total"].iloc[0]

    pandas_customers = pd.read_sql_query(
        "SELECT * FROM customers", connection
    ).shape[0]

    # Completed orders and revenue: SQL vs Pandas
    sql_orders = pd.read_sql_query(
        """
        SELECT COUNT(*) AS total_orders,
               ROUND(SUM(order_amount), 2) AS revenue
        FROM orders
        WHERE status = 'completed'
        """,
        connection
    ).iloc[0]

    orders = pd.read_sql_query(
        "SELECT status, order_amount FROM orders",
        connection
    )
    completed = orders[orders["status"] == "completed"]

    pandas_orders = len(completed)
    pandas_revenue = round(completed["order_amount"].sum(), 2)

print("CUSTOMER COUNT")
print("SQL:", sql_customers)
print("Pandas:", pandas_customers)
print("Match:", sql_customers == pandas_customers)

print("\nCOMPLETED ORDERS")
print("SQL:", sql_orders["total_orders"])
print("Pandas:", pandas_orders)
print("Match:", sql_orders["total_orders"] == pandas_orders)

print("\nCOMPLETED ORDER REVENUE")
print("SQL: ₹", f"{sql_orders['revenue']:,.2f}")
print("Pandas: ₹", f"{pandas_revenue:,.2f}")
print("Match:", abs(sql_orders["revenue"] - pandas_revenue) < 0.01)
