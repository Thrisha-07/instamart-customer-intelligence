from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "instamart.db"

with sqlite3.connect(DB_PATH) as connection:
    customers = pd.read_sql_query(
        "SELECT * FROM customers", connection
    )
    products = pd.read_sql_query(
        "SELECT * FROM products", connection
    )
    orders = pd.read_sql_query(
        "SELECT * FROM orders", connection
    )

    # Checkpoint 1: Customer count
    assert len(customers) == 1200

    # Checkpoint 2: Product count
    assert len(products) == 40

    # Checkpoint 3: Total cleaned orders
    assert len(orders) == 12309

    # Checkpoint 4: Completed orders
    completed = orders[orders["status"] == "completed"].copy()
    assert len(completed) == 11110

    # Checkpoint 5: Cancelled orders
    cancelled = orders[orders["status"] == "cancelled"]
    assert len(cancelled) == 1199

    # Checkpoint 6: No pending orders remain
    assert not orders["status"].eq("pending").any()

    # Checkpoint 7: No missing order amounts
    assert orders["order_amount"].isna().sum() == 0

    # Checkpoint 8: Completed orders have delivery times
    assert completed["delivery_minutes"].isna().sum() == 0

    # Checkpoint 9: Cancelled orders have missing delivery times
    assert cancelled["delivery_minutes"].isna().sum() == 1199

    # Checkpoint 10: Completed revenue
    sql_revenue = pd.read_sql_query(
        """
        SELECT ROUND(SUM(order_amount), 2) AS revenue
        FROM orders
        WHERE status = 'completed'
        """,
        connection
    )["revenue"].iloc[0]

    pandas_revenue = round(completed["order_amount"].sum(), 2)

    assert abs(sql_revenue - 2998668.80) < 0.01
    assert abs(sql_revenue - pandas_revenue) < 0.01

    # Checkpoint 11: SQL and Pandas city/category revenue
    sql_groups = pd.read_sql_query(
        """
        SELECT c.city, p.category,
               ROUND(SUM(o.order_amount), 2) AS revenue
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN products p ON o.product_id = p.product_id
        WHERE o.status = 'completed'
        GROUP BY c.city, p.category
        """,
        connection
    )

    # SQL Query 1: Customers with qualifying high-value spend
    q1 = pd.read_sql_query(
        """
        SELECT customer_id, SUM(order_amount) AS total_spend
        FROM orders
        WHERE status = 'completed' AND order_amount > 500
        GROUP BY customer_id
        HAVING SUM(order_amount) > 8000
        """,
        connection
    )
    assert len(q1) == 30

    # SQL Query 2: City/category revenue
    assert len(sql_groups) == 30
    assert abs(sql_groups["revenue"].sum() - 2998668.80) < 0.01

    # SQL Query 3: Products above average catalogue price
    q3 = pd.read_sql_query(
        """
        SELECT product_id
        FROM products
        WHERE price_inr > (SELECT AVG(price_inr) FROM products)
        """,
        connection
    )
    assert len(q3) == 11

    # SQL Query 4: Customers without completed orders
    q4 = pd.read_sql_query(
        """
        SELECT c.customer_id
        FROM customers c
        LEFT JOIN orders o
          ON c.customer_id = o.customer_id
         AND o.status = 'completed'
        WHERE o.order_id IS NULL
        """,
        connection
    )
    assert len(q4) == 25

    # SQL Query 5: Customers above average completed spend
    q5 = pd.read_sql_query(
        """
        WITH customer_spend AS (
            SELECT customer_id, SUM(order_amount) AS total_spend
            FROM orders
            WHERE status = 'completed'
            GROUP BY customer_id
        )
        SELECT customer_id
        FROM customer_spend
        WHERE total_spend > (SELECT AVG(total_spend) FROM customer_spend)
        """,
        connection
    )
    assert len(q5) == 550

    # SQL Query 6: Ranking tie check
    q6 = pd.read_sql_query(
        """
        WITH customer_spend AS (
            SELECT c.city, o.customer_id,
                   SUM(o.order_amount) AS total_spend
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            WHERE o.status = 'completed'
            GROUP BY c.city, o.customer_id
        ),
        ranked AS (
            SELECT city, customer_id, total_spend,
                   RANK() OVER (
                       PARTITION BY city ORDER BY total_spend DESC
                   ) AS spend_rank,
                   ROW_NUMBER() OVER (
                       PARTITION BY city
                       ORDER BY total_spend DESC, customer_id
                   ) AS row_number
            FROM customer_spend
        )
        SELECT *
        FROM ranked
        """,
        connection
    )
    assert (q6["spend_rank"] != q6["row_number"]).sum() == 8

    # SQL Query 7: Daily cumulative revenue
    q7 = pd.read_sql_query(
        """
        WITH daily_revenue AS (
            SELECT SUBSTR(order_ts, 1, 10) AS order_day,
                   SUM(order_amount) AS daily_revenue
            FROM orders
            WHERE status = 'completed'
            GROUP BY SUBSTR(order_ts, 1, 10)
        )
        SELECT order_day, daily_revenue,
               SUM(daily_revenue) OVER (
                   ORDER BY order_day
               ) AS cumulative_revenue
        FROM daily_revenue
        ORDER BY order_day
        """,
        connection
    )

    assert len(q7) == 183
    assert abs(q7["cumulative_revenue"].iloc[-1] - 2998668.80) < 0.01

# Checkpoint 12: Pandas pivot-table grand total
analysis = (
    completed
    .merge(customers[["customer_id", "city"]], on="customer_id")
    .merge(products[["product_id", "category"]], on="product_id")
)

pivot = pd.pivot_table(
    analysis,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    fill_value=0,
    margins=True
)

assert abs(pivot.loc["All", "All"] - 2998668.80) < 0.01

# Checkpoint 13: SQL/Pandas group-level revenue comparison
pandas_groups = (
    analysis.groupby(["city", "category"])["order_amount"]
    .sum()
    .reset_index(name="pandas_revenue")
)

comparison = sql_groups.merge(
    pandas_groups,
    on=["city", "category"]
)

assert len(comparison) == 30
assert (
    comparison["revenue"] - comparison["pandas_revenue"]
).abs().max() < 0.01

print("ALL CHECKPOINTS PASSED")
