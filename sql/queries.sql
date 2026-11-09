
-- QUERY 1: Customers with completed orders above INR 500
-- and total completed spending above INR 8,000
SELECT
    customer_id,
    COUNT(*) AS completed_orders,
    ROUND(SUM(order_amount), 2) AS total_spending_inr
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spending_inr DESC;


-- QUERY 2: Revenue by city and product category
-- Uses customers, orders and products tables
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS total_revenue_inr
FROM orders AS o
JOIN customers AS c
    ON o.customer_id = c.customer_id
JOIN products AS p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;


-- QUERY 3: Products priced above the average catalogue price
-- Uses a scalar subquery
SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;


-- QUERY 4: Customers who have never placed a completed order
-- Uses LEFT JOIN
SELECT
    c.customer_id,
    c.city,
    c.gender
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customer_id = o.customer_id
   AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;


-- QUERY 5: Customers whose completed spending is above average
-- Uses a CTE
WITH customer_spending AS (
    SELECT
        customer_id,
        ROUND(SUM(order_amount), 2) AS total_spending_inr
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spending AS (
    SELECT AVG(total_spending_inr) AS average_spending
    FROM customer_spending
)
SELECT
    cs.customer_id,
    cs.total_spending_inr,
    ROUND(a.average_spending, 2) AS average_spending_inr
FROM customer_spending AS cs
CROSS JOIN average_spending AS a
WHERE cs.total_spending_inr > a.average_spending
ORDER BY cs.total_spending_inr DESC;


-- QUERY 6: Rank customers by completed spending within each city
-- Uses RANK, DENSE_RANK and ROW_NUMBER
SELECT
    c.city,
    o.customer_id,
    ROUND(SUM(o.order_amount), 2) AS total_spending_inr,
    RANK() OVER (
        PARTITION BY c.city
        ORDER BY SUM(o.order_amount) DESC
    ) AS spending_rank,
    DENSE_RANK() OVER (
        PARTITION BY c.city
        ORDER BY SUM(o.order_amount) DESC
    ) AS dense_spending_rank,
    ROW_NUMBER() OVER (
        PARTITION BY c.city
        ORDER BY SUM(o.order_amount) DESC, o.customer_id
    ) AS spending_row_number
FROM orders AS o
JOIN customers AS c
    ON o.customer_id = c.customer_id
WHERE o.status = 'completed'
GROUP BY c.city, o.customer_id
ORDER BY c.city, total_spending_inr DESC;


-- QUERY 7: Daily completed revenue and cumulative revenue
-- Uses a windowed SUM
WITH daily_revenue AS (
    SELECT
        SUBSTR(order_ts, 1, 10) AS order_date,
        ROUND(SUM(order_amount), 2) AS daily_revenue_inr
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    order_date,
    daily_revenue_inr,
    ROUND(
        SUM(daily_revenue_inr) OVER (
            ORDER BY order_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        2
    ) AS cumulative_revenue_inr
FROM daily_revenue
ORDER BY order_date;
