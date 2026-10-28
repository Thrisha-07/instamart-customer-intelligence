
-- 1. Total customers
SELECT COUNT(*) AS total_customers
FROM customers;

-- 2. Customers by city
SELECT city, COUNT(*) AS customer_count
FROM customers
GROUP BY city
ORDER BY customer_count DESC;

-- 3. Customer distribution by gender
SELECT gender, COUNT(*) AS customer_count
FROM customers
GROUP BY gender
ORDER BY customer_count DESC;

-- 4. Total products by category
SELECT category, COUNT(*) AS product_count
FROM products
GROUP BY category
ORDER BY product_count DESC;

-- 5. Orders by status
SELECT status, COUNT(*) AS order_count
FROM orders
GROUP BY status
ORDER BY order_count DESC;

-- 6. Completed orders and total revenue
SELECT
    COUNT(*) AS completed_orders,
    ROUND(SUM(order_amount), 2) AS total_revenue_inr,
    ROUND(AVG(order_amount), 2) AS average_order_value_inr
FROM orders
WHERE status = 'completed';

-- 7. Top 10 customers by completed-order spending
SELECT
    customer_id,
    COUNT(*) AS completed_orders,
    ROUND(SUM(order_amount), 2) AS total_spending_inr
FROM orders
WHERE status = 'completed'
GROUP BY customer_id
ORDER BY total_spending_inr DESC
LIMIT 10;

-- 8. Average delivery time by product category
SELECT
    p.category,
    ROUND(AVG(o.delivery_minutes), 2) AS avg_delivery_minutes,
    COUNT(o.delivery_minutes) AS orders_with_delivery_data
FROM orders AS o
JOIN products AS p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY p.category
ORDER BY avg_delivery_minutes DESC;
