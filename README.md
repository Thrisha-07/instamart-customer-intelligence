
Instamart Customer Intelligence & Support Copilot
Overview
This project builds an offline customer intelligence and support copilot using Python, Pandas, SQLite, scikit-learn, TF-IDF retrieval, and a deterministic mock language model.

The pipeline covers:

Customer, product, order, and ticket data auditing and cleaning
SQL analytics and SQL/Pandas validation
Exploratory analysis and business insights
RFM-based customer segmentation
Customer lapse prediction and model evaluation
Privacy-aware ticket classification and routing
Policy retrieval with and without chunk overlap
A tool-calling refund support agent with checkpoint/resume and safety-limit demonstrations
All model interactions use the supplied mock_llm.py. No live Instamart service, external LLM API, or paid API key is required.

Reproduce
Run all commands from the repository root. The commands below use PowerShell on Windows.

1. Create and activate the environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Use Python 3.11 or later.

2. Generate the raw datasets and policy documents
python generate_data.py
This creates the raw customer, product, order, and ticket datasets and the policy/evaluation files used by later stages.

3. Audit and clean the data
python scripts/audit_data.py
python scripts/audit_details.py
python scripts/clean_customers.py
python scripts/clean_products.py
python scripts/clean_orders.py
python scripts/clean_tickets.py
python scripts/audit_cleaning_counts.py
Cleaned datasets are written under data/clean/.

4. Build the database and run SQL analysis
python scripts/build_database.py
python scripts/run_sql_queries.py
python scripts/validate_sql_pandas.py
The SQLite database is written to data/instamart.db. The validation script compares key SQL results with Pandas calculations and should finish with:

ALL CHECKPOINTS PASSED
5. Run exploratory analysis and customer models
python scripts/exploratory_analysis.py
python scripts/customer_segmentation.py
The segmentation script evaluates K-means configurations, creates customer segments, trains lapse-prediction models, and saves the selected model and retention list.

6. Run ticket triage
python scripts/ticket_triage.py
The script evaluates zero-shot, few-shot, and budget-limited classification and writes the triage outputs under outputs/.

7. Evaluate policy retrieval
python scripts/inspect_evaluation_data.py
python scripts/inspect_policies.py
python scripts/policy_search.py
The policy search script evaluates retrieval with 10-word chunk overlap and with no overlap.

8. Run the support agent and its tests
python scripts/support_agent.py
python scripts/test_agent_resume.py
python scripts/test_agent_limits.py
These commands run the seven support scenarios, a simulated crash-and-resume demonstration, and token, timeout, and maximum-iteration safety-limit demonstrations.

The scripts regenerate files under data/, outputs/, and state/. Run them in the order shown when reproducing the complete workflow from a fresh checkout.

Part A â€” Data auditing, cleaning, and SQL
1. Data audit and cleaning
The raw datasets were audited before cleaning. The final cleaned datasets contain 1,200 customers, 40 products, and 12,309 unique orders. The ticket dataset contains 164 records.

Validated dataset results
Dataset or measure	Result
Clean customers	1,200
Clean products	40
Clean orders	12,309
Completed orders	11,110
Cancelled orders	1,199
Pending orders after cleaning	0
Completed-order revenue	INR 2,998,668.80
Customer cleaning
Standardized city names by trimming whitespace and applying title case. This changed 476 values and resulted in six standardized cities.
Normalized gender values into male, female, and unknown. This changed 950 values, including 30 originally missing gender values.
Converted age values to numeric. There were 61 originally missing ages and 6 ages outside the accepted 10â€“100 range. All 67 missing or invalid ages were imputed using the median age of the respective city.
Converted signup_date to datetime. No invalid signup dates were found.
The final customer dataset contains 1,200 rows with no missing ages.
The cleaned customer records contain 610 female records, 560 male records, and 30 unknown gender records. The final age range is 18â€“60 years.

Order cleaning
Converted order timestamps to datetime. No invalid timestamps were found.
Removed 119 exact duplicate rows.
Kept the latest record for each order_id, removing 656 older or repeated records. The cleaned dataset contains 12,309 unique orders.
No pending statuses remained after latest-record selection.
Converted order_amount to numeric and imputed 112 missing or invalid amounts using the product price, quantity, and discount formula.
Converted delivery times to numeric and invalidated 105 negative delivery-time values.
Before delivery-time imputation, 1,973 delivery-time values were missing.
Imputed 774 missing delivery times for completed orders using the median delivery time of the respective product category.
Left all 1,199 cancelled-order delivery times missing because no delivery time should be invented for cancelled orders.
The final order dataset contains 12,309 rows: 11,110 completed and 1,199 cancelled orders.
Product cleaning
Checked for exact duplicate rows; none were found.
Checked and trimmed whitespace in product IDs, names, and categories; no changes were required.
Converted product prices to numeric; no invalid or missing prices were found.
Checked for missing essential product information and non-positive prices; no rows were removed.
Checked duplicate product IDs; none were found.
The final product dataset contains 40 rows, unchanged after cleaning.
2. SQL analysis and validation
The SQLite database contains three analytical tables: customers, products, and orders. Seven SQL queries were executed.

Query	Analysis	Validated result
Q1	Customers with completed orders above INR 500 and qualifying spend above INR 8,000	30 customers
Q2	Completed revenue by city and product category	30 city/category groups; total INR 2,998,668.80
Q3	Products priced above the average catalogue price	11 products
Q4	Customers with no completed orders	25 customers
Q5	Customers whose completed-order spend is above average customer spend	550 customers
Q6	Customer spend rankings within each city using RANK, DENSE_RANK, and ROW_NUMBER	8 rows where RANK differs from ROW_NUMBER
Q7	Daily completed revenue with cumulative revenue	183 days; final cumulative revenue INR 2,998,668.80
For Q6, RANK and DENSE_RANK assign the same rank to tied spend values, while ROW_NUMBER assigns a unique sequential number. Ties can therefore cause the ranking functions to return different values.

SQL and Pandas results were cross-checked for customer counts, completed orders, total revenue, and city/category revenue. The Pandas pivot-table grand total also matched INR 2,998,668.80. The validation script checks reference counts and query results and prints ALL CHECKPOINTS PASSED when all assertions succeed.

3. Exploratory analysis and business insights
The analysis covers 11,110 completed orders across six cities and five product categories, with total revenue of INR 2,998,668.80.

Order amounts are strongly right-skewed: the mean is INR 269.91, compared with a median of INR 135.00, and skewness is 4.61. The middle 50% of orders range from INR 64.80 to INR 297.00.

Using the 1.5 Ã— IQR method, the upper outlier fence is INR 645.30. There are 897 orders above this fence and none below the lower fence. These high-value orders may represent bulk or unusually expensive purchases and should be investigated rather than automatically removed.

The strongest absolute correlation is between order amount and discount percentage (âˆ’0.095), followed by order amount and quantity (+0.087). Both relationships are weak; correlation does not establish causation.

Bengaluru contributes INR 523,643.40 in completed revenue, while Hyderabad has the highest city revenue at INR 528,589.90. Mumbai has the highest average order value at INR 297.00.

Part B â€” Customer segmentation and lapse prediction
1. RFM analysis
Customer segmentation was performed using Recency, Frequency, and Monetary (RFM) analysis based on completed orders. The analysis included 1,175 customers.

RFM measure	Mean	Median	Minimum	Maximum
Recency (days)	33.56	19	0	180
Frequency	9.46	5	1	33
Monetary (INR)	2,552.06	2,286.50	12.00	14,766.55
2. K-means clustering
K-means clustering was evaluated for K values from 2 to 8 using inertia and silhouette scores.

K	Inertia	Silhouette score
2	1820.76	0.465
3	1268.23	0.445
4	743.07	0.526
5	617.37	0.480
6	503.17	0.482
7	408.46	0.416
8	366.50	0.412
Inertia decreases as K increases because adding clusters reduces within-cluster variation. Inertia alone does not identify the best number of clusters. K=4 was selected because it achieved the highest silhouette score (0.526) among the tested configurations and provided four interpretable business segments.

3. Customer segment profiles
The Premium segment was identified as the cluster with the highest mean monetary value.

Segment	Customers	Mean recency (days)	Mean frequency	Mean monetary (INR)
Occasional Value Customers	436	28.86	3.77	1,057.62
Premium High Spenders	138	24.12	6.77	7,129.70
Loyal Frequent Customers	419	8.77	19.40	3,410.02
Dormant At-Risk Customers	182	109.04	2.20	685.99
Total	1,175
Segment labels were assigned based on observed cluster characteristics. Cluster IDs do not represent an inherent ranking.

Recommended business actions:

Occasional Value Customers: Use relevant product recommendations and modest targeted offers to encourage more frequent purchases.
Premium High Spenders: Prioritize personalized service, relevant offers, and retention outreach when activity declines.
Loyal Frequent Customers: Focus on loyalty benefits, convenience, and complementary product recommendations without over-discounting.
Dormant At-Risk Customers: Use a reactivation campaign with a clear incentive, then monitor whether customers return to purchasing.
The first two principal components explained 85.23% of the variance in the scaled RFM data.

Segmentation outputs:

outputs/customer_segments.csv
outputs/kmeans_metrics.csv
4. Customer lapse prediction
The lapse prediction cohort contains customers with completed orders before 1 September 2025. A customer was labelled as lapsed if they had no completed order during September 2025.

Measure	Result
Cohort customers	1,156
Lapsed customers	443
Lapse rate	38.32%
Training rows	924
Test rows	232
Features were constructed exclusively from pre-cutoff data. They included recency, frequency, monetary value, average discount percentage, cancellation rate, average delivery time, favourite product category, city, age, gender, and customer tenure in days.

A stratified 80/20 train-test split was used with random state 42. Preprocessing, including numerical scaling and categorical encoding, was fitted within training pipelines to avoid fitting preprocessing transformations on the test set.

5. Model comparison and hyperparameter tuning
Logistic Regression and Random Forest were tuned using five-fold GridSearchCV with F1 scoring.

Model	Best parameters	Cross-validation F1	Test F1
Logistic Regression	C = 100	0.651	0.648
Random Forest	Max depth = 8, minimum leaf = 10	0.687	0.640
Random Forest was selected based on its higher cross-validation F1 score. Its untouched test-set F1 was 0.640, exceeding the assignment target of 0.55.

The baseline model predicted every customer as not lapsed and achieved an accuracy of 61.64%. However, its precision, recall, and F1 for the lapsed class were zero.

6. Class balancing experiment
The training data was balanced by oversampling the lapsed class with replacement. Only training data was oversampled; the test set remained unchanged.

Metric	Original Logistic Regression	Balanced Logistic Regression
Precision	0.644	0.589
Recall	0.652	0.854
F1	0.648	0.697
Oversampling improved recall for lapsed customers but reduced precision. This demonstrates the trade-off between identifying more at-risk customers and increasing false-positive predictions.

7. Feature importance and model persistence
The five most important Random Forest features were:

Monetary value
Frequency
Recency
Average delivery time
Average discount percentage
The selected model was saved as outputs/lapse_model.joblib. The saved model was reloaded, and its predictions were compared with the original model. Both sets of predictions matched exactly.

8. Premium customer retention list
The ten Premium customers with the highest predicted lapse probabilities were saved to outputs/retention_list.csv. The output includes customer ID, lapse probability, and actual lapse status.

Limitation: Premium membership is based on whole-window RFM data, including September 2025. Therefore, the retention list is illustrative and should not be interpreted as a leakage-free evaluation of a historical premium-customer targeting strategy.

Part C â€” Privacy-aware ticket triage
The ticket triage pipeline redacts personally identifiable information before calling the mock language model. Email addresses, phone numbers, and order IDs are replaced with [EMAIL], [PHONE], and [ORDER_ID] placeholders.

1. Input guardrails and redaction
Measure	Result
Tickets processed	164
Eligible tickets	156
Empty tickets skipped	4
Long tickets flagged	4
Email matches redacted	55
Phone matches redacted	91
Order ID matches redacted	132
Total redactions	278
Remaining matching PII patterns	0
Empty tickets and tickets exceeding the 400-token input threshold are not sent for classification. Transient errors are retried up to three times, with exponential jitter. In the recorded run, 19 transient failures were recovered through retries.

2. Prompt comparison
Zero-shot and few-shot prompting strategies were evaluated using the mock language model.

Metric	Zero-shot	Few-shot
Classified tickets	147	147
Accuracy among classified tickets	85.7%	100.0%
Total tokens	14,236	27,184
Tokens per classified ticket	96.84	184.93
The zero-shot confusion matrix showed errors for fraud and technical tickets that were predicted as billing. Few-shot examples improved measured classification accuracy to 100% in this deterministic evaluation, at approximately 1.91 times the token usage.

The few-shot output included:

135 valid JSON classifications
12 safe keyword-based classifications
9 invalid outputs
Invalid outputs are not guessed; they are flagged for review. Empty and long tickets are skipped or flagged by the input guardrails.

3. Budget demonstration
With a configured budget of 6,000 tokens, the budget demonstration classified 63 tickets and marked 90 as budget_exhausted. The run recorded 6,022 tokens.

The budget is checked before each model call. A response can take accumulated usage above the configured budget; after that response, the system stops before making another call. The 22-token overshoot is therefore a consequence of checking actual usage after the response completes.

4. Final routing
The final few-shot results are saved to outputs/triage_results.csv. The latest validated run produced the following routing counts:

Route	Tickets
Support queue	102
Security team	34
Premium desk	11
Human review	17
Total	164
Routing follows this priority:

Empty, long, invalid-output, and budget-exhausted tickets go to human review.
Fraud tickets go to the security team with high priority.
Other premium customers go to the premium desk with high priority.
Remaining tickets go to the support queue with normal priority.
The routing counts are run outputs and may change if the input data or deterministic mock-model behavior changes.

Part D â€” Policy retrieval and tool-calling support agent
1. Policy retrieval
Policy documents are split into 40-word chunks with 10-word overlap. A default TfidfVectorizer creates document vectors, and cosine similarity retrieves the top three chunks.

The policy answer function supplies the retrieved context and question to mock_llm.chat(). It instructs the model to use only the provided context and to respond with the policy fallback when the answer is unavailable.

Policy retrieval evaluation
Evaluation measure	10-word overlap	No overlap
Chunks	15	13
Expected document in top three	100.0%	100.0%
Expected fact in answer	100.0%	83.3%
Correct out-of-scope fallback	100.0%	100.0%
The no-overlap evaluation demonstrates a reduction in expected-fact matching. Chunk boundaries can separate relevant policy conditions, while overlap helps preserve context across adjacent chunks.

TF-IDF is a lexical retrieval method. It weights terms by their frequency within a document and rarity across the corpus. A limitation is that semantically related wording can receive low similarity when the query and policy use different vocabulary.

2. ReAct support agent
The support agent uses the following tools:

get_order: Reads order details from SQLite.
search_policy: Retrieves relevant policy chunks.
calculate_refund: Calculates eligibility for a full refund when delivery exceeds 45 minutes.
issue_refund: Records a refund in outputs/refund_ledger.jsonl.
escalate_to_human: Records an escalation in the same ledger.
The agent exchanges tool calls and observations with mock_llm.chat(). Its execution controls include maximum iterations, token limits, timeout fallback, and JSON checkpoint files under state/.

3. Scenario evaluation
Scenario	Expected outcome	Observed outcome	Model calls
S1	REFUND INR 128.25	REFUND	5
S2	REFUND INR 135.00	REFUND	5
S3	ESCALATE INR 1,999.00	ESCALATE	5
S4	DENIED	DENIED	4
S5	DENIED	DENIED	3
S6	NOT_FOUND	NOT_FOUND	2
S7	FALLBACK	FALLBACK_MAX_ITERATIONS	6
The seven scenario outcomes matched the expected outcomes. Refund and escalation actions were recorded for S1, S2, and S3.

4. Safety and resume demonstrations
Demonstration	Observed result
Token limit of 200	FALLBACK_TOKEN_LIMIT
Zero-second time limit	FALLBACK_TIMEOUT
Six-iteration loop scenario	FALLBACK_MAX_ITERATIONS
Crash after second tool execution, followed by resume	Final outcome REFUND INR 128.25
The resume demonstration recorded the tool sequence:

get_order
search_policy
calculate_refund
issue_refund
The checkpoint was saved after the second tool observation. The resumed run completed with a refund of INR 128.25 and recorded the expected four-tool sequence without repeating a tool execution in the demonstrated run.

Further testing is needed to validate cumulative token accounting and prevent duplicate ledger writes across repeated restarts. The token-limit implementation can exceed the configured limit by one model response because actual response usage is known only after the call completes.

5. ReAct versus plan-and-execute
ReAct interleaves model reasoning with tool execution and observations. This allows it to respond to missing orders, cancelled orders, or policy results as they are discovered.

Plan-and-execute creates a plan first and then executes it. This can be more efficient for predictable workflows but may require replanning when observations differ from expectations.

For this support workflow, ReAct is a useful fit because decisions depend on tool observations from the local database and policy retrieval. Its additional model calls and token usage should be monitored, and high-impact actions should retain human approval and audit controls.

Limitations and Next Steps
The project uses deterministic offline mock-model behavior rather than a live hosted LLM.
The policy evaluation contains 14 questions. Broader and independently authored evaluation data would provide stronger evidence of generalization.
The budget guard checks usage before each call. The final response can cause a small overshoot because actual token usage is only known after the response completes.
Checkpoint/resume passed the simulated crash-and-recovery test; duplicate ledger writes and cumulative token accounting still need broader testing.
Refund actions are appended to a local JSONL ledger; this is not a production payment integration.
The agent covers seven predefined scenarios and should be tested with more edge cases before deployment.
The Premium retention list uses whole-window RFM information, including the lapse-label period, and is illustrative rather than a leakage-free historical evaluation.
The system does not connect to live Instamart services.
