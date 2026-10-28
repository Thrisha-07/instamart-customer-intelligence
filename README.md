
# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds an offline customer intelligence and support copilot using Python, Pandas, SQLite, scikit-learn, TF-IDF retrieval, and a deterministic mock language model.

The pipeline covers customer and order data cleaning, SQL analytics, customer segmentation, privacy-aware ticket classification, policy retrieval, and a tool-calling refund support agent.

All model interactions use the supplied `mock_llm.py`. No live Instamart service or external LLM API is required.

## Reproduce

Run the commands from the repository root in PowerShell.

### 1. Set up the environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Generate datasets and policy documents

```powershell
python generate_data.py
```

### 3. Audit and clean data

```powershell
python scripts/audit_data.py
python scripts/audit_details.py
python scripts/clean_customers.py
python scripts/clean_products.py
python scripts/clean_orders.py
python scripts/clean_tickets.py
```

### 4. Build and validate the database

```powershell
python scripts/build_database.py
python scripts/run_sql_queries.py
python scripts/validate_sql_pandas.py
python scripts/customer_segmentation.py
```

### 5. Run ticket triage

```powershell
python scripts/ticket_triage.py
```

Outputs include:

- `outputs/triage_zero_shot.csv`
- `outputs/triage_few_shot.csv`
- `outputs/triage_budget_demo.csv`
- `outputs/triage_results.csv`

### 6. Evaluate policy retrieval

```powershell
python scripts/inspect_evaluation_data.py
python scripts/inspect_policies.py
python scripts/policy_search.py
```

Outputs include:

- `outputs/policy_search_results.csv`
- `outputs/policy_search_no_overlap_results.csv`

### 7. Run the support agent

```powershell
python scripts/support_agent.py
python scripts/test_agent_resume.py
python scripts/test_agent_limits.py
```

These commands run the seven support scenarios and the checkpoint/resume and safety-limit demonstrations.

## Part A — Data auditing, cleaning, and SQL

Raw customer, product, order, and ticket datasets were audited and cleaned. Cleaned datasets are stored under `data/clean/`, and the SQLite database is stored at `data/instamart.db`.

### Validated data results

| Dataset or measure | Result |
|---|---:|
| Clean customers | 1,200 |
| Clean products | 40 |
| Clean orders | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |
| Pending orders after cleaning | 0 |
| Completed-order revenue | ₹2,998,668.80 |

The cleaned customer data contains 610 female, 560 male, and 30 unknown gender records. Ages are complete and range from 18 to 60.

SQL query results were cross-checked against Pandas calculations for customer counts, completed orders, and completed-order revenue.

## Part B — Customer segmentation

Customer segments were generated using completed-order spending and order frequency. The top 20% spending threshold identifies 240 premium customers.

| Segment | Customers |
|---|---:|
| Regular | 960 |
| Loyal premium | 218 |
| High value | 22 |
| Total | 1,200 |

The segment output is saved to `outputs/customer_segments.csv`. The `is_premium` field is used by ticket routing to prioritize eligible customers.

## Part C — Privacy-aware ticket triage

The ticket triage pipeline redacts personally identifiable information before calling the mock language model. Email addresses, phone numbers, and order IDs are replaced with `[EMAIL]`, `[PHONE]`, and `[ORDER_ID]`.

### Input guardrails and redaction

| Measure | Result |
|---|---:|
| Tickets processed | 164 |
| Eligible tickets | 156 |
| Empty tickets skipped | 4 |
| Long tickets flagged | 4 |
| Email matches redacted | 55 |
| Phone matches redacted | 91 |
| Order ID matches redacted | 132 |
| Total redactions | 278 |
| Remaining matching PII patterns | 0 |

Empty tickets and tickets exceeding the 400-token input threshold are not sent for classification. Transient errors are retried up to three times, with exponential jitter.

### Prompt comparison

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Accuracy among classified tickets | 85.7% | 100.0% |
| Total tokens | 14,236 | 27,184 |
| Tokens per classified ticket | 96.84 | 184.93 |

The zero-shot confusion matrix showed errors for fraud and technical tickets that were predicted as billing. Few-shot examples improved measured classification accuracy to 100% in this deterministic evaluation, at approximately 1.91 times the token usage.

The few-shot output includes 135 valid JSON classifications, 12 safe keyword-based classifications, and 9 invalid outputs. Invalid outputs are not guessed; they are flagged for review.

### Budget demonstration

With a configured budget of 6,000 tokens, the budget demonstration classified 63 tickets and marked 90 as `budget_exhausted`. It recorded 6,022 tokens because the current implementation checks accumulated usage before a call and checks the updated total after the response. This overshoot is a known limitation to correct before final submission.

### Final routing

The final few-shot results are saved to `outputs/triage_results.csv`.

| Route | Tickets |
|---|---:|
| Support queue | 90 |
| Security team | 34 |
| Premium desk | 23 |
| Human review | 17 |

Invalid, empty, long, and budget-exhausted tickets are routed to human review. Fraud tickets are prioritized for the security team. Other premium customers are routed to the premium desk.

## Part D — Policy retrieval and tool-calling support agent

### Policy retrieval

Policy documents are split into 40-word chunks with 10-word overlap. A default `TfidfVectorizer` creates document vectors, and cosine similarity retrieves the top three chunks.

The policy answer function supplies the retrieved context and question to `mock_llm.chat()`. It instructs the model to use only the provided context and to respond with the policy fallback when the answer is unavailable.

| Evaluation measure | 10-word overlap | No overlap |
|---|---:|---:|
| Chunks | 15 | 13 |
| Expected document in top three | 100.0% | 100.0% |
| Expected fact in answer | 100.0% | 83.3% |
| Correct out-of-scope fallback | 100.0% | 100.0% |

The no-overlap evaluation demonstrates a reduction in expected-fact matching. Chunk boundaries can separate relevant policy conditions, while overlap helps preserve context across adjacent chunks.

TF-IDF is a lexical retrieval method: it weights terms by their frequency within a document and rarity across the corpus. A limitation is that semantically related wording can receive low similarity when the query and policy use different vocabulary.

### ReAct support agent

The support agent uses these tools:

- `get_order`: reads order details from SQLite.
- `search_policy`: retrieves relevant policy chunks.
- `calculate_refund`: calculates eligibility for a full refund when delivery exceeds 45 minutes.
- `issue_refund`: records a refund in `outputs/refund_ledger.jsonl`.
- `escalate_to_human`: records an escalation in the same ledger.

The agent exchanges tool calls and observations with `mock_llm.chat()`. Its execution controls include maximum iterations, token limits, timeout fallback, and JSON checkpoint files under `state/`.

### Scenario evaluation

| Scenario | Expected outcome | Observed outcome | Model calls |
|---|---|---|---:|
| S1 | REFUND ₹128.25 | REFUND | 5 |
| S2 | REFUND ₹135.00 | REFUND | 5 |
| S3 | ESCALATE ₹1,999.00 | ESCALATE | 5 |
| S4 | DENIED | DENIED | 4 |
| S5 | DENIED | DENIED | 3 |
| S6 | NOT_FOUND | NOT_FOUND | 2 |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS | 6 |

The seven scenario outcomes matched the expected outcomes. Refund and escalation actions were recorded for S1, S2, and S3.

### Safety and resume demonstrations

| Demonstration | Observed result |
|---|---|
| Token limit of 200 | `FALLBACK_TOKEN_LIMIT` |
| Zero-second time limit | `FALLBACK_TIMEOUT` |
| Six-iteration loop scenario | `FALLBACK_MAX_ITERATIONS` |
| Crash after second tool execution, followed by resume | Final outcome REFUND ₹128.25 |

The resume demonstration recorded the tool sequence `get_order`, `search_policy`, `calculate_refund`, and `issue_refund`. The checkpoint was saved after the second tool observation.

The checkpoint/resume demonstration successfully recovered after a simulated crash following the second tool observation. The resumed run completed with a refund of INR 128.25 and recorded the expected four-tool sequence without repeating a tool execution in the demonstrated run. Further testing is needed to validate cumulative token accounting and prevent duplicate ledger writes across repeated restarts.

The token-limit implementation can exceed the configured limit by one model response because actual response usage is known only after the call completes.

### ReAct versus plan-and-execute

ReAct interleaves model reasoning with tool execution and observations. This allows it to respond to missing orders, cancelled orders, or policy results as they are discovered. Plan-and-execute creates a plan first and then executes it, which can be more efficient for predictable workflows but may need replanning when observations differ from expectations.

For this support workflow, ReAct is a useful fit because decisions depend on live tool observations from the local database and policy retrieval. Its additional model calls and token usage should be monitored, and high-impact actions should retain human approval and audit controls.

## Limitations and Next Steps

- The project uses deterministic offline mock-model behavior rather than a live hosted LLM.
- The policy evaluation contains 14 questions; broader and independently authored evaluation data would provide stronger evidence of generalization.
- The budget guard currently allows a response to exceed the configured token limit before stopping.
- Checkpoint/resume passed the simulated crash-and-recovery test; duplicate ledger writes and cumulative token accounting still need broader testing.
- Refund actions are appended to a local JSONL ledger; this is not a production payment integration.
- The agent covers seven predefined scenarios and should be tested with more edge cases before deployment.
- The system does not connect to live Instamart services.
