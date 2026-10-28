
from pathlib import Path
import json
import random
import re
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mock_llm

RAW_PATH = ROOT / "data" / "raw" / "tickets.csv"
MAX_INPUT_TOKENS = 400

ZERO_SHOT_PROMPT = """
CLASSIFY each customer support ticket into exactly one allowed category:
billing, delivery, technical, fraud.

Use only the supplied ticket text. Return JSON only, with this exact form:
{"category": "...", "reason": "..."}
The category must be one of: billing, delivery, technical, fraud.
Do not include markdown or additional text.
"""

FEW_SHOT_PROMPT = """
CLASSIFY each customer support ticket into exactly one allowed category:
billing, delivery, technical, fraud.

EXAMPLES:
Ticket: "My invoice shows the wrong amount and I was charged twice."
Reply: {"category": "billing", "reason": "Incorrect invoice and duplicate charge."}

Ticket: "Someone made an unauthorized transaction on my card."
Reply: {"category": "fraud", "reason": "Unauthorized transaction."}

Ticket: "The app freezes whenever I try to log in."
Reply: {"category": "technical", "reason": "The app freezes during login."}

Ticket: "My order arrived much later than promised."
Reply: {"category": "delivery", "reason": "Late delivery."}

Return JSON only, with this exact form:
{"category": "...", "reason": "..."}
The category must be one of: billing, delivery, technical, fraud.
Do not include markdown or additional text.
"""

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}
BASE_DELAY = 0.05
MAX_RETRIES = 3

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_PATTERN = re.compile(r"ORD-\d{6}")


def call_with_retry(messages, ticket_id):
    """Retry transient model errors up to three times."""
    for retry_number in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry_number >= MAX_RETRIES:
                raise

            wait = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"RETRY ticket_id={ticket_id} "
                f"retry_number={retry_number} wait={wait:.4f}s"
            )
            time.sleep(wait)


def validate_response(content):
    """Validate JSON first, then safely try keyword extraction."""
    try:
        data = json.loads(content or "")
    except (json.JSONDecodeError, TypeError):
        data = None

    if isinstance(data, dict):
        category = data.get("category")

        if category in ALLOWED_CATEGORIES:
            return category, "classified_json", data.get("reason", "")

        return "", "flagged_invalid_output", "Invalid JSON category."

    words = re.findall(r"[a-z]+", (content or "").lower())
    found = {word for word in words if word in ALLOWED_CATEGORIES}

    if len(found) == 1:
        return (
            next(iter(found)),
            "classified_keyword",
            "Category extracted from text.",
        )

    return (
        "",
        "flagged_invalid_output",
        "Could not safely validate model output.",
    )


def redact_text(text):
    """Redact email, phone, then order ID."""
    text = str(text)

    text, email_count = EMAIL_PATTERN.subn("[EMAIL]", text)
    text, phone_count = PHONE_PATTERN.subn("[PHONE]", text)
    text, order_count = ORDER_PATTERN.subn("[ORDER_ID]", text)

    counts = {
        "emails": email_count,
        "phones": phone_count,
        "order_ids": order_count,
    }

    return text, counts


def remaining_pii(text):
    """Count sensitive patterns remaining after redaction."""
    return {
        "emails": len(EMAIL_PATTERN.findall(text)),
        "phones": len(PHONE_PATTERN.findall(text)),
        "order_ids": len(ORDER_PATTERN.findall(text)),
    }


def prepare_tickets():
    """Load tickets, redact PII, and apply input guardrails."""
    tickets = pd.read_csv(RAW_PATH, keep_default_na=False)

    prepared = []
    totals = {"emails": 0, "phones": 0, "order_ids": 0}

    for _, row in tickets.iterrows():
        original_text = str(row["text"])
        redacted_text, counts = redact_text(original_text)

        for key, count in counts.items():
            totals[key] += count

        leftovers = remaining_pii(redacted_text)
        assert sum(leftovers.values()) == 0, (
            f"PII remains in ticket {row['ticket_id']}: {leftovers}"
        )

        if not original_text.strip():
            status = "skipped_empty"
        elif mock_llm.est_tokens(original_text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        else:
            status = "ready"

        prepared.append({
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "order_id": row["order_id"],
            "text": redacted_text,
            "true_category": row["true_category"],
            "status": status,
        })

    prepared_df = pd.DataFrame(prepared)

    print("Total tickets:", len(prepared_df))
    print("\nInput guardrail statuses:")
    print(prepared_df["status"].value_counts().to_string())

    print("\nRedaction counts:")
    for key, count in totals.items():
        print(f"{key}: {count}")

    print("total:", sum(totals.values()))

    leftovers_total = {"emails": 0, "phones": 0, "order_ids": 0}

    for text in prepared_df["text"]:
        leftovers = remaining_pii(text)
        for key, count in leftovers.items():
            leftovers_total[key] += count

    print("\nRemaining PII pattern matches:")
    print(leftovers_total)

    assert len(prepared_df) == 164
    assert (prepared_df["status"] == "skipped_empty").sum() == 4
    assert (prepared_df["status"] == "flagged_long").sum() == 4
    assert totals == {
        "emails": 55,
        "phones": 91,
        "order_ids": 132,
    }
    assert sum(leftovers_total.values()) == 0

    print("\nGuardrail and redaction checks passed.")

    return prepared_df


def classify_tickets(prompt, token_budget=None):
    """Classify eligible tickets and track token usage."""
    prepared = prepare_tickets()
    mock_llm.reset()

    results = []
    total_tokens = 0

    for _, ticket in prepared.iterrows():
        ticket_id = ticket["ticket_id"]
        status = ticket["status"]
        category = ""
        reason = ""
        tokens_used = 0

        if status == "ready":
            if token_budget is not None and total_tokens >= token_budget:
                status = "budget_exhausted"
            else:
                leftovers = remaining_pii(ticket["text"])
                assert sum(leftovers.values()) == 0, (
                    f"PII remains in ticket {ticket_id}"
                )

                # The user message contains only the redacted ticket text.
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": ticket["text"]},
                ]

                response = call_with_retry(messages, ticket_id)
                tokens_used = response.usage.total_tokens
                total_tokens += tokens_used

                category, status, reason = validate_response(
                    response.choices[0].message.content
                )

        results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "order_id": ticket["order_id"],
            "true_category": ticket["true_category"],
            "status": status,
            "category": category,
            "reason": reason,
            "tokens_used": tokens_used,
        })

    results_df = pd.DataFrame(results)

    classified = results_df[
        results_df["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]

    accuracy = (
        (classified["category"] == classified["true_category"]).mean()
        if not classified.empty
        else 0.0
    )

    print("\nClassification summary")
    print("----------------------")
    print(results_df["status"].value_counts().to_string())
    print(f"Classified tickets: {len(classified)}")
    print(f"Accuracy among classified: {accuracy:.1%}")
    print(f"Total tokens: {total_tokens}")

    if len(classified):
        print(
            "Tokens per classified ticket: "
            f"{total_tokens / len(classified):.2f}"
        )
    else:
        print("Tokens per classified ticket: N/A")

    return results_df, total_tokens


def print_comparison(name, results, total_tokens):
    """Print accuracy and token usage for one run."""
    classified = results[
        results["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]

    accuracy = (
        (classified["category"] == classified["true_category"]).mean()
        if not classified.empty
        else 0.0
    )

    tokens_per_ticket = (
        f"{total_tokens / len(classified):.2f}"
        if len(classified)
        else "N/A"
    )

    print(
        f"{name}: classified={len(classified)}, "
        f"accuracy={accuracy:.1%}, tokens={total_tokens}, "
        f"tokens/classified={tokens_per_ticket}"
    )


def main():
    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Zero-shot comparison run
    print("\n========== ZERO-SHOT RUN ==========")
    zero_results, zero_tokens = classify_tickets(ZERO_SHOT_PROMPT)

    zero_results.to_csv(
        outputs_dir / "triage_zero_shot.csv",
        index=False,
    )

    # Token budget demonstration
    print("\n========== TOKEN BUDGET DEMO ==========")
    budget_results, budget_tokens = classify_tickets(
        ZERO_SHOT_PROMPT,
        token_budget=6000,
    )

    budget_results.to_csv(
        outputs_dir / "triage_budget_demo.csv",
        index=False,
    )

    exhausted_count = (
        budget_results["status"] == "budget_exhausted"
    ).sum()

    print(f"Budget demo tokens used: {budget_tokens}")
    print(f"Budget-exhausted tickets: {exhausted_count}")

    # Few-shot comparison run
    print("\n========== FEW-SHOT RUN ==========")
    few_results, few_tokens = classify_tickets(FEW_SHOT_PROMPT)

    few_results.to_csv(
        outputs_dir / "triage_few_shot.csv",
        index=False,
    )

    create_final_triage_results(few_results)

    # Compare both prompts
    print("\n========== COMPARISON ==========")
    print_comparison("Zero-shot", zero_results, zero_tokens)
    print_comparison("Few-shot", few_results, few_tokens)

    # Zero-shot confusion matrix
    print("\nZero-shot confusion matrix:")

    zero_classified = zero_results[
        zero_results["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]

    print(
        pd.crosstab(
            zero_classified["true_category"],
            zero_classified["category"],
            rownames=["Actual"],
            colnames=["Predicted"],
        )
    )

def create_final_triage_results(few_results):
    """Route few-shot ticket results using customer segments."""
    outputs_dir = ROOT / "outputs"
    segments_path = outputs_dir / "customer_segments.csv"

    if not segments_path.exists():
        raise FileNotFoundError(
            "Missing customer_segments.csv. "
            "Run scripts/customer_segmentation.py first."
        )

    segments = pd.read_csv(segments_path)
    segments = segments[
        ["customer_id", "segment_name", "is_premium"]
    ].copy()

    segments["is_premium"] = (
        segments["is_premium"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    final = few_results.merge(
        segments,
        on="customer_id",
        how="left",
    )

    final["is_premium"] = final["is_premium"].fillna(False)
    final["segment_name"] = final["segment_name"].fillna("unknown")

    final["priority"] = "NORMAL"
    final["route"] = "support_queue"

    human_review = {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    }

    for index, row in final.iterrows():
        if row["status"] in human_review:
            final.at[index, "route"] = "human_review"
        elif row["category"] == "fraud":
            final.at[index, "priority"] = "HIGH"
            final.at[index, "route"] = "security_team"
        elif row["is_premium"]:
            final.at[index, "priority"] = "HIGH"
            final.at[index, "route"] = "premium_desk"

    columns = [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used",
    ]

    final[columns].to_csv(
        outputs_dir / "triage_results.csv",
        index=False,
    )

    print("\n========== FINAL TRIAGE ROUTING ==========")
    print("Saved:", outputs_dir / "triage_results.csv")
    print("\nRouting counts:")
    print(final["route"].value_counts().to_string())

    return final[columns]


if __name__ == "__main__":
    main()



def create_final_triage_results(few_results):
    """Route few-shot ticket results using customer segments."""
    outputs_dir = ROOT / "outputs"

    segments_path = outputs_dir / "customer_segments.csv"
    if not segments_path.exists():
        raise FileNotFoundError(
            "customer_segments.csv is missing. "
            "Run scripts/customer_segmentation.py first."
        )

    segments = pd.read_csv(segments_path)

    # Keep only the fields needed for routing.
    segments = segments[
        ["customer_id", "segment_name", "is_premium"]
    ].copy()

    segments["is_premium"] = (
        segments["is_premium"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    final = few_results.merge(
        segments,
        on="customer_id",
        how="left"
    )

    # Missing segment records are treated as non-premium.
    final["is_premium"] = final["is_premium"].fillna(False).astype(bool)
    final["segment_name"] = final["segment_name"].fillna("unknown")

    final["priority"] = "NORMAL"
    final["route"] = "support_queue"

    human_review_statuses = {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted",
    }

    for index, row in final.iterrows():
        if row["status"] in human_review_statuses:
            final.at[index, "route"] = "human_review"

        elif row["category"] == "fraud":
            final.at[index, "priority"] = "HIGH"
            final.at[index, "route"] = "security_team"

        elif row["is_premium"]:
            final.at[index, "priority"] = "HIGH"
            final.at[index, "route"] = "premium_desk"

    # Save only the required final submission columns.
    required_columns = [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used",
    ]

    final[required_columns].to_csv(
        outputs_dir / "triage_results.csv",
        index=False
    )

    print("\n========== FINAL TRIAGE ROUTING ==========")
    print("Saved:", outputs_dir / "triage_results.csv")
    print("\nRouting counts:")
    print(final["route"].value_counts().to_string())
    print("\nPriority counts:")
    print(final["priority"].value_counts().to_string())
    print("\nStatus counts:")
    print(final["status"].value_counts().to_string())

    return final[required_columns]
