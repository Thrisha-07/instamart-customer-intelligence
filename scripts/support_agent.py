
from pathlib import Path
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mock_llm
from scripts.policy_search import PolicySearch

DB_PATH = ROOT / "data" / "instamart.db"
OUTPUTS = ROOT / "outputs"
STATE_DIR = ROOT / "state"
SCENARIOS_PATH = ROOT / "data" / "raw" / "agent_scenarios.csv"

OUTPUTS.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

LEDGER_PATH = OUTPUTS / "refund_ledger.jsonl"
RUNS_PATH = OUTPUTS / "agent_runs.jsonl"

policy_search = PolicySearch(chunk_size=40, overlap=10)

SYSTEM_PROMPT = """
AGENT
You are an Instamart refund support agent.
Use the supplied tools to check orders and policy before deciding.
Never claim a refund was issued unless issue_refund succeeded.
Return a final answer with a status prefix such as REFUND:, ESCALATE:,
DENIED:, NOT_FOUND:, or FALLBACK_MAX_ITERATIONS:.
For requests that cannot be safely completed, hand off to a human.
"""

TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order in the SQLite database.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Search the policy documents for relevant guidance.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate_refund",
        "description": "Calculate late-delivery refund eligibility.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"},
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Record an eligible refund in the refund ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund request for human approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
]


def get_order(order_id):
    """Return order details or an error from SQLite."""
    try:
        with sqlite3.connect(DB_PATH) as connection:
            row = connection.execute(
                """
                SELECT order_id, status, order_amount, delivery_minutes
                FROM orders
                WHERE order_id = ?
                """,
                (order_id,),
            ).fetchone()

        if row is None:
            return {"error": "Order not found", "order_id": order_id}

        return {
            "order_id": row[0],
            "status": str(row[1]).lower(),
            "order_amount": float(row[2]),
            "delivery_minutes": (
                None if row[3] is None else float(row[3])
            ),
        }

    except (sqlite3.Error, TypeError, ValueError) as exc:
        return {"error": f"Order lookup failed: {exc}"}


def search_policy(query):
    """Return the top three policy matches."""
    matches = policy_search.retrieve(query, k=3)

    return {
        "results": [
            {
                "source": item["source"],
                "score": item["score"],
                "text": item["text"],
            }
            for item in matches
        ]
    }


def calculate_refund(order_amount, delivery_minutes):
    """Full refund only when delivery exceeds 45 minutes."""
    if delivery_minutes is None:
        return {
            "refund_inr": 0.0,
            "reason": "Delivery time is missing; human review required.",
        }

    if float(delivery_minutes) > 45:
        return {
            "refund_inr": round(float(order_amount), 2),
            "reason": "Delivery exceeded 45 minutes.",
        }

    return {
        "refund_inr": 0.0,
        "reason": "Delivery took 45 minutes or less.",
    }


def append_ledger(tool_name, order_id, refund_inr):
    """Append a refund or escalation record to the JSONL ledger."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }

    with LEDGER_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")

    return {
        "refund_inr": entry["refund_inr"],
        "recorded": True,
    }


def issue_refund(order_id, refund_inr):
    return append_ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return append_ledger("escalate_to_human", order_id, refund_inr)


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


def checkpoint_path(run_id):
    """Return a safe checkpoint filename for this run."""
    safe_id = "".join(
        char for char in str(run_id)
        if char.isalnum() or char in ("-", "_")
    )
    return STATE_DIR / f"{safe_id}.json"


def save_checkpoint(run_id, state):
    """Persist the current agent state."""
    checkpoint_path(run_id).write_text(
        json.dumps(state, indent=2),
        encoding="utf-8",
    )


def append_run_log(run_id, state):
    """Append the run summary and steps to the JSONL log."""
    record = {
        "run_id": run_id,
        "scenario_id": state["scenario_id"],
        "status": state.get("status", ""),
        "final_answer": state.get("final_answer", ""),
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
        "steps": state["steps"],
    }

    with RUNS_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def status_from_answer(answer):
    """Extract the status prefix from a final answer."""
    return str(answer).split(":", 1)[0].strip()


def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tool_execution=None,
):
    """Run the ReAct agent and checkpoint each tool observation."""
    run_id = run_id or f"{scenario_id}_{int(time.time() * 1000)}"

    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "tokens_used": 0,
        "llm_calls": 0,
        "steps": [],
        "executed_tools": [],
        "started_at": time.monotonic(),
        "status": "",
        "final_answer": "",
        "max_iterations": max_iterations,
        "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds,
    }

    save_checkpoint(run_id, state)

    return _continue_agent(
        state,
        crash_after_tool_execution=crash_after_tool_execution,
    )


def _continue_agent(state, crash_after_tool_execution=None):
    """Continue a run from its current state."""
    run_id = state["run_id"]
    start = time.monotonic()
    iterations = 0

    while iterations < state["max_iterations"]:
        # Check the time limit before each model call.
        if time_limit_expired(state, start):
            return finish_fallback(
                state,
                "FALLBACK_TIMEOUT: Human support handoff required.",
            )

        # Do not make another call when the budget is already exhausted.
        if state["tokens_used"] >= state["token_limit"]:
            return finish_fallback(
                state,
                "FALLBACK_TOKEN_LIMIT: Human support handoff required.",
            )

        response = mock_llm.chat(state["messages"])
        usage = response.usage.total_tokens

        state["llm_calls"] += 1
        state["tokens_used"] += usage
        iterations += 1

        # If usage exceeds the budget, do not execute returned tools.
        if state["tokens_used"] > state["token_limit"]:
            return finish_fallback(
                state,
                "FALLBACK_TOKEN_LIMIT: Human support handoff required.",
            )

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        # The model has returned its final answer.
        if not tool_calls:
            state["final_answer"] = message.content or (
                "FALLBACK: No final answer returned."
            )
            state["status"] = status_from_answer(state["final_answer"])

            save_checkpoint(run_id, state)
            append_run_log(run_id, state)
            return public_result(state)

        # Preserve the assistant's tool requests in message history.
        state["messages"].append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            tool_name = tc.name
            arguments = {}

            try:
                if isinstance(tc.arguments, dict):
                    arguments = tc.arguments
                elif isinstance(tc.arguments, str):
                    arguments = json.loads(tc.arguments or "{}")
                else:
                    arguments = {}

                if tool_name not in TOOL_MAP:
                    observation = {
                        "error": f"Unknown tool: {tool_name}"
                    }
                else:
                    observation = TOOL_MAP[tool_name](**arguments)

            except Exception as exc:
                observation = {"error": str(exc)}

            # Use tool_name so the test script can read each step.
            step = {
                "tool_name": tool_name,
                "arguments": arguments,
                "observation": observation,
            }

            state["steps"].append(step)
            state["executed_tools"].append(tool_name)

            # Add the tool observation to the conversation.
            state["messages"].append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tool_name,
                "content": json.dumps(
                    observation,
                    ensure_ascii=False,
                ),
            })

            # Save the checkpoint immediately after every tool result.
            save_checkpoint(run_id, state)

            if (
                crash_after_tool_execution is not None
                and len(state["executed_tools"])
                == crash_after_tool_execution
            ):
                raise RuntimeError(
                    "Simulated crash after checkpoint save."
                )

    return finish_fallback(
        state,
        "FALLBACK_MAX_ITERATIONS: Human support handoff required.",
    )


def time_limit_expired(state, start):
    """Check whether the time limit has elapsed."""
    return (
        state["time_limit_seconds"] <= 0
        or time.monotonic() - start >= state["time_limit_seconds"]
    )


def finish_fallback(state, answer):
    """Save and return a safe fallback answer."""
    state["status"] = status_from_answer(answer)
    state["final_answer"] = answer

    save_checkpoint(state["run_id"], state)
    append_run_log(state["run_id"], state)

    return public_result(state)


def public_result(state):
    """Return the public fields for an agent run."""
    return {
        "run_id": state["run_id"],
        "scenario_id": state["scenario_id"],
        "status": state["status"],
        "final_answer": state["final_answer"],
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
    }


def resume_agent(run_id):
    """Resume from a checkpoint without re-running completed tools."""
    path = checkpoint_path(run_id)

    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint found for {run_id}"
        )

    state = json.loads(path.read_text(encoding="utf-8"))
    state["started_at"] = time.monotonic()

    # Completed tool observations are already stored in the messages.
    return _continue_agent(state)


def main():
    import pandas as pd

    scenarios = pd.read_csv(
        SCENARIOS_PATH,
        keep_default_na=False,
    )

    mock_llm.reset()

    # Remove old demonstration logs for a reproducible batch run.
    for path in (LEDGER_PATH, RUNS_PATH):
        if path.exists():
            path.unlink()

    results = []

    for _, scenario in scenarios.iterrows():
        result = run_agent(
            scenario["message"],
            scenario["scenario_id"],
        )

        results.append({
            "scenario_id": scenario["scenario_id"],
            "expected_outcome": scenario["expected_outcome"],
            "predicted_outcome": result["status"],
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
        })

    results_df = pd.DataFrame(results)
    output_path = OUTPUTS / "support_agent_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nAgent scenario results:")
    print(results_df.to_string(index=False))
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
