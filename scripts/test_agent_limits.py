
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm
from scripts.support_agent import run_agent

mock_llm.reset()
print("=== TOKEN LIMIT TEST ===")
token_result = run_agent(
    "My order ORD-100048 reached me very late. I would like a refund.",
    scenario_id="S1_TOKEN_LIMIT",
    run_id="S1_token_limit_demo",
    token_limit=200,
)
print("Status:", token_result["status"])
print("Answer:", token_result["final_answer"])
print("Tokens:", token_result["tokens_used"])

mock_llm.reset()
print("\n=== TIME LIMIT TEST ===")
time_result = run_agent(
    "My order ORD-100048 reached me very late. I would like a refund.",
    scenario_id="S1_TIMEOUT",
    run_id="S1_timeout_demo",
    time_limit_seconds=0,
)
print("Status:", time_result["status"])
print("Answer:", time_result["final_answer"])

mock_llm.reset()
print("\n=== MAX ITERATIONS TEST ===")
loop_result = run_agent(
    "[LOOP] Refund order ORD-100048.",
    scenario_id="S7_LOOP",
    run_id="S7_max_iterations_demo",
    max_iterations=6,
    token_limit=6000,
)
print("Status:", loop_result["status"])
print("Answer:", loop_result["final_answer"])
print("Model calls:", loop_result["llm_calls"])
print("Tokens:", loop_result["tokens_used"])
