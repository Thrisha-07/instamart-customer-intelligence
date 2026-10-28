
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm
from scripts.support_agent import (
    run_agent,
    resume_agent,
    checkpoint_path,
)

run_id = "S1_resume_demo"
message = (
    "My order ORD-100048 reached me very late. "
    "I would like a refund."
)

# Remove an earlier checkpoint for this demonstration.
path = checkpoint_path(run_id)
if path.exists():
    path.unlink()

mock_llm.reset()

print("Running S1 and simulating a crash after tool 2...")

try:
    run_agent(
        message,
        scenario_id="S1",
        run_id=run_id,
        crash_after_tool_execution=2,
    )
except RuntimeError as exc:
    print("Expected simulated crash:", exc)

print("Checkpoint exists:", path.exists())

print("\nResuming from the saved checkpoint...")
resumed = resume_agent(run_id)

print("Final status:", resumed["status"])
print("Final answer:", resumed["final_answer"])
print("Model calls in resumed segment:", resumed["llm_calls"])
print("Total tool executions in checkpoint:", len(resumed["steps"]))
print("Executed tools:", [
    step["tool_name"] for step in resumed["steps"]
])
