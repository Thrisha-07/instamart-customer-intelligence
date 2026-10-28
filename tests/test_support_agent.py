
from pathlib import Path
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "support_agent.py"
RESULTS = ROOT / "outputs" / "support_agent_results.csv"

# Run the support-agent script
subprocess.run(
    [sys.executable, str(SCRIPT)],
    cwd=ROOT,
    check=True
)

# Verify all supplied scenarios
df = pd.read_csv(RESULTS)

assert len(df) == 7, "Expected 7 scenarios"
assert df["outcome_match"].all(), "Some outcomes do not match"
assert df["refund_match"].all(), "Some refund amounts do not match"

print("\nAll support-agent scenario tests passed!")
print("Scenarios tested:", len(df))
print("Outcome checks: PASS")
print("Refund checks: PASS")
