
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

path = RAW / "agent_scenarios.csv"
scenarios = pd.read_csv(path, keep_default_na=False)

print("Total scenarios:", len(scenarios))
print("\nExpected outcomes:")
print(scenarios["expected_outcome"].value_counts().to_string())

print("\nAll scenarios:")
print(scenarios.to_string(index=False))
