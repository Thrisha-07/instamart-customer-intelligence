
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "outputs" / "policy_search_results.csv"

results = pd.read_csv(RESULTS_PATH, keep_default_na=False)

for _, row in results.iterrows():
    print("\nQuestion:", row["question"])
    print("Expected document:", row["expected_doc"])
    print("Retrieved document:", row["retrieved_doc"])
    print("Retrieved text:", row["retrieved_text"])
    print("Expected fact:", row["expected_fact"])
    print("Document match:", row["document_match"])
    print("Fact match:", row["fact_in_retrieved_text"])
    print("-" * 60)
