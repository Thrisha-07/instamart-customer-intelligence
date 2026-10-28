
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

tickets = pd.read_csv(
    RAW / "tickets.csv",
    keep_default_na=False
)

print("Tickets shape:", tickets.shape)
print("\nColumns:")
print(tickets.columns.tolist())

print("\nSample tickets:")
print(tickets.head(5).to_string(index=False))

print("\nTicket column summary:")
print(tickets.info())

for column in tickets.columns:
    print(f"\n{column} — unique values:")
    print(tickets[column].value_counts().head(10).to_string())
