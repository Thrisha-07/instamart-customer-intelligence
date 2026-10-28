
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

tickets = pd.read_csv(
    RAW / "tickets.csv",
    keep_default_na=False
)

# Remove leading/trailing whitespace
for column in tickets.columns:
    tickets[column] = tickets[column].astype(str).str.strip()

# Normalize whitespace in ticket text
tickets["text"] = (
    tickets["text"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Remove exact duplicate rows
tickets = tickets.drop_duplicates().copy()

# Remove repeated identical sentences within a ticket
def remove_repeated_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    unique_sentences = []
    seen = set()

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence.casefold() not in seen:
            seen.add(sentence.casefold())
            unique_sentences.append(sentence)

    return " ".join(unique_sentences)

tickets["text"] = tickets["text"].apply(remove_repeated_sentences)

# Preserve "none" as a separate label for now.
# We will decide how to handle these rows during model preparation.

tickets.to_csv(CLEAN / "tickets.csv", index=False)

print("Clean tickets shape:", tickets.shape)
print("\nCategory counts:")
print(tickets["true_category"].value_counts().to_string())
print("\nBlank ticket texts:", tickets["text"].eq("").sum())
print("Blank order IDs:", tickets["order_id"].eq("").sum())
print("Duplicate ticket IDs:", tickets["ticket_id"].duplicated().sum())
print("\nSaved to:", CLEAN / "tickets.csv")
