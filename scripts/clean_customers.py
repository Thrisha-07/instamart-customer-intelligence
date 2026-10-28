
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

customers = pd.read_csv(RAW / "customers.csv")

# 1. Standardize city names.
customers["city"] = customers["city"].str.strip().str.title()

# 2. Standardize gender values.
def normalize_gender(value):
    if pd.isna(value) or not str(value).strip():
        return "unknown"

    value = str(value).strip().lower()

    if value in {"m", "male", "man"}:
        return "male"
    if value in {"f", "female", "woman"}:
        return "female"

    return "unknown"

customers["gender"] = customers["gender"].apply(normalize_gender)

# 3. Convert ages to numeric and invalidate ages outside 10–100.
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[
    ~customers["age"].between(10, 100), "age"
] = float("nan")

# 4. Fill missing ages with the median of each city.
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)

# 5. Convert signup_date to datetime.
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

# Save the cleaned dataset separately.
customers.to_csv(CLEAN / "customers.csv", index=False)

print("Cleaned customers:", customers.shape)
print("\nCity counts:")
print(customers["city"].value_counts().to_string())
print("\nGender counts:")
print(customers["gender"].value_counts().to_string())
print("\nMissing ages:", customers["age"].isna().sum())
print("Minimum age:", customers["age"].min())
print("Maximum age:", customers["age"].max())
print("Missing signup dates:", customers["signup_date"].isna().sum())
print("\nSaved to:", CLEAN / "customers.csv")
