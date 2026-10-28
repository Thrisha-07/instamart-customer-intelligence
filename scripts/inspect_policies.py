
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "data" / "policies"

for filename in [
    "refund_policy.txt",
    "delivery_policy.txt",
    "escalation_policy.txt",
    "payment_policy.txt",
    "privacy_policy.txt",
]:
    path = POLICY_DIR / filename

    print("\n" + "=" * 60)
    print(filename)
    print("=" * 60)

    print(path.read_text(encoding="utf-8"))
