"""The fraud reason codes: one registry for what the rules and the hard blocks can say."""
from __future__ import annotations

from credit_core.reason_codes import ReasonCode, ReasonCodeRegistry

from fraud_interdiction import vocab

FAMILY_BASE = {"CF": 1000, "AT": 2000, "MS": 3000, "FP": 4000, "AA": 5000}
WORDINGS_PER_FAMILY = 40

# Most severe first: AML-adjacent codes rank above mule and scam, and so on down the family order.
REGISTRY = ReasonCodeRegistry("fraud-reasons-2026.01", [
    *(ReasonCode(FAMILY_BASE[f] + i, 10 * vocab.FAMILIES.index(f) + i % 10 + 1,
                 f"{vocab.FAMILY_NAMES[f]} indicator {i}", f == "AA")
      for f in vocab.FAMILIES for i in range(WORDINGS_PER_FAMILY)),
    *(ReasonCode(code, 1, f"hard block: {name}", True) for name, _, code in vocab.GATES.values()),
])
