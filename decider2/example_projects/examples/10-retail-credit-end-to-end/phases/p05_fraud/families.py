"""P05 fraud families — 188 live rules across five families, plus 31 shadow
rules. Spec 5.6. `evaluate_all=True`: no early exit, every applicable rule
fires or does not, and every firing is recorded — a genuinely fraudulent
application typically fires seven rules across three families, and all seven
are wanted in the record, not just the one that decided the verdict.

Each family is a `ruleset` (doc 08 §3): a data-shaped module whose interior —
the 188 rule bodies — is a validated document Financial Crime edits under
four-eyes approval, not a Python change. `families.py` declares the five
rulesets' interfaces; it does not contain a single rule.
"""

from __future__ import annotations

from decider2 import ruleset

F1 = ruleset(
    name="f1_identity_manipulation",
    reads=["identity_number", "date_of_birth", "biographical_fields",
           "device_fingerprint"],
    writes=["f1_firing_bits", "f1_max_severity"],
    # 46 live, 9 shadow. Synthetic identities, identity-number reuse,
    # manipulated dates of birth, mismatched biographical data.
)

F2 = ruleset(
    name="f2_application_content_inconsistency",
    reads=["declared_income", "employer_sector_code", "contact_history",
           "address_history"],
    writes=["f2_firing_bits", "f2_max_severity"],
    # 38 live, 6 shadow. Income inconsistent with employer, employer
    # inconsistent with sector, address velocity, contact reuse.
)

F3 = ruleset(
    name="f3_device_and_channel",
    reads=["device_reputation_score", "automation_signals", "session_anomalies",
           "geolocation"],
    writes=["f3_firing_bits", "f3_max_severity"],
    # 34 live, 8 shadow. Device reputation, emulator/automation signals,
    # session anomalies, geolocation conflicts.
)

F4 = ruleset(
    name="f4_syndicate_and_network",
    reads=["device_fingerprint", "beneficiary_id", "employer_sector_code",
           "referral_chain"],
    writes=["f4_firing_bits", "f4_max_severity"],
    # 41 live, 5 shadow. Shared devices, shared beneficiaries, application
    # bursts against one employer, referral-chain patterns.
)

F5 = ruleset(
    name="f5_known_fraud_and_watchlist",
    reads=["internal_fraud_register_hit", "consortium_hit", "mule_watchlist_hit"],
    writes=["f5_firing_bits", "f5_max_severity"],
    # 29 live, 3 shadow. Internal fraud register, consortium hits, first-party
    # fraud history, mule watchlists. F5 severity-4 is the sole decline trigger
    # in precedence.py.
)

# Shadow rules (31 total across the five families) are declared inside the
# same ruleset documents with `enabled: false` equivalent semantics that still
# EVALUATE and RECORD but never feed precedence.py's verdict. They are
# measured and governed; per OWNERS.toml and spec 5.6 they are not decision
# points, because they provably cannot affect the answer.
