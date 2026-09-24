# 01-transaction-fraud (Sonnet implementation)

## What It Decides

Real-time fraud risk assessment for instant payment transactions. For each transaction, the system decides: allow, monitor, escalate to step-up, hold for manual review, decline, block the channel, or freeze the account.

## Inputs and Outputs

**Inputs:**
- Transaction event with enriched features: transaction amount, timestamp, client/device/channel info
- Risk signals: model score, velocity metrics (1-min, 10-min, 1-hour counts and aggregates), device reputation, counterparty age
- Compliance flags: sanctions list match, account frozen, mule watchlist, device blocked, card compromised, court order
- Enrichment freshness: boolean flags indicating stale/absent enrichment data sources

**Outputs:**
- `action_code`: one of 7 actions (allow → freeze account by severity)
- `decline_reason_codes`: list of internal reason codes from fired rules
- `client_wording_key`: client-facing (intentionally generic) message key
- Governance fields: source rule, governance exceptions, counterfactual action (what would have happened without overlays)

## Main Steps

1. **Degradation assessment:** Compute velocity completeness band (fresh/partially degraded/materially degraded) and overall degraded-mode verdict (normal/reduced/restricted/fail-closed), which gate rule applicability.

2. **Hard blocks:** Check membership in sanctions list, account freeze, confirmed mule watchlist, device block, card compromise, court order. These set an action floor that no rule softens.

3. **Feature engineering:** Event timestamp lateness flag; enrichment degradation bitset (8 sources tracked).

4. **Rule evaluation:** ~635 rules split into live (production) and shadow (monitoring only) sets. Each rule fires based on a predicate (conditions on features, velocity bands, model score, etc.), gated by applicability (event type, effective date, client segment, degraded mode).

5. **Suppression logic:** Rules referencing stale/absent velocity data may be unevaluable (suppressed) or forced to false depending on the rule's declared behavior; shadow rules are structurally isolated from action output (by dag construction, not convention).

6. **Overlay adjustments:** Three adjustment types applied only to eligible rules: sensitivity dial (inverted as amount-threshold multiplier), threshold multiplier (0.7x for festive period), action escalation (monitor → step-up on web channel for account takeover).

7. **Action resolution:** From fired rules, select the highest-precedence action by severity, priority, critical status, and family precedence. Apply hard-block floor. Compute counterfactual action (same logic without overlays). Flag governance exceptions when two critical rules demand conflicting actions.

8. **Wording:** Map action to client-safe text (generic per action, not per rule, to avoid leaking thresholds to fraudsters). Record registry version, reason ranking.

## Hard-to-Understand Aspects

**Overlay complexity:** The base vs. raw fired distinction requires tracking two parallel evaluations (with and without overlay thresholds). Overlays target specific rule families and have composition order—the second adjustment depends on the first's result—which is semantically important but invisible in the final rule output.

**Degraded-mode gating:** Multiple degradation sources are tracked as bitsets, but only specific combinations trigger mode transitions (e.g., 2+ degraded sources → restricted mode, which suspends suppressible rules). Rules declare velocity stale/absent behavior (suppress, evaluate-false, or last-known-good) independently of mode, creating four interaction cases per rule.

**Critical rule tie-breaking:** When multiple critical rules fire with different actions, the code skips non-critical rules in that family and flags an exception. The logic for handling "critical rules of different families with conflicting actions of same severity" is defined by max() on action precedence, not explicitly enumerated.

**Governance coupling:** The rule catalog (governance metadata: critical, priority, severity, family) and the rules.json document must be regenerated together and stay in sync; they're built from the same seed but are separate files, creating an implicit coupling recorded in NOTES.md.
