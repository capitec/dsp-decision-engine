# Transaction Fraud Interdiction (Project 01) — Haiku Summary

## Business Decision

This system evaluates payment transactions in real-time to prevent fraud. For each instant payment (event type 210), it decides one of seven actions: **allow**, **monitor**, **step-up challenge**, **hold for review**, **decline**, **block the channel**, or **freeze the account**. The decision balances risk (blocking fraud) against customer friction.

## Inputs and Outputs

**Inputs:**
- Raw transaction event: amount, beneficiary, device ID, session, channel, timestamps
- Enrichment (stubbed): beneficiary age, device reputation, device/SIM change windows, velocity metrics (1m, 10m, 1h, 24h counts/sums), ML model score
- Client context: ID, segments (e.g., retail/standard), account status
- Metadata: decision date, rule set version, degradation mode flags

**Outputs:**
- `action_code` (7 severity levels via enum)
- `action_source_rule_id` (which rule fired)
- `fired_rule_ids` and `shadow_fired_rule_ids`
- `reason_codes` (ranked list)
- `DecisionRecord`: complete audit with all inputs, rule evaluations, and decision logic for replay compliance

## Main Steps (In Order)

1. **Normalise event** — Validate mandatory fields, parse timestamps, stub enrichment fallbacks to sane defaults
2. **Resolve rule set** — Load ~520 live + 100+ shadow rules for the decision date (5 families: card fraud, account takeover, mule scam, first-party fraud, AML-adjacent)
3. **Get client context** — Load profile data (stubbed: no live reads, hard-coded active/unfrozen)
4. **Resolve hard blocks** — Check for account frozen or beneficiary on mule list (overrides rules but doesn't skip evaluation)
5. **Evaluate rules** — Run all applicable rules on event features; records if each fired on base thresholds vs. overlay-adjusted thresholds
6. **Resolve action** — Sort fired rules by severity and priority; critical rules checked for conflicts; apply hard-block floor
7. **Emit decision record** — Assemble complete audit trail with decision ID, fired rules, reason codes, counterfactual (what base rules alone would decide)

## Hard to Understand from Code Alone

- **Shadow rules:** They fire but never change the decision; exist to monitor new rules before going live. The data structure tracks them separately but the business logic is opaque.
- **Overlay system:** Code references overlays (sensitivity dials, threshold multipliers) but none are actually applied (overlay_stack is always None). Conceptually these let ops adjust thresholds without code redeploy, but the mechanism is stubbed.
- **Degradation modes:** `enrichment_degradation_code` and `degraded_mode_code` propagate through the pipeline; suggest handling missing enrichment (e.g., velocity store latency) by suspending certain rules and entering fail-closed mode, but the actual behavior is minimal.
- **Enrichment stubbing:** Hard-coded fallback values (amount → 1000, beneficiary_age_hours → 24, etc.) replace real store lookups. The rule logic assumes these fields always exist, hiding the real data-availability complexity.
- **Counterfactual action:** Emitted for every decision but only meaningful when overlays have moved thresholds; otherwise matches the actual action. Use case unclear without downstream audit tooling context.

