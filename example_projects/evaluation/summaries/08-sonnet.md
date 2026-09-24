# Collections Treatment Assignment (Project 08)

## What This Decides

This system decides what collections action to take on a customer account in arrears. It assigns a treatment strategy (SMS, email, voice, agent call, field visit, legal action, etc.) at a specific intensity level, with configured retry limits and cooling-off periods.

## Inputs and Outputs

**Inputs:** Account status (days past due, arrears amount, balance), historical behavior (payment patterns, contact success, cure history, prior arrangements), compliance status (debt review, litigation, disputes, hardship arrangements), customer financials (income, expenses, dependants, other debts), and treatment history (prior escalation position, contact counts over rolling periods).

**Outputs:** Treatment code and intensity, retry allowances, cooling-off days, collections risk score with probability bands, escalation path position (as-known and as-at-now variants), arrangement affordability verdict, and capacity allocation decision (which operational pool, ranking rank, whether allocated).

## Main Steps

1. **Account State** – Compute arrears bucket (1–8 based on days past due, overridden by arrangement hold or cure floor) and balance band.
2. **Suspensions** – Check compliance holds (debt review, litigation, insolvency, disputes) that block or restrict treatment.
3. **Risk Scoring** – Build collections score from account characteristics, adjust via overlays (shift, scaling, odds multiplier, band boundary), output probability of default and band.
4. **Treatment Matrix** – Lookup cell in 5,376-cell matrix (8 buckets × 6 risk bands × 7 balance bands × 4 contact-responsiveness bands × 4 product families) to find base treatment, retries, cooling-off days.
5. **Escalation Path** – Track sequence position over time; detect reset events (qualifying payments, promises captured, cures, disputes); determine whether to escalate, hold, or reset; apply intensity floor.
6. **Arrangement Assessment** – Test proposed payment affordability using project 02's assessment in distressed mode; confirm sustainability.
7. **Capacity Allocation** – Rank accounts within treatment pools (SMS, email, voice, agents, field, legal, etc.) by scoring basis; allocate if daily supply allows.

## Hard-to-Understand Parts

- **Temporal replay (as-known vs as-at-now):** Path resolution runs twice—once with event flags as recorded (drives today's actual treatment), again with `_as_at_now` variants (shows what would happen with late data included). Same logic, two input sets, labeling confusion.
- **Module import collision:** This project and project 02 both define `pipeline.py`. Fixed via `importlib.util.spec_from_file_location` to load 02's pipeline under a private module name, avoiding `sys.modules["pipeline"]` cache poisoning.
- **Reset event rules and re-entry after cure:** Subtle state machine—five reset event types each trigger different path effects (entry, hold, episode closed, counters cleared or not); re-entry thresholds depend on cure count in 12 months (0 → entry 1, 1 → entry 3, 2+ → entry 3 with bucket floor, 3+ → chronic path). Not obvious from code alone.
- **Treatment matrix severity heuristic:** Cell lookup uses calculated severity (bucket + band + contact-responsiveness, adjusted for small balances and ultra-delinquent accounts) to select treatment code. Logic is compact but requires mental model of intended escalation shape.
