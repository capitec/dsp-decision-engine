# Campaign Trees Targeting Decision (Project 04)

## What It Decides

This system decides which retail financial campaigns to target each customer with, on what terms (amount, term, tier), and through which channels. It also determines holdout assignments (control groups) and applies contact caps to prevent over-engagement (max 2 campaigns per customer per cycle).

## Inputs and Outputs

**Inputs:**
- Customer features: age, tenure, income band, behavior score, flex balance, payment history, discretionary income, employment type
- Cycle metadata: cycle_id, cycle_date_str, client_id
- Campaign registry: 60 campaigns, each with tree_version and targeting rules
- Pre-assessed amounts from upstream (project 03): per-campaign credit limits, risk grades, binding constraints

**Outputs:**
- Assignment records: client_id, campaign_id, offered_amount, term_months, channel, priority_weight, reason_label, contacted status
- Path records: decision tree path taken per campaign (node keys visited)
- Suppression records: legal blocks (deceased, opt-out, do-not-target, cooling-off)

## Main Steps

1. **Load campaigns and trees** → Generate 60 deterministic campaign trees; each tree has 20–100 nodes, 4–8 levels deep
2. **Apply suppressions** → Check for absolute blocks (deceased ~0.01%, debt review ~0.2%, do-not-target ~0.022%) and measurement-relevant suppressions (opt-out ~7.7%); skip suppressed campaigns
3. **Evaluate trees** → For each non-suppressed campaign, traverse tree by evaluating node conditions (e.g., "age > threshold") until reaching a leaf; capture path and leaf outcome (target/control/do-not-target tier)
4. **Apply overlays** → Adjust amounts and tiers per active policy overlays (e.g., "cap tier 1 to 90%"); record unadjusted values when changed
5. **Consume pre-assessments** → Merge pre-assessed amounts, term, risk grade, and binding constraint (from project 03 batch) into tree evaluations
6. **Arbitration** → Rank campaigns by priority_weight; deterministically assign control group (5%) and variant (90% champion, 10% challenger); apply 2-contact cap per cycle; record non-contact reasons (control, contact_cap_reached, ranked_out)
7. **Build output** → Package assignments, paths, suppressions into final record

## Unclear Parts

- **Overlay semantics**: Code shows only cap_reduction overlays applied; real spec mentions volume_dial and cut_off_shift that would change tree leaf outcomes, but apply logic is stubbed. Unclear how these interact with tree evaluation order.
- **Measurement-relevant suppressions**: Marked in suppression records but unclear how they differ in downstream handling vs. absolute suppressions.
- **Amount rule vs. pre-assessed amount**: Tree leaves define amount_rule expressions (e.g., "min(pre_assessed, 250000)"), but tree evaluation returns outcome_code and tier_code only; amount_rule is never evaluated in code—pre_assessed_amount from upstream is used directly.
- **Tree condition language**: Condition parsing supports `>`, `<`, `>=`, `<=`, `==`, `in [...]`, but no operator precedence or AND/OR—unclear if compound conditions exist in real trees.
- **Determinism guarantee**: Assignment_id and control determination use MD5 hash of client_id, cycle_id; tree traversal is deterministic given features, but order of tree evaluation across 60 campaigns could drift if tree load order changes.
