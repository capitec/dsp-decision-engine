# Project 04: Campaign Trees (Campaign 23, Flex Loan Top-up)

## Business Decision

Campaign 23 decides whether to **offer an eligible customer a pre-approved flex loan top-up**, which tier (amount ceiling) to place them in, and whether they should be placed in a control or holdout group for experimental measurement. The output is one assignment record per customer per cycle containing the offer tier, validation evidence, and group assignment.

## Inputs and Outputs

**Inputs** (sample_request.json): Customer profile across seven dimensions—demographics (age, residency, capacity), regulatory/fraud flags (sanctions, deceased, debt review), consent/channel state (marketing opt-out, channel consent, contact fatigue), credit risk (risk grade, behaviour score, arrears history), flex loan tenure (months on book, settlement ratio, term remaining), affordability estimates (discretionary income, instalment-to-income ratio), and holdout design versions for group assignment.

**Outputs** (pipeline.emit()): Assignment ID, eligibility decision with decline reasons, suppression codes (absolute and measurement-relevant), pre-assessed amount and term from the pricing engine, tree outputs for both adjusted and unadjusted tree runs (leaf ID, outcome code, tier ceiling), advertised amount, overlay stack ID, advertised tier ceiling before cap reduction, control/holdout group membership.

## Main Steps (in order)

1. **Identifiers pass-through** – echo client_id, campaign_id, cycle_id, cycle_date, assignment_id, tree_version
2. **Suppressions** (§5.2) – evaluate ~9 suppression codes; mark absolute suppressions (exclude from tree) vs. measurement-relevant (still traverse tree)
3. **Eligibility and consent** – core eligibility check; channel permission and consent verdict
4. **Appetite** – lookup max amount and term from core appetite table
5. **Pre-assessment** (§5.5) – calculate max affordable instalment from discretionary income; use pricing engine to solve for term and rate; establish validity window
6. **Run tree twice** – execute decision tree with overlay parameters (adjusted) and without (unadjusted) to capture both the modified decision and the unmodified baseline
7. **Apply cap overlay** (§5.3.4) – reduce the tree's tier ceiling using adjustment register if any cap-reduction overlay is in force for this cycle
8. **Advertised amount** – min(pre_assessed_amount, advertised_tier_ceiling) per spec requirement
9. **Holdout assignment** (§5.7) – deterministic stable hash of client_id, campaign_id, holdout design version into control/universal-holdout groups
10. **Emit evidence record** – output all fields for the assignment record

## Hard to Understand (code alone)

- **Framework warmup friction** (inference.py): The cycle_date is a datetime.date input, but the decider framework's warmup synthesizer only knows bool/int/str/bytes, failing on date. The workaround (load from sample_request.json) is not obvious from the code structure alone; understanding it requires reading docs 00 and 03's NOTES.md.
- **Dual tree traversal** (tree_model.py): The project needs both a vectorized v3 document for bulk execution and a pure-Python walk for capturing the path. The arithmetic must match between them (validated by test_walker_agrees_with_treeconfig_leaf), but this divergence risk and its resolution are not visible without reading the test and the full module docstring.
- **Overlay mechanism split** (overlays.py, pipeline.py): Threshold shifts (volume dial, cut-off shift) go through TreeConfig's params={...} mechanism and run the tree twice; cap reductions go through AdjustmentRegister, a different code path entirely. Why two paths exist (TreeOverlay's custom tighten-only check vs. AdjustmentRegister's closed registry) is a framework gap explained only in the module docstring, not the code.
- **DAG-resolved stage order** (pipeline.py docstring): Stage 5 (pre-assessment) must run before Stage 3 (tree) because the tree reads pre_assessed_amount at nodes 7-8. The dag() call resolves this from data dependencies, not from spec stage numbers. The inversion is only visible in the docstring.
