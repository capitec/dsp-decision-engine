# Project 08: Collections Treatment Assignment (Haiku)

## Business Decision
This system assigns daily treatment actions to delinquent customer accounts. Given account state (arrears level, balance, payment history), collections risk, and contact responsiveness, it decides which collections strategy to apply: automated messages (SMS/email), IVM, voice calls, field agents, or legal referral—or no action. It also flags when settlement offers are eligible and ranks urgent accounts for scarce agent capacity.

## Inputs and Outputs

**Key Inputs** (~40 parameters):
- Account identity: account_id, client_id, decision_date
- Delinquency: days_past_due, arrears_amount, balance, product_family_code
- Collections metrics: cure history (12m, 24m), contact rate, promise-kept rate, broken promises
- Contact history (90d): SMS/call/email attempts and successful contacts
- Payment history: recent payments, last payment date/amount
- Account flags: debt review status, hardship, deceased, litigation, dispute, external counseling

**Key Outputs** (~30 fields):
- treatment_code (0–13: which action to take)
- treatment_intensity (1–5: escalation level)
- channel_code (1–7: SMS, email, in-app, IVM, voice, field, agency)
- Risk bands: arrears_bucket, balance_band, collections_band, contact_band
- collections_score (0–100) and unadjusted score
- episode_id, path_position (for multi-step escalation)
- active_suspensions, suspension_codes_applied (blocks contact if present)
- arrangement_assessment_required (settlement eligibility flag)
- pool allocation: pool_name, rank_in_pool, pool_size (for agent assignment)

## Main Steps

1. **Account state assembly**: Bucket days_past_due into 8 tiers; band balance into 7 tiers (thresholds: 2.5k, 10k, 25k, 50k, 100k, 250k).
2. **Suspension evaluation**: Check debt review (101–103), hardship (111), deceased (107), external counseling (108), dispute (110), litigation (113). Any blocking suspension overrides treatment.
3. **Risk assessment**: Compute collections_score (base 40 + days/100, adjustments for payments, contact, cure history, promises). Band into 6 tiers (edges: 20, 35, 50, 65, 80). Compute contact_band (responsive/intermittent/silent/unreachable).
4. **Treatment matrix lookup**: 5D sparse lookup on (arrears_bucket, collections_band, balance_band, contact_band, product_family). Returns treatment_code, intensity, retries, cooling-off days.
5. **Escalation path**: Compute episode_id (persistent or hashed), advance path_position if retry window passed.
6. **Arrangement eligibility**: Flag if treatment_code==12 (settlement offer) and no active arrangement.
7. **Capacity allocation**: Rank account in pool (early/late agents based on arrears tier) for call-center assignment.
8. **Output assembly**: Package decision, bands, scores, matrix attribution, suspensions, arrangement flag, pool rank, reason codes.

## Hard to Understand from Code Alone

- **Treatment codes (1–13)** map to channels (1=SMS, 2=email, 3=in-app, 4=IVM, 5–7=voice calls, 8=field, 9=agency, 10=notice, 11=legal, 12=settlement, 13=write-off). Business logic for which treatment is "better" for different customer profiles is not explicit; matrix generation is stub-deterministic.
- **Suspension codes** (101–113) are partially documented inline but real production tracks expiry dates per code, which is not modeled here.
- **Collections score** is simplified heuristic (hardcoded thresholds, ~10 weighted features). Real model uses 28 characteristics with null-handling, normalization, and regressed weights—all omitted.
- **Treatment matrix** generation is deterministic; production loads from a structured spreadsheet (5376 cells, 5 dimensions). Current stub applies simple escalation rules and is not representative.
- **Capacity allocation** is stub: 9 pool names mentioned but only "early/late_agents" populated; real system ranks hundreds of thousands daily across scoped pools with strict capacity enforcement.
- **Episode tracking and escalation** simplified; production maintains full retry/reset history and cooling-off windows per channel.
- **Overlays/adjustments** always empty; production applies regional, cohort, or A/B test overlays that modify treatment or intensity.
