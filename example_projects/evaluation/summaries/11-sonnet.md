# Project 11: Business Credit End-to-End (EP-1)

## Business Decision
Approves new business credit facilities (EP-1 origination) for products 50 (amortizing term loans) and 51 (revolving credit). Produces credit grades, pricing, debt service metrics, and a DSCR (Debt Service Coverage Ratio) covenant binding that persists across the facility's life.

## Inputs
- Application metadata: facility ID, decision/knowledge dates, product code (50 or 51)
- Business financials: requested amount/term, EBITDA, annual turnover, debt levels, security type
- Existing accounts: list of current debts with balances, instalments, and dates
- Related entities: up to four owner/guarantor records with ownership %, bureau scores, adverse events

## Outputs
- Assessment kind (new-to-bank code 1) and comparison basis (origination-only code)
- Facility instalment: direct from pricing model (product 50) or project 07's notional payment (product 51)
- DSCR covenant instance: version ID (frozen at origination), threshold, measured ratio, headroom, breach class (none / technical / material / severe)
- Re-pricing vocabulary for L1 annual reviews

## Main Steps (in order)
1. **Compose project 05's pipeline** (unmodified): entity scoring, financial ratios, combined grade, pricing
2. **Calculate new facility instalment**: if revolving (51), import notional payment logic from project 07; otherwise use pricing output
3. **Bind DSCR covenant**: resolve template version in force at decision date, lock it to the instance forever
4. **Compute debt service**: sum existing account obligations (using project 00's per-account rules) plus new instalment
5. **Measure DSCR**: EBITDA ÷ 12 ÷ total debt service; classify breach severity (≥threshold vs 90% vs 75% of threshold)
6. **Emit** covenant and facility outputs for underwriting and future reviews

## Hard to Understand (Code-Only)
- **Covenant version freezing**: Binding locks the template version to the instance at origination; later requests reuse that pinned version, never re-resolving even if new definitions exist (e.g., 2028-09 IFRS 16 lease treatment only applies to instances bound after 2028-09-01).
- **Framework pass-throughs**: Inputs like `facility_id` and `existing_accounts` flow through DAG untouched but aren't explicitly emitted; naming them raises WiringError despite surviving in output (documented as known friction point).
- **Product 51 pricing strategy**: Product 51 has no pricing in project 05 (single-product 50 scope); rather than fork 05's pricing, this project **composes** project 07's notional-payment calculation—pattern choice not obvious from code alone.
- **Three-date covenant testing**: `test_date` (when ratio is measured), `delivery_date` (when financials arrived), `determination_date` (when test ran) must never conflate; grace period for cure runs from min(delivery, determination), not from a single clock.
