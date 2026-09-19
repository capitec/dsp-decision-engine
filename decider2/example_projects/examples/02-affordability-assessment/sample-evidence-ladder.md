# Affordability assessment — evidence record

> Generated from execution trace `AFF-2026-08-14-0009182`. Not hand-maintained.
> Produced by `decider ladder render --assessment AFF-2026-08-14-0009182`.
>
> **Sensitivity: applicant financial data.** Access is logged.

Application 90114772 · Decision date **14 August 2026** · Product **Flex Loan (10)**
· Channel **Broker (6)** · Mode **New application** · Assessed **14 August 2026**

---

## 1. Who was assessed

Assessed as a **joint household** of two applicants with **2 dependants**. The two
applicants declared different dependant counts (2 and 1); the higher was used and
the discrepancy is recorded.

## 2. Income and how it was evidenced

Gross monthly income of **R38 412.00** was established from **3 income sources**.
The weakest tier among sources contributing more than 10.00% of the household
total was **tier 2**; the largest source was at **tier 2**. The effective blended
haircut was **4.31%**, reducing **R40 142.61** to **R38 412.00**.

| # | Applicant | Source | Tier | Evidence | Before | Haircut | After |
|---|---|---|---|---|---|---|---|
| 1 | A | Salary, permanent | 2 — payslips | 3 consecutive payslips to 2026-07-25 (`DOC-4471182`) | R26 400.00 | 0.00% | R26 400.00 |
| 2 | A | Commission | 2 — payslips | same payslip set | R4 918.44 | 3.00% | R4 771.09 |
| 3 | B | Salary, contract | 3 — internal deposits | 6 salary credits, originator `ACME PAYROLL`, account ···4471 | R8 824.17 | 13.00% | R7 240.91 |

Two further tiers were available and did not bind, and are recorded because they
were considered:

| Tier | Figure | Why it did not bind |
|---|---|---|
| 5 — bureau estimate | R36 100.00 | A stronger tier was available for every source. |
| 6 — client declared | R42 000.00 | A stronger tier was available for every source. Declared income exceeds evidenced income by R3 588.00. |

**Commission** averaged over **9 qualifying months** of a 12-month window.
3 months excluded: February 2026 (no record — not yet employed), May 2026
(once-off, R31 204.00 against a window median of R4 812.00 — back-pay), June 2026
(no record — payslip not supplied). Included at **90%** of the average.

A 13.00% verification haircut applied to source 3 for tier 3 evidence on
**contract employment**, from matrix version **2026-07-01**, cell (3, 2).
Modifiers added 0.00pp for statement confidence, **5.00pp** for employment tenure
(4 months on a contract) and 3.00pp for income variability, giving **13.00%**.

## 3. Statutory deductions

Statutory deductions of **R7 981.66**: income tax **R7 627.42**, unemployment
insurance **R354.24** across **2 employers**, compulsory retirement **R0.00**.
Net monthly income **R30 430.34**.

Income tax of R7 627.42 per month. Gross of R38 412.00 annualised to
R460 944.00, placing the applicant in **bracket 4** of table version
**2026-03-01** (base R121 506.00 plus 36.00% of the excess over R512 900.00…)

> *Note: the bracket figures above are the household's; the per-applicant
> computation is in the appendix, because tax is per person and the household
> total is a sum of two liabilities, not a liability on a sum.*

Rebate class **primary** applied, being the class for age **41.3 years** at
14 August 2026, worth R17 235.00 a year.

Unemployment insurance was computed **per employer**, not on the household total:
R264.00 for employer 1 (capped at the R177.12 ceiling → R177.12) and R177.12 for
employer 2. The household contribution of **R354.24 legitimately exceeds the
single-employer maximum of R177.12.**

## 4. Court-ordered deductions

Court-ordered deductions of **R1 200.00** under **1 instrument**: maintenance
order, magistrate's court, reference `MNT-2019-04418`.

*This instrument is not in respect of a credit agreement and therefore does not
suppress any account in section 6.*

## 5. Living expenses

All four bases were computed. The **statutory minimum expense norm** bound.

| Basis | Figure | |
|---|---|---|
| A — declared | R9 140.00 | across 11 of 12 categories asked |
| B — statement-derived | R10 882.14 | 83% category coverage over 3 months, debt service excluded |
| C — **statutory norm** | **R12 205.92** | **bound** |
| D — internal norm | R11 940.45 | Flex Loan, ×1.10 fixed and +0.5pp marginal |

The statutory minimum expense norm is **R12 205.92**: **band 8** of table version
**2026-03-01** at **2 dependants**, being a fixed component of **R4 888.26** plus
**8.95%** of the **R81 800.00** by which annualised gross income exceeds the band
floor of R35 000.00.

Household expenses were consolidated using classification version **2026-01-01**:
accommodation, water and electricity, food and insurance took the higher of the
two declarations; transport, medical, communication, education and maintenance
were summed.

No policy overlay altered this figure. `living_expenses` and
`living_expenses_unadjusted` are both **R12 205.92**.

## 6. Existing debt obligations

Existing monthly debt obligations of **R6 291.00** across **7 of 11 accounts
considered**, from treatment matrix version **2026-07-01**. **R1 842.00** on
facilities with the Bank and **R4 449.00** elsewhere. **4 accounts were
excluded**: 2 settled, 1 closed, 1 duplicate of an internal facility.

Each account, its treatment and the figure used, is listed overleaf (annex A).

A policy uplift of **R750.00** was added for credit-seeking activity: **3 credit
enquiries** in the 30 days to 12 August 2026, against a threshold of 2. Accounts
opened after 12 August 2026 are not visible on this bureau view.

### Annex A — every account considered

| Ref | Type | Treatment | Basis | Figure | Source | Note |
|---|---|---|---|---|---|---|
| A-01 | Personal loan | `STATED` | stated | R2 412.00 | bureau | |
| A-02 | Vehicle finance | `TERM_AWARE` | stated | R1 842.00 | **internal** | matures 2027-02-28, inside the proposed 48-month term. **Not reduced.** |
| A-03 | Credit card | `PCT_LIMIT` @ 5.00% | imputed from limit R24 000 | R1 200.00 | bureau | |
| A-04 | Store card | `PCT_LIMIT` @ 5.00% | imputed from limit R6 000 | R300.00 | bureau | |
| A-05 | Overdraft | `PCT_LIMIT` @ 3.00% | imputed from limit R9 000 | R270.00 | **internal** | bureau reflection A-09 merged; internal figures used |
| A-06 | Surety — instalment | `CONTINGENT` @ 20.00% | 20% of R1 335.00 | R267.00 | bureau | principal debtor not in arrears |
| A-07 | Telecoms | `PCT_BALANCE` @ 8.00% | imputed from balance R2 500, floored at R150 | R200.00 | bureau | |
| A-08 | Personal loan | `EXCLUDE_CLOSED` | — | R0.00 | both | settled internally **and** closed at the bureau |
| A-09 | Overdraft | — | — | R0.00 | bureau | merged into A-05, dedup group 4 |
| A-10 | Furniture | `EXCLUDE_CLOSED` | — | R0.00 | both | paid up |
| A-11 | Store card | `STATED` | stated | R0.00 | bureau | closed at the bureau only; **the Bank's internal view says settled but the bureau does not.** One source alone is not closure, so the account stays in at its stated instalment of R0.00 and the discrepancy is recorded. |
| — | Enquiry velocity uplift | `POLICY_UPLIFT` | policy | R750.00 | — | 3 enquiries in 30 days |

## 7. Discretionary income

```
    gross monthly income          R38 412.00
  - statutory deductions           R7 981.66
  = net monthly income            R30 430.34
  - living expenses               R12 205.92
  - court-ordered deductions       R1 200.00
  - existing obligations           R6 291.00
  = discretionary income          R10 733.42
```

## 8. The Bank's affordability buffer

The maximum affordable instalment before any policy overlay is **R8 333.42**.
Two constraints were computed and the **absolute residual floor** bound:

- a proportional buffer retaining **20.00%** of discretionary income (grade 6,
  Flex Loan, grid `standard`, version 2026-07-01) gave R8 586.74;
- an absolute residual floor of **R2 400.00** at 2 dependants gave
  **R8 333.42** — **binding**.

> The sentence "the absolute residual floor bound" is rendered from
> `affordability_buffer_basis_code`, which is the value the computation
> *selected*, not a restatement of the comparison. That matters more than it
> looks: a hand-written description of this stage would say "the proportional
> buffer bound" — it is the one everybody thinks of first, it is right in most
> cases, and the residual floor only binds for low-income applicants with
> dependants, who are exactly the population the floor exists to protect and
> exactly the population an ombud hears from. A narration generated from what
> ran cannot make that error. A maintained one makes it and keeps making it.

## 9. Policy overlays applied by the Bank

**One overlay applied.**

| Overlay | Target | Effect | Before | After |
|---|---|---|---|---|
| `ADJ-2026-014` | proportional buffer | +3.00 percentage points | 20.00% | 23.00% |

**ADJ-2026-014** — *Raise the affordability buffer on broker-originated Flex Loan
business following observed early-arrears deterioration.* Approved by the Credit
Committee on 19 August 2026 under reference `CC-2026-08-031`, in force from
1 September 2026 to 31 March 2027, review date 28 February 2027. Scope: products
10 and 11, channel 6, all grades. This application is within scope.

**The statutory calculation and the Bank's conservatism, side by side:**

| | |
|---|---|
| Maximum affordable instalment, **unadjusted** | **R8 333.42** |
| Maximum affordable instalment, **as applied** | **R8 263.73** |
| The difference, being the Bank's chosen conservatism | R69.69 |

## 10. The conclusion

**Verdict: PASS.** The proposed instalment of **R6 480.00** against a maximum
affordable instalment of **R8 263.73** leaves **R4 253.42** of discretionary
income.

---

### Artefact versions resolved on this assessment

| Artefact | Version | Owner |
|---|---|---|
| `statutory_expense_norms` | 2026-03-01 | Regulatory Compliance |
| `internal_expense_norms` | 2026-07-01 | Credit Risk Policy |
| `paye_brackets` | 2026-03-01 | Credit Systems |
| `paye_rebates` | 2026-03-01 | Credit Systems |
| `uif` | 2026-03-01 | Credit Systems |
| `obligation_treatment` | 2026-07-01 | Credit Risk Policy |
| `income_haircuts` | 2026-07-01 | Credit Risk Policy |
| `haircut_modifiers` | 2026-07-01 | Credit Risk Policy |
| `minimum_evidence_tier` | 2025-04-01 | Regulatory Compliance |
| `buffer_grid` | 2026-07-01 | Credit Risk Policy |
| `residual_floor` | 2026-07-01 | Credit Risk Policy |
| `expense_category_classification` | 2026-01-01 | Credit Risk Policy |
| `social_grant_amounts` | 2026-04-01 | Regulatory Compliance |

Structure fingerprint `f4a1…9c02` · params digest `7b30…11de` ·
params origin `git:config/affordability@a41f9c2` · adjustment set `417` ·
generation `19` · fallback set **empty**
