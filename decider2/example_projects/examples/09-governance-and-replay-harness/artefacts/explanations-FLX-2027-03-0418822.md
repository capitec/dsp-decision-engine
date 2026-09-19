# Three renderings of one decision

**Decision `FLX-2027-03-0418822`** · flow 03 unsecured granting · 2027-03-04
Generated from `artefacts/decision-record-FLX-2027-03-0418822.json` by
`explain/render.py`. Nothing here is written by a person. The prose comes from
the reason registry at `RC-2027-02` and from the `holds` clause sentences in
build `flow03@2027-02-28`.

The three are not the same document at three lengths. They are three
projections of one record through `explain/disclosure.py`, and the projection
is what makes the omissions structural: the consultant rendering does not
*omit* the score, it never receives it.

---

## 1 — Contact-centre consultant

Rendered at `Disclose.CONSULTANT`. Target: under 90 seconds, read aloud, no
credit training. Three languages from the registry; English shown.

Generated in 1.2 s. 41 words.

```
┌─ FLX-2027-03-0418822 ─ 4 March 2027 ─ Flex Loan ────────────────────────────┐
│                                                                             │
│  APPROVED FOR LESS THAN ASKED                                               │
│  Asked for R180 000 · Offered R95 000 over 72 months · R3 118.04 a month     │
│                                                                             │
│  MAIN REASON                                                                │
│  The amount we can offer is limited by this client's credit record with      │
│  other lenders, and by the credit they already hold with us.                 │
│                                                                             │
│  WHAT WOULD HAVE MADE THE BIGGEST DIFFERENCE                                 │
│  Reducing what they already owe us. They currently hold R104 000 of our      │
│  R180 000 group limit; that is what brought the offer down the furthest.     │
│                                                                             │
│  WHAT THE CLIENT CAN DO                                                      │
│  • Settle or reduce an existing Bank account and reapply.                    │
│  • Apply for the R95 000 offered, which stands until 18 March 2027.          │
│  • Ask for the full written reasons, which we must give within 20 days.      │
│                                                                             │
│  ALSO TELL THEM, IF ASKED                                                    │
│  Our own lending policy is currently more cautious than usual for            │
│  applications through the app. That affected this assessment. It is a Bank   │
│  decision, not something about this client, and it is reviewed on            │
│  31 March 2027.                                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

Four things about this box:

- **No threshold, no score, no grade, no code.** Not by editorial habit: the
  cut-off is a param at ownership class *library-policy*, which carries
  `Disclose.ANALYST`, so the projection drops it before the template runs.
  Spec §5.2's explanation-versus-gaming tension resolves as a classification,
  not as a style guide.
- **"What would have made the biggest difference" is computed**, not written.
  It is the link in the `amount_cap` chain with the largest delta —
  CAP-0361, R120 000 → R76 000, a fall of R44 000, against CAP-0100's
  R350 000 fall which is not actionable by the client. *Largest actionable
  delta*, where actionable is a flag on the clause.
- **The last paragraph is the overlay**, and §5.14.5 requires it for all three
  audiences. A consultant who does not know the Bank tightened will tell the
  client something false when asked "why has this changed since last year".
  The wording is Compliance's, attached to the overlay's *kind*, not written
  per overlay.
- **What it costs to be wrong here is the reason for the 99.5% availability
  requirement.** 31 000 of these a year, live, with a client on the phone.

---

## 2 — Credit analyst (extract)

Rendered at `Disclose.ANALYST`. Full document is 7 pages; the cap waterfall
section is reproduced verbatim. Generated in 4.8 s.

```
FLX-2027-03-0418822   flow03@2027-02-28   fp:9a3d41c7e0b58f22   grade A
────────────────────────────────────────────────────────────────────────────────
GRADE ASSIGNMENT                                       base        adjusted
  score                                                631.0       613.0
  probability_of_default                               0.0412      0.0554
  risk_grade                                           6           7
  overlay                                              ADJ-2027-008  -18 pts
                                                       CC-2026-41, expires 2027-06-30
  >> THE GRADE MOVE IS THE OVERLAY. Without ADJ-2027-008 this application is
     grade 6, and CAP-0100 at sequence 3 would have seeded R240 000 rather
     than R150 000. Three of the five subsequent binds are unaffected; see
     the counterfactual below.

CAP WATERFALL — amount_cap                  52 rules evaluated, 4 bound
  seq  rule       applicable  before        after         status
   1   CAP-0010   yes         —             R500 000.00   seed
   3   CAP-0100   yes         R500 000.00   R150 000.00   BOUND   appetite, grade 7
   9   CAP-0140   no          R150 000.00   R150 000.00   n/a     client since 2019-08
  12   CAP-0175   yes         R150 000.00   R150 000.00   no bind tenure 41m > 24m
  17   CAP-0210   yes         R150 000.00   R120 000.00   BOUND   acct 31, 2m arrears 2026-01
  21   CAP-0240   yes         R120 000.00   R120 000.00   no bind 3 enquiries < 6
  26   CAP-0305   no          R120 000.00   R120 000.00   n/a     sector 205 not listed
  33   CAP-0330   yes         R120 000.00   R120 000.00   no bind channel 2
  38   CAP-0361   yes         R120 000.00   R76 000.00    BOUND   180 000 − 104 000
  52   CAP-0420   yes         R76 000.00    R95 000.00    RAISED  campaign 4471 / CC-2026-14
  ── 42 rules evaluated, not applicable, no bind: expand ──
  final R95 000.00  bound by CAP-0420  coincident: none

  >> CAP-0420 raised to its full 25% authority and was NOT restrained. Had a
     regulatory-class rule been binding, both the authorised value and the
     restraining rule would appear here.

SCORE CONTRIBUTIONS                                    signed, ranked, binned
  −34.0  months_since_worst_arrears   14      bin 12-18
  −21.0  enquiries_60d                3       bin 3-5
  +48.0  months_on_book               91      bin 84+
  −17.0  utilisation_revolving        0.62    bin 0.50-0.75
  +29.0  employment_type_code         1       bin permanent
   −6.0  vehicle_finance_balance      NULL    bin NULL   ← null is a bin, not an error
   −9.0  dependants_count             2       bin 2-3

CELLS READ                                             version         value
  appetite_grid[g7,p10,s1,max_amount]                  AG-2027-Q1      R150 000
  rate_card.flex[amt_090-100k, term_72, grade_7]       RC-FLEX-2027-03 0.2475
  credit_life[age_34_38, term_61_72, emp_1]            CLP-2027        4.50/1000
  expense_norms.internal[band_7, dep_2, food]          EXN-INT-2027-01 R2 410
  … 7 more

INTERVENE  (whatif/intervene — results are WHATIF-namespaced and cannot be issued)
  [i] requested_amount   [i] existing_internal_exposure   [i] months_since_worst_arrears
  [p] cap_register.group_headroom_factor   [o] disable ADJ-2027-008   [s] sweep
```

The analyst rendering is the only one that is *also* an intervention surface.
Spec §5.3's 5-second budget exists because an analyst does this forty times in
an afternoon; the handles being inline is why they do it rather than guess.

---

## 3 — Ombud adjudicator (opening pages)

Rendered at `Disclose.ANALYST` minus internal identifiers, every remaining
identifier glossed. Reviewed by Compliance before issue; the review is itself
recorded. Full document 9 pages plus appendices; see
`artefacts/ombud-pack-OMB-2028-0117.md` for the whole pack.

> ### What the Bank decided
>
> On 4 March 2027 the complainant applied through the Bank's mobile
> application for a Flex Loan of R180 000 over 72 months. The Bank approved a
> loan of R95 000 over 72 months at an instalment of R3 118.04 a month. It did
> not refuse the application; it offered less than was asked for.
>
> ### How the amount was arrived at
>
> The amount was not chosen. It is the result of five separate policies
> applied in a fixed order, each of which reduced or raised a ceiling. The
> Bank records every step of that sequence for every application. For this
> application the sequence was:
>
> | | The Bank's policy | What it did |
> |---|---|---|
> | 1 | The Flex Loan product may not exceed R500 000. | Set the starting ceiling at **R500 000**. |
> | 2 | At the risk grade the Bank assigned to this application (grade 7 of 12, where 1 is best), the Bank will not advance more than R150 000 on an unsecured loan. This limit is set by the Bank's Credit Committee and applies to every applicant at that grade. | Reduced the ceiling to **R150 000**. |
> | 3 | Where an applicant has been two or more months in arrears on any credit account in the previous 24 months, the amount is limited to R120 000. The credit bureau record obtained on 3 March 2027 shows account 31 two months in arrears in January 2026. | Reduced the ceiling to **R120 000**. |
> | 4 | The Bank limits its total exposure to one client. The complainant's group limit was R180 000 and the complainant already held R104 000 of Bank credit, leaving R76 000. | Reduced the ceiling to **R76 000**. |
> | 5 | Where an applicant is on a pre-approved campaign authorised by the Credit Committee, the ceiling may be raised by up to 25%. The complainant was on campaign 4471, authorised under Credit Committee resolution CC-2026-14 of 11 August 2026. | **Raised** the ceiling to **R95 000**. |
>
> Forty-seven further policies were applied to this application and did not
> reduce the amount. They are listed in Appendix B with the reason each did
> not apply, because a policy that was considered and did not apply and a
> policy that did not apply at all are different facts.
>
> ### An important distinction: the Bank's own caution
>
> Step 2 above turns on the risk grade. The Bank's scorecard — a statistical
> model, validated independently and approved in May 2026 — assessed this
> application at **grade 6**, which would have permitted **R240 000** at step 2.
>
> The grade used was **7**, not 6, because of a separate decision by the Bank.
> In January 2027 the Bank's Credit Committee approved a temporary tightening
> of 18 points for applications made through the mobile application by clients
> new to this product, on the grounds of deteriorating early-life arrears in
> that channel (Credit Committee resolution CC-2026-41 of 9 December 2026,
> effective 1 January 2027, due for review on 31 March 2027).
>
> This tightening is not a judgement about the complainant. It applies to
> every applicant meeting the same description. The Bank records it separately
> from the model's own assessment precisely so that this distinction can be
> made, and it is stated here because the adjudicator is entitled to know
> which of the two produced the outcome.
>
> **Had the tightening not been in force, the amount offered would still have
> been R95 000.** Step 2 would have set R240 000 instead of R150 000, but
> steps 3, 4 and 5 are unaffected by the grade, and step 4 — the
> complainant's existing R104 000 of Bank credit — would still have reduced
> the ceiling to R76 000 before the campaign uplift raised it to R95 000.
> The Bank has re-run the application with the tightening removed to
> establish this; the result is at Appendix D and is marked as a simulation.
>
> ### Glossary of every term used above
>
> *Risk grade* — a number from 1 to 12 the Bank assigns to each application,
> where 1 is the lowest assessed risk … *Group limit* — the total amount of
> credit the Bank is willing to extend to one client across all products …
> *Campaign* — a marketing programme under which selected existing clients are
> pre-approved for a specified amount …

Three properties of the adjudicator rendering that the other two do not have:

- **Every identifier is glossed by `explain/render.gloss()`**, which fails
  loudly if a rule has no clause sentence. A rule with no `holds` cannot be
  explained to an ombud, and the failure is visible to the team that owns the
  rule rather than to nobody.
- **The counterfactual is inside the narrative**, marked as a simulation, with
  its own `WHATIF-` identity in the appendix. Answering "would it have been
  different without the overlay" is the single question that determines whether
  a complaint succeeds, and answering it with a guess is how a Bank loses.
- **The policy statements are the same strings the Credit Committee approved**
  — the `holds` sentences from `REN-flow03-2027-Q1-v3`. Not a paraphrase
  written by disputes. If they were a paraphrase, the pack and the approved
  rendering could disagree, and spec §5.7 is explicit that that divergence is a
  finding on its own.
