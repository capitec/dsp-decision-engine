<!-- GENERATED from the rule definition + the params generation + the overlay
     stack. Never hand-edited. This is doc 04 s6's reviewable artefact, sized
     for a rule set of 635 rather than a waterfall of 30: nobody reads 635
     diagrams, so the unit is a one-page sheet and the index is a table.

     Three readers, one artefact:
       - the author, confirming it says what she authorised (criterion 16)
       - the operations agent, who has 90 seconds (s7)
       - the regulator, for whom this is an inventory entry (s9.1)
     Anything that serves only one of the three is in a section the other two
     can skip. -->

# MS-0208 — First payment to a very new beneficiary at a high-mule-rate bank

|  |  |
|---|---|
| **Status** | live since 2026-03-01 |
| **Family** | MS — mule and scam |
| **Owner** | Fraud Strategy · N. Dlamini |
| **Rule version** | 7 |
| **Shape version** | 2 *(unchanged since 2026-02-27 — only thresholds have moved)* |
| **Severity** | 4 of 5 |
| **Action asked for** | hold for review → queue SCAM-1, SLA 15 minutes |
| **Critical** | no |
| **Overlay-exempt** | no — thresholds may be moved by an approved overlay |
| **Reason shown to client** | 4412 · "We've paused this payment to check it with you" |

## What it does

> First payment to a beneficiary added less than two hours ago, above R8 000,
> where the receiving bank is in the top mule-rate band and the client changed
> device within 72 hours. The four together are the authorised-push-payment
> scam signature after a remote-access takeover.

## When it applies

| | |
|---|---|
| Event types | instant payment (210), EFT credit (211), beneficiary addition (313) |
| Segments | all except business-linked |
| Degraded modes | suspended in **restricted** and **fail-closed** |
| Effective | from 2026-03-01 00:00 UTC, no end date |

## The test, as it will run on the next event

Four conditions, all must hold.

| # | Feature | Test | Authored value | **In force now** | Why it differs |
|---|---|---|---|---|---|
| 1 | beneficiary age | less than | 2 hours | **3 hours** | threshold change, 2026-09-11 14:00, approved by R. Pillay |
| 2 | amount | greater than | R8 000.00 | **R4 550.00** | threshold change to R6 500, then ×0.70 by overlay **ADJ-0031** |
| 3 | receiving bank mule band | at least | 4 | **3** | threshold change, 2026-09-11 14:00 |
| 4 | hours since device change | less than | 72 hours | 72 hours | — |

> **This rule is currently modified by an overlay.** ADJ-0031 "Festive mule/scam
> amount tightening" multiplies every amount threshold in the MS family by 0.70.
> It was approved on 2026-10-19 by H. Abrahams and T. Mokoena, it expires on
> 2027-01-15, and **47 days remain**. Without it condition 2 would read R6 500.
> The rule's own definition still says R8 000 and has not been edited.

## When an input is missing or stale

| Feature | If stale or absent |
|---|---|
| beneficiary age | **rule is suppressed** — recorded as unevaluable, not as "did not fire" |
| amount | **rule is suppressed** |
| receiving bank mule band | last known good value is used |
| hours since device change | treated as false — the rule does not fire on this alone |

## Limits and protections

| | |
|---|---|
| Maximum fire rate | 0.40% of applicable events in any 5 minutes |
| If exceeded | self-demotes and pages N. Dlamini within one 5-minute window |
| Currently tripped | no |

## How it has performed

| Period | Version | Overlay state | Firings | Confirmed fraud | Precision | Incremental catch |
|---|---|---|---|---|---|---|
| 2026-03 to 2026-09-11 | 7 (base) | none | 41 208 | 9 044 | 21.9% | 3 112 |
| 2026-09-11 to 2026-10-19 | 7 (retuned) | none | 58 771 | 11 402 | 19.4% | 3 880 |
| 2026-10-19 to now | 7 (retuned) | ADJ-0031 | 81 344 | 13 907 | 17.1% | 4 201 |

*Incremental catch = confirmed fraud this rule caught that no other live rule
caught. It is the number that matters, and it is 5.2% of firings.*

## Approval history

| When | What changed | Class | Author | Approver | Backtest |
|---|---|---|---|---|---|
| 2026-02-27 11:04 | created | shape | N. Dlamini | T. Mokoena | BT-2026-0227-14 |
| 2026-09-11 13:52 | three thresholds | **threshold** | N. Dlamini | R. Pillay | BT-2026-0911-03 |

*A threshold change and a shape change are different rows, different approval
routes and different deployment paths. The 2026-09-11 change was live 6 minutes
14 seconds after it was proposed, of which 11 milliseconds was machine time.*
