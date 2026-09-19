"""THE PROBE.  One candidate amount, one term, one grade -> one priced offer.

§13.9 asks whether `core.rate_card`, `core.fees`, `core.credit_life`,
`core.instalment` and `core.rounding` are one composite the library should
publish or five calls the consumer assembles -- "and if the latter, what stops
six consumers assembling them differently?"

Nothing does.  So this is the composite, and it is the unit the search counts.
One probe = one execution of this module = one row of evidence.  Making the
probe the composite rather than the five calls buys three things:

  * the evaluation ceiling counts something meaningful.  "24 pricing
    evaluations" is 24 executions of THIS, not 24 rate lookups and however many
    fee calls somebody happened to write.
  * the evidence record has a fixed shape.  Every probe emits the same eleven
    fields, so "why R50 000 and not R49 900?" is answered by two rows of one
    table rather than by correlating four logs.
  * the final validation has something to be independent OF.  `FinalValidation`
    declares `independent_of=[PriceCandidate]`, and the framework checks that
    no value this module produced reaches the validator.  Without a named
    composite there is nothing to name.
"""

from __future__ import annotations

from decider2 import fuse, module
from decider2.overlay import overlay

from adjustments.points import RATE_ADD_ON
from modules.pricing import credit_life, fees, instalment, rate_card


PriceCandidate = fuse(
    # (a) the rate.  A cell read, and the cell id travels with it.
    rate_card.Lookup
    # (e) the rate add-on.  Applied AFTER lookup and BEFORE the annuity, with
    #     the card's own cell value and the add-on recorded separately -- so a
    #     regulator gets "cell 18.50% plus a 75 basis point overlay approved
    #     under CC-2026-22" rather than a single unexplained 19.25%.
    | overlay(RATE_ADD_ON)
    # (b) fees.  Capitalised into the advance; amount financed != amount advanced.
    | fees.Fees
    # (c) credit life.  On the amount financed, so it inherits the fee's kink.
    | credit_life.CreditLife
    # (d) the instalment, the total cost and the effective rate.
    | instalment.Instalment
).named("price_candidate")

# `fuse(...)` here and nowhere else in this project.  Doc 02 §1.1 measured
# fusion as non-monotone and harmful past ~5 modules; this group is four
# modules and it is executed up to 216 times per application, which is the one
# place where per-call overhead amortisation is worth having.  It changes
# codegen only and is asserted on the equivalence ladder, so it cannot change a
# price.  Everywhere else in the pipeline the default -- one kernel per module
# -- stands.

# --- what every probe emits, in order, as one evidence row -----------------
#
#   candidate_amount, amount_band_index, rate_cell_id, nominal_annual_rate_card_value,
#   rate_add_on_bp, nominal_annual_rate, initiation_fee, amount_financed,
#   credit_life_premium, instalment, feasible
#
# Eleven fields x 216 probes = 2 376 values per application.  At 14.2 M
# applications a month that is 34 billion values, which is why `evidence=` has
# levels -- see modules/solve/search.py and FRAMEWORK-DEMANDS #16.
