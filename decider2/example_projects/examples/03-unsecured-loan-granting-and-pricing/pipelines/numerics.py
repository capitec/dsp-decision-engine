"""The numerics contract for this pipeline.

Doc 03 §1.2 establishes that money is scaled int64 cents, that bare `round()`
means three different things in three execution modes, and that `fastmath`
makes 46-73% of rows differ by up to 17 ULP.  Doc 02 §3.3 then makes `fastmath`
a per-kernel choice an author makes locally.

Those two cannot both hold here.  Acceptance criterion 11 is "real-time and
batch produce identical outputs on a 100 000-record reconciliation sample, zero
differences" and criterion 14 is "a 2026 decision replayed in 2030 reproduces
to the cent".  A single author enabling `fastmath` on a single hot kernel four
files away breaks both, silently, and the equivalence ladder would not catch it
because doc 02 §3.1 *excludes* a fastmath kernel from the exact-agreement
assertion by design.

So this project declares numerics at the pipeline level, and the declaration
wins over any per-kernel choice inside it.  `with_numerics(...)` is part of the
structure fingerprint, so a change to it is a release with a new audit
identity.  See FRAMEWORK-DEMANDS #21.
"""

from __future__ import annotations

from decider2.numerics import Numerics, Rounding

FLEX_NUMERICS = Numerics(
    # Money is int64 cents everywhere.  R500 000 is 50_000_000; the widest
    # intermediate in this pipeline is total_cost_of_credit on a R500 000 /
    # 84-month agreement, about 1.1e8 cents, so int64 has 11 orders of
    # headroom.  Accumulators over money (the disclosure five-way split, the
    # expected-value objective) are declared float64 per doc 03 §1.2 -- exact
    # to 2^53, about R90 trillion in cents.
    money="int64_cents",
    accumulate="float64",
    # Rates are int32 basis-points x 100 (18.25% -> 182_500).  Four decimals
    # is core.rounding's published convention.  This also makes the rate card
    # 63 360 x 4 bytes = 253 KB, which is L2-resident -- the reason 140 000
    # lookups/second is not interesting (see README, "Why the grid is not the
    # problem").
    rate="int32_bp100",
    rounding=Rounding.HALF_UP,       # round_half_up only; bare round() is a lint error
    # Hard bans, enforced at build.  Each one is a wrong-answer class rather
    # than a style preference.
    fastmath=False,                  # overrides any per-kernel fastmath=True below it
    allow_decimal=False,             # Decimal raises a Rust panic at the boundary
    allow_float_money=False,         # core's published float64 money is converted at
                                     # the frame boundary; see FRAMEWORK-DEMANDS #20
    # `parallel(...)` regions must contain no cross-row reduction.  Checked
    # statically against the graph, not hoped for.
    parallel_requires_no_reduction=True,
)
