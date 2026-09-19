"""Stage 5.4 (part 2) -- the application scorecard.

Four scorecards x 45 characteristics x <=8 bins.  The output that makes this
hard is not the score; it is the **45 point contributions, as required output,
for every application including declined ones** (§5.4, §9.1).  180 values --
raw value, bin index, points, contribution-vs-neutral -- per application.

Doc 03 has no vocabulary for a step producing 180 values.  Writing them as 180
named scalar steps is absurd; writing them as a struct is not expressible; and
a `tap` is a diagnostic, not required output, so tapping them would put a
regulatory obligation behind a switch somebody can turn off.

So a `scorecard` is a declared kind whose interface is
`(characteristic vector) -> (score, Contributions[45])`, where `Contributions`
is a fixed-capacity structured output the framework materialises as four
parallel arrays.  It is tabular -- bins to points, uniform -- so doc 08 §3.4
puts it on a generic kernel: a model release is an interior document, not a
compile.  See FRAMEWORK-DEMANDS #10.
"""

from __future__ import annotations

from decider2 import module, scorecard
from decider2.collections import Contributions

Score = scorecard(
    name="application_scorecard",
    # Which of the four definitions is used is selected per record by
    # `scorecard_id`, which segment.py produced.  One kind, four instances, one
    # kernel -- the definitions are rows in the interior, not four modules.
    selected_by="scorecard_id",
    characteristics=45,
    max_bins=8,
    # Null is its own bin, ALWAYS, with its own points.  For the thin-file
    # scorecard 14 of 45 characteristics are null for more than a third of
    # applicants, so a framework that treats a null as an extraction error
    # makes segment 1012 unscoreable.  `null_bin=REQUIRED` means a definition
    # that omits a null bin for any characteristic fails interior validation.
    null_bin="REQUIRED",
    # Contributions are expressed relative to the population-neutral points for
    # that characteristic, so "you scored below average on this" is meaningful.
    contributions=Contributions(
        capacity=64,                      # 45 today, 52 in the self-employed segment
        emit=["raw_value", "bin_index", "points", "contribution_vs_neutral"],
        required=True,                    # NOT a tap.  Cannot be switched off.
    ),
    reason_codes=4,                       # top four negative contributions
    reason_registry="core.reason_codes",
    interior="config/flex_loan/interiors/scorecards/",
    writes=["score", "scorecard_version", "score_reason_codes"],
)

# The challenger.  Same kind, different instance, different params namespace,
# no overlay points attached in the pipeline expression -- which is the whole
# mechanism for "the challenger is overlaid separately or not at all".
Challenger = Score.at(
    outputs={
        "score": "challenger_score",
        "score_reason_codes": "challenger_score_reason_codes",
        "contributions": "challenger_contributions",
    },
    inputs={"scorecard_id": "challenger_scorecard_id"},
).named("challenger")
