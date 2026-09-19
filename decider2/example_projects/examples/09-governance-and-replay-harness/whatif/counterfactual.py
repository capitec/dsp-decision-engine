"""What-if, and making the result structurally incapable of being mistaken for
a decision - after it has been exported to a spreadsheet and emailed.

Spec §5.3 tells the real story: an analyst's exploratory re-run, exported,
emailed, and six weeks later quoted back to the Bank by a complainant's
attorney as what the Bank's own system says.

Doc 03 §6's `dbg.set("term_cap", 36.0)` is mechanically right - it uses real
compiled step code, so the numerics match production, which is spec §5.3's
"numerically faithful" requirement satisfied by the equivalence ladder rather
than by assertion - and governance-blind. `dbg.values` gives you a dict whose
keys are the flow's real output names. Print it and it IS a decision.

FOUR CHANGES, AND THE FOURTH IS THE ONLY ONE THAT SURVIVES A SPREADSHEET.

1. A what-if does not return outputs. It returns a `Counterfactual`, whose
   `.outputs` keys are prefixed `wi_` - so a copy-paste into a template that
   expects `offered_amount` fails rather than succeeds.

2. A `Counterfactual` cannot be constructed without an operator identity and a
   `because=` sentence. Not optional, not defaulted. ~1 800 runs a month, each
   one logged, retained 24 months, routinely requested during disputes - which
   means the `because` is read by people, which means it gets written.

3. The identifier lives in a separate namespace with a different shape:
   `WHATIF-2028-04-00913`, never `FLX-...`. It cannot be written to the
   decision store - not "should not": `store.put()` type-rejects it.

4. THE MARKING IS PER ROW, NOT PER DOCUMENT. This is the one that matters.
   A banner at the top of a document does not survive one row being copied out
   of it. So every serialisation - CSV, JSON, the clipboard form, the PDF -
   carries `__WHATIF__` as the FIRST FIELD OF EVERY ROW, and every monetary
   value is rendered with the counterfactual id appended:
   `R118 000 [WHATIF-2028-04-00913]`. Ugly on purpose. See
   artefacts/whatif-WHATIF-2028-04-00913.csv.

The residual hole, stated rather than hidden: a person who retypes a number
into an email defeats all four. FRAMEWORK-DEMANDS X2.
"""

from __future__ import annotations

from datetime import datetime

from replay.pin_resolution import PinSet


class Intervention:
    target: "Literal['input', 'param', 'cell', 'overlay']"
    name: str
    from_value: object
    to_value: object


class Counterfactual:
    whatif_id: str                       # WHATIF-2028-04-00913. Different namespace, different shape.
    of_decision: str                     # FLX-2027-03-0418822
    interventions: tuple[Intervention, ...]
    because: str                         # required, non-empty, free text, read by humans
    by: str                              # named individual; unmasked-access list (§5.13.2)
    at: datetime
    outputs: dict                        # keys prefixed wi_
    moved: tuple["Movement", ...]        # outputs that changed, gates that flipped, rules that
                                         # started or stopped firing, nodes visited instead
    engine: "Engine"                     # always IMAGE; a what-if in a lower engine is not evidence

    def to_csv(self) -> str:
        pass  # __WHATIF__ first column on EVERY row; ids appended to every money value

    def to_decision(self):
        raise TypeError("a counterfactual is not a decision and cannot become one")


def intervene(pins: PinSet, *interventions: Intervention, because: str, by: str) -> Counterfactual:
    """Re-run the decision as it stood with everything else held at its recorded
    values. < 5 s p95, because an analyst does this forty times in an afternoon
    and a 60-second turnaround means they stop doing it and start guessing.

    The budget is met by the record tier: `sealed.score(...)` on an already-warm
    image is ~40 us, so the 5 s is evidence retrieval and the pin is cached
    across the analyst's session. A frame-tier `apply()` for one record would
    spend more time crossing the polars boundary than deciding (doc 02 §3.5).
    """
    pass


def sweep(pins: PinSet, name: str, values: tuple, *, because: str, by: str) -> tuple[Counterfactual, ...]:
    """The thing analysts actually want and nobody builds: one input across a
    range, so the boundary is visible rather than bisected by hand. Forty
    single runs is the observed behaviour; one sweep is the same compute."""
    pass
