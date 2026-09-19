"""The regulator / ombud pack. ~40 a year, 10 working days each.

Ten working days is comfortable if the pack assembles in an afternoon and
impossible if it requires three teams to go looking. Everything in this module
is a query over evidence that already exists; nothing here asks a flow team for
anything, and that is the acceptance test.

THE ONE GENUINELY HARD PART IS ITEM 6, THE CONSISTENCY COHORT
  "Evidence of consistent application - that comparable applicants were
  treated comparably."

  This is the only capability in the whole project that reads OTHER PEOPLE'S
  decisions to answer a question about one person's. It is a frame-tier
  operation over a month of a flow's volume (150 000 records for flow 03), it
  touches 200 other data subjects' financial data to answer one complaint, and
  it is the part an adjudicator actually cares about - because the question
  they are asking is not "was the rule followed" but "was this person treated
  the way you treat people like them".

  Three consequences, all uncomfortable, all better named than discovered:
   - the cohort is PII about 200 people who did not complain. It never leaves
     the Bank un-aggregated; the pack contains the DISTRIBUTION and the
     complainant's position in it, plus the ONE distinguishing input, and never
     a row-level extract. The row-level extract exists, is retained with the
     pack, and is disclosed only under a specific request with its own approval.
   - the cohort definition is a versioned artefact owned by Compliance, per
     product, with declared keys, bands and a target size of 200. An
     analyst choosing the bands per complaint would be choosing the answer.
   - where the complainant's outcome differs from the cohort's mode, the pack
     MUST state which input or rule accounts for the difference. That is a
     computed statement, not a written one: it is the first link in the
     complainant's cap chain whose cause is rare in the cohort.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from explain.record import DecisionRecord
from replay.verdict import ReplayVerdict


class CohortDefinition:
    """Versioned, effective-dated, Compliance-owned. Flow 03, product 10:"""
    product_code: int
    keys: tuple[str, ...]              # ("product_code", "risk_grade", "decision_month")
    bands: dict[str, tuple[float, float]]   # requested_amount +/- 20%, discretionary_income +/- 25%
    target_size: int = 200
    version: str


def cohort(r: DecisionRecord, defn: CohortDefinition) -> pl.LazyFrame:
    """Frame tier. Widen the bands in declared steps until target_size is met;
    the widening applied is stated in the pack, because a cohort of 200 found
    by tripling the band is a weaker comparison than one found at the first
    band and the reader is entitled to know which."""
    pass


def distinguishing_factor(r: DecisionRecord, coh: pl.LazyFrame) -> str:
    """The first link in the complainant's chain whose cause is rare in the
    cohort. Computed from the chains, which every record has because of item 15.

    For FLX-2027-03-0418822: the group-exposure link (CAP-0361), present in
    11% of the cohort. That one sentence is what the adjudicator's
    determination will quote.
    """
    pass


def assemble(decision_id: str, *, request_ref: str, requester: str) -> "Pack":
    """The seven sections of spec §5.11, generated:

      1 the decision          - from the record
      2 the inputs            - as received, with sources, dates, bureau ref and as-at
      3 the explanation       - explain/render.adjudicator()
      4 the policy in force   - the APPROVED rendering at that decision's date,
                                extracted to the clauses that bore on this
                                decision, with the full rendering attached.
                                Extraction is by the record's branch paths:
                                a clause is in scope iff its rule appears in
                                the record's evaluated set. Nothing is selected
                                by judgement.
      5 the approval record   - who, which forum, which date, which impact
                                analysis. "The part most often missing, and its
                                absence converts a defensible decision into a
                                governance finding."
      6 consistency evidence  - cohort() + distinguishing_factor()
      7 provenance            - including the replay verdict, so the pack is
                                itself auditable
    """
    pass


def thematic(population_query: str, *, request_ref: str) -> "Pack":
    """Spec §11.1: every decline on a specified ground over 14 months,
    ~38 000 decisions, in 10 working days.

    Same seven sections, aggregated, PLUS the population definition and the
    query that produced it, reproducibly - which means the query is stored and
    re-runnable, not described. The per-decision detail is produced for the
    sample the requester specifies, on demand, because 38 000 adjudicator
    renderings is 300 000 pages and nobody wants them.

    The binding cost here is section 4: 14 months spans ~6 approved renderings
    of the flow, so the pack contains a rendering TIMELINE and the diff between
    consecutive ones, not one policy statement. Compliance asked for one
    statement; the honest answer is six and the diffs, and giving them one
    would be the finding.
    """
    pass


def issued(pack: "Pack") -> str:
    """Every pack issued, in full, immutably, with the request it answered.
    The Bank must be able to show, three years later, exactly what it sent -
    which means the pack is stored as RENDERED, not as a query that would
    re-render it, because the renderer will have changed."""
    pass
