"""Sealing, and the failure mode where a replay SUCCEEDS with the wrong answer.

This module exists for one sentence in spec 09 §5.15.4: "A flow that reads the
current date anywhere cannot be replayed, and - worse - will replay
*successfully* with the wrong answer." Generalise it and it is the central risk
of the whole project: a silent, internally consistent, plausible, wrong replay,
with no flag, quoted back to the Bank in 2033.

Four mechanisms, in increasing order of how much they cost and how rarely they
are thought of.

1. NO OUTSIDE. `at_pin` returns a pipeline whose table reads, date resolution,
   capability lookups and overlay resolution all terminate in the PinSet. Not
   "prefer the pin" - there is no fallback path to prefer it over. A read with
   no pin raises.

2. READ COVERAGE, BOTH WAYS. After the run, compare reads performed against
   pins supplied:

     unpinned_reads  -> HARD FAIL. The pin was incomplete; the flow's witness
                        set is wrong; contract item 5, 7 or 10 is violated for
                        this flow and the finding is against the flow.
     unread_pins     -> NOT a failure, but ALWAYS reported and usually the
                        interesting one. It means the replay took a different
                        path from the original - which is either the defect you
                        are looking for, or a sign that the pin was assembled
                        from the wrong decision.

3. SCHEMA PINNING, not just value pinning. This is the one that actually bites
   and it is not in any framework doc.

   doc 08 §2.1's `resolve_params(doc, complete=True)` proves a document names
   every field the model declares. It proves it AGAINST THE MODEL OF THE DAY.
   Three years later `core.affordability` is at major 3, the model has a field
   `marginal_band_width` that did not exist in 2027, and it has a code default.
   The 2027 params document is complete for the 2027 schema and silently
   incomplete for the 2030 one. `complete=True` passes. The replay runs. The
   answer moves by one band at the margin, on maybe 0.4% of records, in a
   defensible-looking direction.

   So the PinSet stores `params_schema_hash` per module, and `seal` refuses on
   schema drift rather than resolving it. FRAMEWORK-DEMANDS D8.

4. CLOCK DENIAL AT THE PROCESS BOUNDARY. The sandbox refuses `clock_gettime`
   for wall-clock, refuses DNS, refuses sockets. Belt and braces over item 4:
   a step that reads today cannot njit and so shows in `fallback_set`, but a
   FRAME-tier operation (`pl.date.today()`) or a `@breaks_lineage` region never
   reaches numba and the structural check cannot see it. That gap is real and
   is written up in FRAMEWORK-DEMANDS D6.
"""

from __future__ import annotations

from decider2 import Pipeline

from replay.pin_resolution import PinSet


class UnpinnedRead(Exception):
    """Raised inside a sealed pipeline. Carries the read, the module, and the
    contract item it violates, so the message is actionable by the flow team
    rather than by the person replaying."""


class SchemaDrift(Exception):
    """The params document is complete for the schema of the day it was
    written and incomplete for the schema in the image. Never resolved
    silently; always a verdict of `not_reproduced`, never `reproduced`."""


def seal(pipeline: Pipeline, pins: PinSet) -> "SealedPipeline":
    """Bind every outside reference to the pin. No fallbacks, no defaults, no today."""
    pass  # rewrite table/date/capability/overlay resolvers to read pins; install the read log


class ReadCoverage:
    unpinned_reads: tuple["Read", ...]       # hard fail
    unread_pins: tuple[str, ...]             # reported, often the finding
    cells_read: tuple[tuple[str, int], ...]  # (table_version, flat_index) - feeds coverage/cells


def cover(sealed: "SealedPipeline", pins: PinSet) -> ReadCoverage:
    """The both-ways comparison. Run after every replay, every what-if, every
    certification record. It is cheap (a set difference) and it is the only
    thing standing between this project and a plausible wrong answer."""
    pass
