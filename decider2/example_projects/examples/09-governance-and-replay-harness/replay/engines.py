"""Three replay engines, in decreasing fidelity, and why the equivalence ladder
is the seven-year insurance policy.

Spec §13 Q6 asks: is replay a re-execution of the same logic, or a separate
interpreter? A re-execution risks not being available in seven years; an
interpreter risks not agreeing with production.

The framework docs answer this without knowing they do. Doc 02 §3.1 defines
`interpreted == stepped == fused` as an automatically tested three-way
agreement, with disagreement localising to a layer. That is exactly the
property that makes a lower-fidelity engine ADMISSIBLE EVIDENCE rather than a
guess, because the agreement was demonstrated at the time, on that build, over
that flow's corpus, and the demonstration is itself a retained artefact.

So: three engines, one ladder, and the engine used is stamped on the verdict.

  IMAGE       the flow's own container image, by `compiled_artefact_id`, run
              network-denied with the PinSet mounted. Bit-identical by
              construction. The only engine whose verdict may be `reproduced`.
              Expected lifetime: 3-5 years before a base image will not run.

  SOURCE      the generated numba driver sources (doc 02 §3.4 requires them to
              be real .py files, deterministic and byte-identical) recompiled
              against a pinned decider2 wheel and a pinned numba. Survives the
              image; does not survive a numba major that changes rounding.
              Verdict ceiling: `reproduced_within_tolerance`.

  INTERPRETED pure-Python execution of the manifest's graph. No numba, no
              polars, no image - just the graph as data and the step sources.
              This is the engine that still works in 2039. Verdict ceiling:
              `reproduced_within_tolerance`, and the tolerance is the one the
              ladder measured on that build, not a tolerance invented later.

The 1e-12 intermediate tolerance of spec §5.1 is EXACTLY the ladder's tolerance
budget, and the "not at all on any monetary figure after rounding" clause is
why doc 03 §1.2's `round_half_up` and scaled-int64 cents are load-bearing for
this project and not merely tidy: if money were float, the INTERPRETED engine
would drift a cent and a cent is a reconciliation failure.

The annual seven-year sample (spec §5.1, 1 000 decisions per year for seven
years) is what detects engine rot. Its real job is not to pass; it is to tell
you the year the IMAGE engine stopped working, while there is still time.
"""

from __future__ import annotations

from enum import Enum

from replay.pin_resolution import PinSet
from replay.seal import ReadCoverage


class Engine(Enum):
    IMAGE = "image"
    SOURCE = "source"
    INTERPRETED = "interpreted"


def run(pins: PinSet, *, engine: Engine = Engine.IMAGE) -> "ReplayRun":
    """Execute. Single record: `score()`, never `apply()`.

    doc 02 §3.5 - the realtime path bypasses polars entirely, and a one-record
    `apply()` sits far below the ~10k-row viability floor. A replay budget of
    5 s p95 is dominated by evidence retrieval, not by execution; the execution
    is ~40 us. This is the harness's own record-tier/frame-tier split and it
    runs the same way as the framework's: per-decision work is record tier,
    population work (swap-set, coverage, cohorts) is frame tier.
    """
    pass  # resolve the image by build id, mount the pin read-only, deny the network, score


def ladder_evidence(build: str) -> "LadderResult":
    """The retained proof that this build's three modes agreed, with the
    measured tolerance per output class. Fetched, never recomputed - the
    machine that could recompute it is the machine that may not exist."""
    pass


def degrade(build: str) -> Engine:
    """Pick the highest-fidelity engine still available for this build."""
    pass  # image if pullable and runnable; else source if the wheel resolves; else interpreted
