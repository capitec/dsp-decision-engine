"""Compare two versions of a pipeline: structure, outputs over data, git revisions, and proof.

`structure` says what changed without running anything, `outputs` which rows
and columns change over a frame, `revisions` does either between two git
revisions, and `prove` says whether two pipelines are provably identical.

Example::

    from decider import diff
    diff.structure(old, new).changed_steps
    diff.outputs(old, new, frame, key="application_id").rows_changed
    diff.prove(old, new).identical      # True, or None when it can't tell
"""
from decider.diff._outputs import ColumnDiff, OutputDiff, outputs
from decider.diff._prove import Proof, SmtEvidence, prove
from decider.diff._revisions import Revisions, revisions
from decider.diff._structure import (EdgeChange, ParamChange, StepChange, StepMove, StructureDiff, structure)

__all__ = [
    "ColumnDiff", "EdgeChange", "OutputDiff", "ParamChange", "Proof", "Revisions", "SmtEvidence", "StepChange", "StepMove",
    "StructureDiff", "outputs", "prove", "revisions", "structure",
]
