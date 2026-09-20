"""Testing support for decider2 pipelines — doc 02 §3.1's equivalence ladder
and doc 05 §9's acceptance criteria, as first-class, reusable assertions
instead of something a handful of tests happen to exercise incidentally.

    assert_equivalent(pipeline, frame, **kwargs)
        Runs interpreted/stepped/fused and asserts EXACT agreement, naming
        the rung, column and row of the first divergence (doc 05 §9.1).

    assert_no_recompile(pipeline, frame, params_a, params_b)
        Asserts a retune leaves the compiled driver untouched — the
        property the entire config story rests on (doc 05 §9 criterion 5,
        doc 08 §2).

    corpus(source)
        Generates a boundary-value frame (zero, negative, the null policy's
        edge, int64 near 2**53) plus an empty-frame case, per declared input
        (doc 05 §9, EXPERIMENTS.md §I).
"""
from __future__ import annotations

from decider2.testing.corpus import corpus
from decider2.testing.equivalence import assert_equivalent
from decider2.testing.recompile import assert_no_recompile

__all__ = ["assert_equivalent", "assert_no_recompile", "corpus"]
