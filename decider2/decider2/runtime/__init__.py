"""apply()/score() and the three execution modes — doc 02 §3.1, §3.5.

Imports polars (the batch entry point needs it); `decider2.compile` and
`decider2.params` do not, so this is the first layer in the package where
that dependency becomes real.
"""
from __future__ import annotations
