"""Codegen, caching and driver assembly — doc 05 §4-§7, doc 02 §3, §6.

Deliberately thin: importing this package must not require polars (only
numba and the fixed seams in `decider2.types`), so a project that only wants
`decider2.param`/`decider2.module` for authoring never pays for the compiled
path (same reasoning as `decider2/__init__.py`).
"""
from __future__ import annotations
