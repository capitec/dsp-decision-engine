"""Decision trees as data-shaped module interiors (doc 08 §3).

    from decider2 import flow
    from decider2.trees import Tree, tree_module

    risk = tree_module(Tree.model_validate(doc))
    pipeline = flow(Affordability, risk.module, Scoring).emit(risk.path_column)

A tree is a pydantic document (`schema.py`), emitted as numba-compilable
source (`codegen.py`) and wrapped as an ordinary `Module` (`build.py`).

MIGRATION NOTES — what did not come across from decider 1, and why:

* **`_ComputedFeature`** (`common/feature.py`): expression strings evaluated
  with `simpleeval` **at runtime**. That runtime step is what did not come
  across — not the feature. `decider2.expr` parses and validates the same
  wire format (`{"type": "computed", "expression": "x - y"}`) at document-load
  time and compiles it to numba source once, at build time (doc 08
  §1.1/§3.2, doc 06 §O15 — both revised from an earlier draft that removed
  this outright). `ComputedFeatureRemoved` is kept only for import
  compatibility; it is no longer raised. Prefer a step in a module before the
  tree instead when the derived value needs its own null policy, its own
  test, or reuse beyond one rule (doc 08 §3.2).
* **`match_type` other than `exact`, `case_sensitive=False`,
  `trim_whitespace=True`** (`common/nodes/operators.UnaryStringMatch`): a
  string reaches a kernel as an int32 dictionary code (doc 05 §1.5), and a
  code comparison cannot express a prefix, a substring or a regex — numba
  has no `re` in nopython. `UnsupportedInKernel` names the frame-tier route,
  which EXPERIMENTS.md §O measured at 7x faster than the Python one anyway.
* **`null_handling`** (`TNullHandling.match`/`no_match`/`error`): decider2
  declares null policy in a signature, not per condition (doc 03 §1's four
  tiers). A tree's nulls are handled by the pipeline's
  `.on_missing_input(...)` and the feature's own declared tier, which is one
  mechanism instead of two that must agree.
* **`output_fn`** (`flat_rules`' `{module_name, function_name}` pointer):
  forbidden by doc 08 §1.1 — "config may reference code by registered id. It
  may not contain code, and it may not carry an unregistered pointer." Doc
  01 §5.4 records what it cost decider 1: diagnostics became an
  unvalidated, non-inspectable side channel with no declared interface. The
  thing people actually used it for — path capture — is now a first-class
  column, `<name>_path`, emitted like any other value (doc 03 §7).
* **Auto-generated node ids.** decider 1's `CompositeCondition` fills a
  missing `id` with `str(uuid.uuid4())`, which doc 01 §5.4 calls out: path
  codes are then not comparable across versions. Ids here stay exactly as
  authored, and a missing one stays `None`, so emitted source is
  deterministic (doc 05 §4.2).
"""
from __future__ import annotations

from decider2.trees.build import TreeModule, tree_module
from decider2.trees.codegen import (
    LINE_CAP,
    EmittedTree,
    TreeTooLarge,
    UnsupportedInKernel,
    emit_tree,
)
from decider2.trees.schema import (
    CasesIsIn,
    CasesRanges,
    CasesStringMatch,
    ComputedFeatureRemoved,
    CompositeCondition,
    CompositeNode,
    Feature,
    InputRef,
    IsInCondition,
    LeafNode,
    MultiEdgeData,
    MultiSourceEdge,
    Position,
    PositionedNode,
    RangeCondition,
    RangeEndLogic,
    StringMatchCondition,
    SubTree,
    TLogicOp,
    Tree,
    TreeMetadata,
    TreeOutput,
    TStringMatchType,
    UnaryBetween,
    UnaryEqual,
    UnaryGreaterThan,
    UnaryGreaterThanEqual,
    UnaryIsFalse,
    UnaryIsIn,
    UnaryIsTrue,
    UnaryLessThan,
    UnaryLessThanEqual,
    UnaryNode,
    UnaryNotEqual,
    UnaryStringMatch,
    from_v2_range,
)

__all__ = [
    "Tree",
    "tree_module",
    "TreeModule",
    "emit_tree",
    "EmittedTree",
    "TreeTooLarge",
    "UnsupportedInKernel",
    "ComputedFeatureRemoved",
    "LINE_CAP",
    "TreeOutput",
    "TreeMetadata",
    "SubTree",
    "LeafNode",
    "UnaryNode",
    "CasesRanges",
    "CasesStringMatch",
    "CasesIsIn",
    "CompositeNode",
    "PositionedNode",
    "Position",
    "MultiSourceEdge",
    "MultiEdgeData",
    "RangeCondition",
    "StringMatchCondition",
    "IsInCondition",
    "CompositeCondition",
    "Feature",
    "InputRef",
    "RangeEndLogic",
    "TStringMatchType",
    "TLogicOp",
    "UnaryLessThan",
    "UnaryLessThanEqual",
    "UnaryEqual",
    "UnaryGreaterThan",
    "UnaryGreaterThanEqual",
    "UnaryNotEqual",
    "UnaryBetween",
    "UnaryIsIn",
    "UnaryIsTrue",
    "UnaryIsFalse",
    "UnaryStringMatch",
    "from_v2_range",
]
