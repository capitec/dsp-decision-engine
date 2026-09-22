"""A tree document as a pipeline element (doc 03 §5, §5.3).

`tree_module(doc)` returns an ordinary `types.Module`. Not a subclass, not a
special node kind, not a new element type the pipeline has to know about —
the same frozen dataclass `module(fn, name=...)` builds, so
`flow(Affordability, my_tree, Scoring)` works because there is nothing to
make work. Doc 03 §8 states the invariant this preserves: "There is **one
type** — Module — and three combinators over it."

That is also the answer to the question doc 03 §8 leaves open for this work.
`Branch` and `Loop` are specified there but not built. **A tree does not
supersede `Branch`, and this deliberately does not build one.** They answer
different questions:

* `Branch(cond, ArmA, ArmB, modifies=[...])` routes between *modules* —
  arbitrary Python bodies, each its own scope. Its hard rule is cross-arm
  type agreement on `modifies`, and its payoff is that only the taken arm
  executes.
* A tree routes between *leaf values* in one closed vocabulary. Its arms are
  rows of a table, so they agree on type by construction, and it is a
  document a UI writes rather than code an engineer writes.

A tree is the data-shaped interior of doc 08 §3; a `Branch` is skeleton, and
doc 08 §2 is explicit that "config may not add a module, rewire modules, or
introduce a `Branch`". Building `Branch` out of trees would put skeleton in
config. Building trees out of `Branch` would need `Branch` to exist first
and would gain nothing: a tree's emitted body is already the nested
`if`/`elif` a `Branch` compiles to (doc 05 §4.3 shows exactly that shape).
So `Branch`/`Loop` remain unbuilt and this work neither blocks nor
duplicates them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from decider2.graph.module import module
from decider2.trees.encode import EncodedTree, encode_tree
from decider2.trees.schema import Tree
from decider2.types import Module

__all__ = ["TreeModule", "tree_module", "DEFAULT_BUILD_DIR"]

# Kept only for backward-compatible signatures (a `build_dir=` a caller
# still passes) — nothing writes source under it any more (doc 05 §4.1's
# whole reason for a content-addressed build dir was "numba cannot cache a
# function with no source file"; a tree's Step.fn is a real njit closure
# built directly, never written to disk).
DEFAULT_BUILD_DIR = Path(".decider2_build")


@dataclass(frozen=True)
class TreeModule:
    """A built tree: its `Module`, plus what only the document knows.

    The `Module` is what a pipeline takes. This wrapper carries the two
    things that are true of the *document* rather than of the compiled
    steps: which string-valued output columns were not compiled (and so
    need `decode`), and the report `explain()` prints.
    """

    module: Module
    tree: Tree
    encoded: EncodedTree

    @property
    def name(self) -> str:
        return self.module.name

    @property
    def path_column(self) -> str:
        """The column naming which leaf was reached — doc 03 §7's
        `<Name>_path`. Emit it like any other value: `.emit("risk_path")`."""
        return self.encoded.path_column

    def decode(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Add the string-valued output columns back.

        A kernel writes float64/int64/bool arrays only
        (`boundary.writeback.KernelOutputs`), so a string leaf value has no
        array to land in. It is mapped here instead, in polars, from the
        `<name>_path` column the tree already produces — one `replace_strict`
        per column, vectorised, outside the kernel. Doc 05 §1.5 and
        EXPERIMENTS.md §O both put string work on this side of the boundary
        anyway: the dictionary route measured 31x faster end to end than
        carrying strings through a kernel.

        A no-op when the tree declares no string columns.
        """
        path_col = self.path_column
        if path_col not in frame.columns:
            raise KeyError(
                f"decode() needs the '{path_col}' column, which carries which leaf "
                f"each row reached. It is produced by the tree module but may have "
                f"been dropped — pass it through, or add .emit('{path_col}')."
            )
        exprs = []
        for column, dtype in self.tree.output.dtypes:
            if dtype not in ("String", "Utf8"):
                continue
            mapping = {i: row.get(column) for i, row in enumerate(self.tree.output.data)}
            default = (self.tree.output.default or {}).get(column)
            exprs.append(
                pl.col(path_col)
                .replace_strict(mapping, default=default, return_dtype=pl.String)
                .alias(column)
            )
        return frame.with_columns(exprs) if exprs else frame

    def explain(self) -> str:
        """What was built, for a reviewer — the tree analogue of
        `pipeline.explain_kernels()`. No more "emitted lines"/"cap 500":
        nothing is emitted any more (doc 08 §3.4) — `fn` is a pre-built
        njit closure over the tree's own arrays, so there is no line count
        or line cap left to report.
        """
        e = self.encoded
        tunable = [p for p in e.params if p.annotation == "float"]
        literals = [p for p in e.params if p.annotation == "str"]
        output_names = [s.name for s in e.output_steps]
        lines = [
            f"tree {self.tree.name!r} -> module {self.module.name!r}",
            f"  leaves          : {e.leaf_count}",
            f"  max depth       : {e.max_depth}",
            f"  features read   : {', '.join(e.features) or '(none)'}",
            f"  string features : {', '.join(e.string_features) or '(none)'}",
            f"  path column     : {self.path_column}",
            f"  output steps    : {', '.join(output_names) or '(none)'}",
            f"  thresholds      : {len(tunable)} (kernel arguments, retune is free)",
            f"  string literals : {len(literals)} (int32 codes, retune is free)",
        ]
        return "\n".join(lines)


def tree_module(
    tree: Tree,
    *,
    name: str | None = None,
    build_dir: "str | Path | None" = None,
    params: Mapping[str, Any] | None = None,
) -> TreeModule:
    """Compile a tree document into a pipeline element.

    `name` defaults to the document's own `name`, and becomes the module
    instance name — which is the namespace every threshold is addressed
    under (doc 03 §4.1, §10). So a tree called `risk` retunes as
    `pipeline.apply(frame, params={"risk": {"score_thr": 700.0}})`, and the
    same names appear in `pipeline.params_schema()` and over the serving
    `/params` endpoint, with no extra wiring: they are `param()` fields on
    the tree's own `Step`s, indistinguishable from hand-written ones (doc
    03 §4.4).

    `build_dir=` is accepted only for backward-compatible call sites; it is
    unused (see `DEFAULT_BUILD_DIR`'s own docstring).

    `params=` pre-binds thresholds at composition (doc 03 §4.3's `.bind()`),
    for a value that is settled and should leave the caller-facing
    interface.
    """
    del build_dir
    encoded = encode_tree(tree, name=name)

    steps = list(encoded.matcher_steps) + [encoded.path_step] + list(encoded.output_steps)

    instance_name = name or tree.name or "tree"
    built = module(*steps, name=instance_name)
    if params:
        built = built.bind(**dict(params))

    return TreeModule(module=built, tree=tree, encoded=encoded)
