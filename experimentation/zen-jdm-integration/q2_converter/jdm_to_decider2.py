"""JDM decisionTableNode -> decider2 DecisionTable.

The subset this converter accepts, deliberately narrower than JDM's real
surface (found by trying, not assumed up front — see `NotConvertible`
messages, which are the actual findings):

  - `hitPolicy: "first"` only. JDM's `collect` (return every matching row,
    or every matching row per output column via a `[]`-suffixed field) has
    no decider2 equivalent — decider2 tables are "first match wins, one row
    of output," full stop (schema.py's `DecisionTable` docstring).
  - Every input column must be *targeted* (a real `field`, unary-test
    cells) and every cell in that column must parse as ONE of: a numeric
    range test (`>`, `>=`, `<`, `<=`, `[a..b]` and the three other bracket
    combinations), a quoted string-literal set (`"a","b"`), or a bare
    `true`/`false`. A *generic* column (`field: "-"`, a full ZEL expression
    per cell) has no decider2 equivalent — decider2's `Expression` tree
    tests one named variable against row data, never an arbitrary
    per-row expression.
  - Every output cell must be a literal (a number, a quoted string, or
    `true`/`false`). An output cell that is itself a ZEL expression
    (`input.amount * 0.1`) has no decider2 equivalent — `DecisionTable.
    outputs` names columns of `parameters.data`, which are values, never
    formulas (`tables/__init__.py`'s migration notes make the same point
    about the *old* decider 1 `output_fn`).
  - A trailing all-wildcard row (every input cell `""`) becomes decider2's
    `default` — the one shape where JDM's per-row wildcard and decider2's
    single fallback coincide exactly.
  - A NUMERIC column additionally needs its rows' own stated bound
    operators to be reproducible by ONE of decider2's two global
    `BoundMode`s (`lower_inclusive`: every row's lower uses `>=`, every
    row's upper uses `<`; or `upper_inclusive`: the mirror). decider2
    applies one bound-inclusivity choice to every row in the table; JDM
    lets every cell pick its own bracket independently. A row that states
    BOTH bounds with the SAME inclusivity (`[a..b]` — both closed, seen in
    the ZEN repo's own `credit-analysis.json` "Turnover" node) can never be
    produced by a single global `BoundMode` and is refused, exactly (with
    the row and the boundary value named), rather than silently converted
    with a one-value boundary error — see `run_table_conversions.py` for a
    live demonstration of exactly where that boundary error would land.
  - A non-numeric column's non-trailing wildcard cell is convertible only
    if the column is Eq-shaped (at most one value per row) — decider2's
    `EqExpression` skips a row whose column value is `None`
    (`tables/schema.py`, confirmed in q1). `InExpression` has no such
    skip (an empty per-row value set means "never matches," not "always
    matches" — found while building this converter, see `EMPTY_IN_IS_NOT_
    WILDCARD` below) so an In-shaped column's non-trailing wildcard is
    refused.
"""
from __future__ import annotations

import dataclasses
import re
import typing as t

from decider2.tables import (
    AndExpression,
    BetweenExpression,
    BoundMode,
    DecisionTable,
    EqExpression,
    Expression,
    InExpression,
    ParametersConfig,
)

EMPTY_IN_IS_NOT_WILDCARD = (
    "decider2's InExpression treats an empty per-row value set as 'this row's condition "
    "can never be satisfied,' not 'this row's condition is skipped' — the opposite of "
    "JDM's blank-cell wildcard. Only EqExpression (at most one value/row) has the None-"
    "skip decider2 needs to reproduce a non-trailing wildcard cell."
)


class NotConvertible(Exception):
    pass


@dataclasses.dataclass
class ConversionReport:
    rows_in: int = 0
    rows_out: int = 0
    default_from_trailing_wildcard: bool = False
    column_kinds: dict[str, str] = dataclasses.field(default_factory=dict)
    notes: list[str] = dataclasses.field(default_factory=list)


_RANGE_RE = re.compile(r"^([\[\(])\s*(-?[\d_]+(?:\.\d+)?)\s*\.\.\s*(-?[\d_]+(?:\.\d+)?)\s*([\]\)])$")
_CMP_RE = re.compile(r"^(<=|>=|<|>)\s*(-?[\d_]+(?:\.\d+)?)$")
_NUM_RE = re.compile(r"^-?[\d_]+(?:\.\d+)?$")
_STRSET_RE = re.compile(r'^\s*(?:"[^"]*"|\'[^\']*\')(?:\s*,\s*(?:"[^"]*"|\'[^\']*\'))*\s*$')
_STR_ITEM_RE = re.compile(r'"([^"]*)"|\'([^\']*)\'')


def _num(s: str) -> float:
    return float(s.replace("_", ""))


@dataclasses.dataclass
class Bound:
    lo: float | None
    lo_incl: bool | None  # None means "not stated"
    hi: float | None
    hi_incl: bool | None


def _parse_range_cell(cell: str) -> Bound | None:
    """None means wildcard (empty cell). Raises NotConvertible for anything
    that isn't a range/comparison cell (caller only reaches this once the
    column has been classified as numeric)."""
    cell = cell.strip()
    if cell == "":
        return None
    m = _RANGE_RE.match(cell)
    if m:
        lo_br, lo_s, hi_s, hi_br = m.groups()
        return Bound(_num(lo_s), lo_br == "[", _num(hi_s), hi_br == "]")
    m = _CMP_RE.match(cell)
    if m:
        op, val_s = m.groups()
        val = _num(val_s)
        if op in (">", ">="):
            return Bound(val, op == ">=", None, None)
        return Bound(None, None, val, op == "<=")
    if _NUM_RE.match(cell):
        v = _num(cell)
        return Bound(v, True, v, True)
    raise NotConvertible(f"cell {cell!r} does not parse as a numeric range/comparison test")


def _classify_column(cells: list[str]) -> str:
    non_empty = [c.strip() for c in cells if c.strip() != ""]
    if not non_empty:
        return "empty"
    if all(c in ("true", "false") for c in non_empty):
        return "bool"
    if all(_RANGE_RE.match(c) or _CMP_RE.match(c) or _NUM_RE.match(c) for c in non_empty):
        return "range"
    if all(_STRSET_RE.match(c) for c in non_empty):
        return "string_set"
    return "unsupported_expression"


def _parse_string_set(cell: str) -> list[str] | None:
    cell = cell.strip()
    if cell == "":
        return None
    return [a or b for a, b in _STR_ITEM_RE.findall(cell)]


def _parse_output_literal(cell: str) -> t.Any:
    cell = cell.strip()
    if cell == "true":
        return True
    if cell == "false":
        return False
    if _NUM_RE.match(cell):
        v = _num(cell)
        return int(v) if v.is_integer() else v
    m = _STR_ITEM_RE.match(cell)
    if m and (m.group(0) == cell):
        return m.group(1) if m.group(1) is not None else m.group(2)
    raise NotConvertible(
        f"output cell {cell!r} is not a literal (number / quoted string / true / false) — "
        "looks like a computed ZEL expression, which decider2 outputs cannot hold"
    )


def convert_table(content: dict, *, name: str = "converted_table") -> tuple[DecisionTable, ConversionReport]:
    report = ConversionReport()
    if content.get("hitPolicy", "first") != "first":
        raise NotConvertible(
            f"hitPolicy={content.get('hitPolicy')!r} — decider2 tables are first-match-wins "
            "only; JDM's 'collect' (return every matching row) has no equivalent"
        )

    inputs = content["inputs"]
    outputs = content["outputs"]
    for o in outputs:
        if o.get("field", "").endswith("[]"):
            raise NotConvertible(
                f"output {o['id']!r} field {o['field']!r} is a per-column collect field — "
                "no decider2 equivalent (implies hitPolicy semantics decider2 doesn't have)"
            )
    for i in inputs:
        if i.get("field") in (None, "-"):
            raise NotConvertible(
                f"input {i['id']!r} is a generic column (field={i.get('field')!r}, full-expression "
                "cells) — decider2's Expression tree only tests one named variable per condition"
            )

    rules = list(content["rules"])
    report.rows_in = len(rules)

    # Trailing all-wildcard row -> decider2 `default`.
    default_values: list[t.Any] | None = None
    if rules:
        last = rules[-1]
        if all(last.get(i["id"], "").strip() == "" for i in inputs):
            default_values = [_parse_output_literal(last[o["id"]]) for o in outputs]
            rules = rules[:-1]
            report.default_from_trailing_wildcard = True

    if not rules:
        raise NotConvertible("no rows left after popping the trailing wildcard row")

    # Classify every input column.
    col_kind: dict[str, str] = {}
    for i in inputs:
        cells = [r.get(i["id"], "") for r in rules]
        kind = _classify_column(cells)
        if kind == "unsupported_expression":
            raise NotConvertible(
                f"column {i['name']!r} ({i['id']}) has a cell that is neither a numeric range, "
                "a quoted string set, nor a boolean literal — not a targeted unary test this "
                "converter understands"
            )
        col_kind[i["id"]] = kind
        report.column_kinds[i["name"]] = kind

    # Build one Expression per column, then AND them together.
    n = len(rules)
    expressions: list[Expression] = []
    param_data: list[dict[str, t.Any]] = [dict() for _ in range(n)]

    for i in inputs:
        cid, cname, kind = i["id"], i["name"], col_kind[i["id"]]
        cells = [r.get(cid, "") for r in rules]
        safe_name = re.sub(r"\W+", "_", cname).strip("_").lower() or cid

        if kind == "range":
            bounds = [_parse_range_cell(c) for c in cells]
            wildcard_rows = [idx for idx, b in enumerate(bounds) if b is None]
            if wildcard_rows:
                raise NotConvertible(
                    f"column {cname!r} row(s) {wildcard_rows} are wildcarded but not trailing — "
                    "decider2's BetweenExpression has no per-row skip (every row needs a "
                    "resolvable bound); only a trailing all-wildcard ROW (all columns at once) "
                    "maps, to `default`"
                )
            lo_incls = {b.lo_incl for b in bounds if b.lo is not None and b.lo_incl is not None}
            hi_incls = {b.hi_incl for b in bounds if b.hi is not None and b.hi_incl is not None}
            both_stated_same = [
                (idx, b) for idx, b in enumerate(bounds)
                if b.lo is not None and b.hi is not None and b.lo_incl == b.hi_incl
            ]
            if both_stated_same:
                idx, b = both_stated_same[0]
                raise NotConvertible(
                    f"column {cname!r} row {idx} states BOTH bounds with the SAME "
                    f"inclusivity ([{b.lo}..{b.hi}]{'  (both closed)' if b.lo_incl else ' (both open)'}) "
                    "— decider2's BoundMode is one global (lower_inclusive XOR upper_inclusive) "
                    "choice for the whole table, which can only ever make ONE side of a range "
                    "inclusive; a row that is closed (or open) on both ends can never be "
                    "reproduced by any single BoundMode"
                )
            if len(lo_incls) > 1 or len(hi_incls) > 1 or (lo_incls and hi_incls and lo_incls == hi_incls):
                raise NotConvertible(
                    f"column {cname!r}: rows disagree on bound inclusivity in a way no single "
                    f"BoundMode reproduces (lower-bound inclusivity seen: {lo_incls}, "
                    f"upper-bound inclusivity seen: {hi_incls})"
                )
            lower_inclusive = (True in lo_incls) or (False in hi_incls) if (lo_incls or hi_incls) else True
            mode = BoundMode.lower_inclusive if lower_inclusive else BoundMode.upper_inclusive

            order = sorted(range(n), key=lambda idx: (bounds[idx].lo if bounds[idx].lo is not None else float("-inf")))
            sorted_bounds = [bounds[idx] for idx in order]
            for k in range(len(sorted_bounds) - 1):
                a, b = sorted_bounds[k], sorted_bounds[k + 1]
                a_hi = a.hi
                b_lo = b.lo
                if a_hi is None or b_lo is None or a_hi != b_lo:
                    raise NotConvertible(
                        f"column {cname!r}: rows (original index {order[k]}, {order[k+1]}) are not "
                        f"contiguous under the detected BoundMode ({a_hi=} vs {b_lo=}) — "
                        "gap or overlap between bands"
                    )
            if order != list(range(n)):
                report.notes.append(
                    f"column {cname!r}: JDM rows were NOT in ascending band order "
                    f"(hitPolicy=first priority order {list(range(n))} -> decider2 storage order "
                    f"{order}) — decider2's row position doubles as band identity, so the "
                    "converter had to re-sort. Output values stayed attached to their original row."
                )

            lo_col, hi_col = f"{safe_name}_lo", f"{safe_name}_hi"
            for pos, idx in enumerate(order):
                b = bounds[idx]
                param_data[pos][lo_col] = b.lo
                param_data[pos][hi_col] = b.hi
            # `rules`/`param_data` must now be reindexed by `order` for every
            # column, so remember the permutation and apply once, at the end.
            expressions.append(BetweenExpression(
                type="between", variable=safe_name,
                lower_bound_column=lo_col, upper_bound_column=hi_col,
                mode=mode, allow_gaps=True,
            ))
            # This pass's fill is correct as long as no OTHER column also
            # needs reordering to a DIFFERENT permutation — the dedicated
            # rebuild pass below handles that by redoing every column's
            # fill from scratch in `final_order`, so this pass only matters
            # when `final_order` ends up == identity (nothing to redo).

        elif kind == "bool":
            val_col = f"{safe_name}_eq"
            for pos, c in enumerate(cells):
                c = c.strip()
                param_data[pos][val_col] = None if c == "" else (c == "true")
            expressions.append(EqExpression(type="eq", variable=safe_name, value_column=val_col))

        elif kind == "string_set":
            parsed = [_parse_string_set(c) for c in cells]
            max_len = max((len(p) for p in parsed if p is not None), default=0)
            if max_len <= 1:
                val_col = f"{safe_name}_eq"
                for pos, p in enumerate(parsed):
                    param_data[pos][val_col] = None if p is None else p[0]
                expressions.append(EqExpression(type="eq", variable=safe_name, value_column=val_col))
            else:
                wildcard_rows = [idx for idx, p in enumerate(parsed) if p is None]
                if wildcard_rows:
                    raise NotConvertible(
                        f"column {cname!r} row(s) {wildcard_rows} are wildcarded but not trailing, "
                        f"and the column needs >1 value on at least one row (InExpression, not Eq). "
                        f"{EMPTY_IN_IS_NOT_WILDCARD}"
                    )
                vals_col = f"{safe_name}_vals"
                for pos, p in enumerate(parsed):
                    param_data[pos][vals_col] = p
                expressions.append(InExpression(type="in", variable=safe_name, values_column=vals_col))

    outputs_by_row: list[list[t.Any]] = [
        [_parse_output_literal(r[o["id"]]) for o in outputs] for r in rules
    ]

    # Re-derive final row order from whichever numeric column (if any) set it;
    # non-numeric-only tables keep JDM's original order (still first-match-wins
    # correct — In/Eq/bool conditions don't have a position-dependent identity
    # the way Between's contiguous bands do).
    final_order = list(range(n))
    for i in inputs:
        if col_kind[i["id"]] == "range":
            cells = [r.get(i["id"], "") for r in rules]
            bounds = [_parse_range_cell(c) for c in cells]
            final_order = sorted(range(n), key=lambda idx: (bounds[idx].lo if bounds[idx].lo is not None else float("-inf")))
            break

    if final_order != list(range(n)):
        # Rebuild cleanly: redo every column's fill, in final row order,
        # from scratch — simpler and less bug-prone than patching the
        # single-column-at-a-time fill above onto a new permutation.
        param_data = [dict() for _ in range(n)]
        for i in inputs:
            cid, cname, kind = i["id"], i["name"], col_kind[i["id"]]
            cells_orig = [r.get(cid, "") for r in rules]
            safe_name = re.sub(r"\W+", "_", cname).strip("_").lower() or cid
            if kind == "range":
                bounds = [_parse_range_cell(c) for c in cells_orig]
                lo_col, hi_col = f"{safe_name}_lo", f"{safe_name}_hi"
                for pos, idx in enumerate(final_order):
                    param_data[pos][lo_col] = bounds[idx].lo
                    param_data[pos][hi_col] = bounds[idx].hi
            elif kind == "bool":
                val_col = f"{safe_name}_eq"
                for pos, idx in enumerate(final_order):
                    c = cells_orig[idx].strip()
                    param_data[pos][val_col] = None if c == "" else (c == "true")
            elif kind == "string_set":
                parsed = [_parse_string_set(c) for c in cells_orig]
                max_len = max((len(p) for p in parsed if p is not None), default=0)
                if max_len <= 1:
                    val_col = f"{safe_name}_eq"
                    for pos, idx in enumerate(final_order):
                        p = parsed[idx]
                        param_data[pos][val_col] = None if p is None else p[0]
                else:
                    vals_col = f"{safe_name}_vals"
                    for pos, idx in enumerate(final_order):
                        param_data[pos][vals_col] = parsed[idx]
        outputs_by_row = [outputs_by_row[idx] for idx in final_order]

    out_names = []
    for o in outputs:
        safe = re.sub(r"\W+", "_", o["name"]).strip("_").lower() or o["id"]
        out_names.append(safe)
    for pos in range(n):
        for oname, val in zip(out_names, outputs_by_row[pos]):
            param_data[pos][oname] = val

    dtypes: dict[str, str] = {}
    for col in param_data[0]:
        sample = next((row[col] for row in param_data if row.get(col) is not None), None)
        if isinstance(sample, bool):
            dtypes[col] = "Boolean"
        elif isinstance(sample, str):
            dtypes[col] = "String"
        elif isinstance(sample, int):
            dtypes[col] = "Int64"
        else:
            dtypes[col] = "Float64"

    expression: Expression = expressions[0] if len(expressions) == 1 else AndExpression(type="and", expressions=expressions)

    table = DecisionTable(
        name=name,
        parameters=ParametersConfig(data=param_data, dtypes=dtypes),
        expression=expression,
        outputs=out_names,
        default=default_values,
    )
    report.rows_out = n
    return table, report
