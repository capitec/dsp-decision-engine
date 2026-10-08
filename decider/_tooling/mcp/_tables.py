"""Decision tables: read a table as a grid and edit its rows and columns, saved only if the table still builds."""
from __future__ import annotations

import typing as t

from decider._tooling.mcp._configs import Refusal, read, refusing, save

_BOUND_COLUMNS = ("value_column", "lower_bound_column", "upper_bound_column", "values_column")


def _load(path: str) -> dict:
    doc = read(path)
    if not isinstance(doc, dict) or doc.get("type") != "decision_table":
        raise Refusal(f"{path} is not a decision_table document")
    if "columns" not in doc or "rows" not in doc:
        raise Refusal("this table uses the older `parameters` spelling; rewrite it with `columns` and `rows`")
    return doc


def _param_rows(doc: dict) -> dict | None:
    rows = doc["rows"]
    return rows if isinstance(rows, dict) and "table" in rows else None


def _inline(doc: dict) -> list[dict]:
    """The table's own row list, mutable in place."""
    ref = _param_rows(doc)
    if ref:
        raise Refusal(f"the rows of '{doc['name']}' live in the params document (param '{ref['table']}'); "
                      "edit them there with params_set")
    return doc["rows"]["data"] if isinstance(doc["rows"], dict) else doc["rows"]


def _match(e: dict, top: bool = True) -> str:
    kind = e["type"]
    if kind in ("and", "or"):
        text = f" {kind.upper()} ".join(_match(s, False) for s in e["expressions"])
        return text if top else f"({text})"
    v = e["variable"]
    if kind == "eq":
        return f"{v} == row.{e['value_column']}"
    if kind == "in":
        return f"{v} in row.{e['values_column']}"
    if kind == "is_true":
        return f"{v} is true"
    lo, hi = e.get("lower_bound_column"), e.get("upper_bound_column")
    upper_closed = e.get("mode") == "upper_inclusive"
    left = f"row.{lo} {'<' if upper_closed else '<='} " if lo else ""
    right = f" {'<=' if upper_closed else '<'} row.{hi}" if hi else ""
    return f"{left}{v}{right}"


def _cell(v: t.Any) -> str:
    return "null" if v is None else str(v).replace("|", "\\|")


def _grid(columns: list[str], rows: list[dict], offset: int) -> str:
    lines = ["| # | " + " | ".join(columns) + " |", "|---" * (len(columns) + 1) + "|"]
    lines += [f"| {offset + i} | " + " | ".join(_cell(r.get(c)) for c in columns) + " |" for i, r in enumerate(rows)]
    return "\n".join(lines)


def table_view(path: str, offset: int = 0, limit: int = 200) -> dict:
    """A decision table: columns, outputs, default, how a record matches, and rows from `offset` (a grid and a list)."""
    doc = _load(path)
    view = {"path": path, "name": doc["name"], "columns": doc["columns"], "outputs": doc["outputs"],
            "default": doc.get("default"), "match": _match(doc["expression"]) + "  (first matching row wins; none: default)"}
    ref = _param_rows(doc)
    if ref:
        where = f"shared.{ref['table']}" if ref.get("shared") else f"<step path>/{ref['table']}"
        return {**view, "rows_in": "params",
                "edit": f"rows are the param '{ref['table']}' ({where} in the params document, one dict per row, a value for "
                        "every column); read them with params_view(path, entry) and change them with params_set"}
    rows = _inline(doc)
    shown = rows[offset:offset + limit]
    return {**view, "rows_in": "document", "row_count": len(rows), "offset": offset,
            "rows_markdown": _grid(list(doc["columns"]), shown, offset), "rows": shown}


@refusing
def table_add_rows(path: str, rows: list[dict], at: int | None = None) -> dict:
    """Insert `rows` at index `at` (default: the end); saved only if the table still builds."""
    doc = _load(path)
    own = _inline(doc)
    at = len(own) if at is None else at
    if not 0 <= at <= len(own):
        raise Refusal(f"at={at} is outside 0..{len(own)}")
    own[at:at] = rows
    result = save(path, doc)
    return {**result, "added_at": list(range(at, at + len(rows)))} if result["saved"] else result


@refusing
def table_update_rows(path: str, updates: list[dict]) -> dict:
    """Change cells: `updates` is `[{"index": 3, "values": {"rate": 0.1}}]`; saved only if the table still builds."""
    doc = _load(path)
    own = _inline(doc)
    for u in updates:
        if not 0 <= u["index"] < len(own):
            raise Refusal(f"row index {u['index']} is outside 0..{len(own) - 1}")
        own[u["index"]].update(u["values"])
    return save(path, doc)


@refusing
def table_delete_rows(path: str, indexes: list[int]) -> dict:
    """Delete the rows at `indexes`; saved only if the table still builds."""
    doc = _load(path)
    own = _inline(doc)
    bad = [i for i in indexes if not 0 <= i < len(own)]
    if bad:
        raise Refusal(f"row indexes {bad} are outside 0..{len(own) - 1}")
    for i in sorted(set(indexes), reverse=True):
        del own[i]
    return save(path, doc)


@refusing
def table_add_column(path: str, name: str, dtype: str | dict, value: t.Any = None, output: bool = False) -> dict:
    """Add column `name` (a polars dtype name like "Float64") holding `value` in every row; `output` also writes it."""
    doc = _load(path)
    if name in doc["columns"]:
        raise Refusal(f"column '{name}' already exists")
    doc["columns"][name] = dtype
    if not _param_rows(doc):
        for row in _inline(doc):
            row[name] = value
    if output:
        doc["outputs"].append(name)
        if doc.get("default") is not None:
            doc["default"].append(value)
    return save(path, doc)


@refusing
def table_drop_column(path: str, name: str) -> dict:
    """Remove column `name` from the columns, every row and the outputs; refused while the match expression uses it."""
    doc = _load(path)
    if name not in doc["columns"]:
        raise Refusal(f"no column '{name}'; columns are {list(doc['columns'])}")
    used = [e for e in _leaves(doc["expression"]) if name in {e.get(k) for k in _BOUND_COLUMNS}]
    if used:
        raise Refusal(f"column '{name}' is used by the match expression ({_match(used[0])}); change the expression "
                      "first (patch_config on /expression), then drop the column")
    del doc["columns"][name]
    if not _param_rows(doc):
        for row in _inline(doc):
            row.pop(name, None)
    if name in doc["outputs"]:
        i = doc["outputs"].index(name)
        del doc["outputs"][i]
        if doc.get("default") is not None:
            del doc["default"][i]
    return save(path, doc)


def _leaves(e: dict) -> t.Iterator[dict]:
    if e["type"] in ("and", "or"):
        for sub in e["expressions"]:
            yield from _leaves(sub)
    else:
        yield e
