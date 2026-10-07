# Handoff: library gaps for moving the tree UI server onto decider

Written: 2026-10-07. Repo: `dsp-decision-engine`.
The user asked for this to be implemented by a Sonnet subagent.

## Context

The tree editor (`data-science-platforms_dsp-decider`,
FastAPI server in `dsp_decider/`, React editor in `ui/`) validates tree documents with an
in-repo package, `dspd`. `dspd` is being deleted and the server will depend on decider instead.
The server work runs in parallel in that repo; it codes against the contract below.

Verified so far: documents the server stores today (dspd's `model_dump(by_alias=True)`, v3 trees
saved as `"type": "ui-tree"`, and `prioritized_flat_rule`) load through `load_document` and run.
The server will store the document **as the editor sent it** and use decider only to validate,
so decider dropping fields on dump is no longer a problem for the editor.

Already done, don't redo: `ui-tree` accepted (uncommitted in `schema/v3.py`), missing output
columns are null, `default: {}` is no default, Custom categorical outputs, v3 edges from an unknown
source are dropped. v1/v2 stay unsupported: the user has no v1/v2 trees left and will port any
stragglers by hand.

## Tasks

1. **`input_schema` on tree documents.** The editor writes `input_schema: [[column, dtype], ...]`
   on v3 trees and prioritized flat rules (dspd had it as `Optional[PolarsSchema]`). Add it back
   to `BaseTreeDocument` as an optional field that validates the structure (`TStruct`) and
   round-trips on dump. Don't wire it into `feature_types`: the editor auto-adds every feature it
   sees as `String`, so trusting it would mistype numeric features. Don't parse dtypes strictly:
   the editor writes bare spellings like `"List"` and `"Datetime"`.
   Done when: a v3 and a prioritized flat document keep `input_schema` through
   `load_document(d).model_dump(by_alias=True)`, and a document without it still loads.

2. **v3 tree to flat rules.** The user wants this in the library (the server's
   `/export?as_type=flat_rule` endpoint is being dropped, so nothing in the server calls it).
   Convert a `V3TreeDocument` to a `FlatRuleDocument`, starting from the root `_root_id` picks
   (what the v3 tree runs). It keeps `name`, `output`, `parameters` and `input_schema`, and keeps
   node ids as the flat nodes' `id`.
   - Unary and composite nodes map to `then` / `otherwise`.
   - A cases node's conditions become `CasesBranch(when=condition, then=i)`. `otherwise` is the
     last branch, and `branches` are the children in order.
   - A branch with no edge is `LeafRule(result_idx=-1)` (or `None` for `then` / `otherwise`).
   - A node shared by two parents is repeated under the same id, which `add_node` already allows
     when both copies are identical.
   Done when: for every v3 fixture in `tests/trees/`, the flat document runs identically to the
   v3 tree in all three modes (`run_modes` in `tests/trees/conftest.py`). Include a shared-node
   (DAG) case and a several-roots case.

3. **Public `PolarsSchema`.** The server's data sources describe table schemas with it and
   currently can't import it without reaching into private `decider._serde`.
   - **Contract the server codes against:** `from decider import PolarsSchema`.
   - The server calls `PolarsSchema.from_polars(pl_schema)`, `.model_validate(...)` and `.schema`.
     It was told the dspd name `from_polars_schema` is gone, so no alias is needed.
   - If a top-level export breaks the layer rules in
     `notes/pythonapi-redesign/directory-structure.md`, pick another public path and tell the
     user, so the server import can be changed.

4. **`decider.__version__`.** The server reports it in `/server/info` and in exports.
   - **Contract:** `decider.__version__` is a `str`.
   - The build uses hatch-vcs, so `importlib.metadata.version("decider")` is the stdlib route.
     hatch-vcs's `version-file` build hook is the versioneer-like alternative.
   - An uninstalled source checkout must still import.

## Out of scope here (still open, owned elsewhere)

- What replaces `is_null` / `is_not_null`. The editor offers both today.
- One safe expression language for computed features and branch conditions. The editor
  autocompletes `p.<param>` and `lit/date/datetime/duration/time`, and decider's grammar rejects
  all of them.
- Custom struct output types (`$key` lookups).
