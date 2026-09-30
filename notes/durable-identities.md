# Durable flow and step identities

The `id=` engine work (an optional `id` on every constructor/decorator, threaded
to `Origin.id`) landed with the experiment-05 follow-up. This note records the
identity model around it and the design of the source-rewriting generator.

## The identity model

Two kinds of identity, never one standing in for the other:

- **Derived** — `path` (`term/cap_by_income`) and `source`
  (`credit.rules:cap_by_income`), built from names and import paths. Good for
  discovery and display; not durable, because rename/extract/reorder changes it.
- **Committed** — `id` (`0123abcdef45`), an opaque 12-hex token written into
  source next to the declaration. Durable across rename, extraction and
  reordering because it is independent of name and position.

An id is one per *definition*, many per *path*: a step reused in two flows has
one id and two paths. `(flow id, step id)` is the global reference; the root
flow is the outermost named flow. Anonymous flows are transparent and stay
id-less — which is why `dag(single)` still collapses unless a name or id forces
the wrapper (an id always gets a home).

The id is additive: with `id=None` (the default) name, path, `step_map` and
execution are byte-for-byte unchanged.

## The generator

`decider ids [PATH]` inserts `id="…"` into the step and named-flow declarations
under `PATH` that lack one. It is a deliberate, reviewable source edit, not a
background action, and treats the source as a trust boundary:

- **Clean-tree gate** — refuses to run unless `git status` is empty, so every
  change is a diff against a committed baseline (revertible, bisectable).
- **Parse before write** — every file is `ast.parse`d first; any syntax error
  aborts with no partial edits. Writes are per-file `os.replace` after the whole
  pass succeeds.
- **Additive and idempotent** — existing valid ids are left alone; a second run
  is a no-op. It inserts the minimal `id=` fragment and never reformats.
- **Byte-splice, not an AST round-trip** — `ast` locates the declaration, bytes
  are inserted at the right offset, so comments and formatting are preserved by
  construction. A pretty-printer would churn unrelated bytes.
- **Uniqueness and validation** — a repo-wide scan rejects a duplicate id
  (re-assigns the later copy under `--fix`); ids must match `[0-9a-f]{12}` and
  step names still go through `check_name`. Failures name the file and line and
  state the remediation rather than editing around them.

A multi-line call whose closing paren trails the last argument is normalised to
the paren-on-its-own-line form; the single-line and paren-on-own-line forms are
left exactly as written.

## Configurable assets and parameters

JSON configs (trees, tables, scorecards) already accept an `"id"` field next to
`name` through `ConfigurableStep.id`; it round-trips through dump/load. The
generator edits Python source only — adding `"id"` to JSON documents is left to
a later pass, since config files have no uniform declaration shape to splice and
the durable-reference work that needs them (tasks 04+) has not landed. Params
keep referring to steps by path; only durable references switch to the id.

## Persisted references

A reference persisted for trace/comparison/experiment use captures the id *and*
the descriptive metadata (flow/step names, source location) as they were at
capture time. After a refactor an unresolved id then still renders a partial
result with the old name and location, which a human can repair — the id is the
stable key, the metadata the repairable label.
