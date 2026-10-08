---
name: decider
description: Decider projects, decision pipelines built from Python functions plus versioned JSON config documents. Covers setting up a project, pipelines and flow files, decision trees, rule sets, decision tables, scorecards, params documents, checks, experiments, and the decider VS Code extension. Use when a repo imports decider or has configs/<version>/, or the task names any of these.
---

# decider

decider builds decision pipelines (credit, pricing, fraud rules, campaigns). A
pipeline runs over a polars frame or one record, interpreted or compiled with
numba, and serves over HTTP with `decider serve`. Every compiled mode gives the
same answers as plain Python.

## Mental model

- A **step** is a function. Its arguments are the values it reads; its name is
  the value it writes.
- **Inputs** are plain arguments: request data (amounts, dates, ids).
- **Params** are `param(default)` arguments: policy that is the same for every
  request (limits, cut-offs, weights). They live in the params document
  `configs/<version>/params.json`, retuned with no code change. Request data is
  an input, never a `param()`.
- **Config documents** are JSON in `configs/<version>/`: decision trees, rule
  sets, decision tables, scorecards. A project's `build()` receives each as a
  step; any number in one can be `{"param": "name", "default": x}`.
- **Pipelines** compose steps in Python (`flow`, `dag`, `branch`, `loop`) or in
  a YAML flow file.
- A config **version** is immutable; a decision is replayed against the version
  that made it.

## Orient

1. Run `uv run decider guide`. Its table "Which construct for which need" picks
   the construct for a task; its examples run. `uv run decider --help` lists
   every command, and the docstrings (`help(decider)`) document the API. The
   repo's `docs/` folder describes an older version.
2. Find the project: a `.env` with `DECIDER_API__PIPELINE`, a `pipeline.py` with
   `build()`, `configs/`.
3. Check the MCP tools are loaded by calling `list_configs`. Claude Code names
   them `mcp__plugin_decider_decider__<tool>`; this skill uses the bare name,
   and a deferred tool loads through ToolSearch. When the call fails or the
   tools are missing, follow [setup-project.md](references/setup-project.md#the-mcp-server).

## Golden rules

- **Edit config documents through the MCP editing tools.** Each validates the
  document before it writes and leaves the file untouched when it would not
  build. Python and flow files are edited directly.
- **Check every change with an exact worked answer**: a record whose outcome
  you computed by hand, asserted in a test.
- **Run `decider build`, then score `sample_request.json`, before saying it
  works.**
- **Put policy numbers in params or config documents**, calculations in
  Python, request data in inputs.
- **Ask the user before running code on real data**: `run_flow`,
  `run_experiment`, `start_debug` and `generate_ids` execute their code or write
  source, and `raw_record` / `raw_trace` return per-record values.

## References

Read the one that matches the task before starting it.

| Read | When |
|---|---|
| [setup-project.md](references/setup-project.md) | new project, project layout, `.env`, config versions, running and serving, installing or fixing the MCP server |
| [best-practices.md](references/best-practices.md) | deciding what belongs in params, config documents or code; versioning; replay; durable ids; reviewing a change |
| [pipelines.md](references/pipelines.md) | reading or changing a Python pipeline or YAML flow file; finding where a value is written and read |
| [testing-and-checks.md](references/testing-and-checks.md) | tests, `check_report`, run summaries, experiments, debugging one record |
| [configs.md](references/configs.md) | any config document or params document: listing, reading, validating, patching, trying rows |
| [decision-tables.md](references/decision-tables.md) | decision tables: rows, columns, outputs |
| [scorecards.md](references/scorecards.md) | scorecards: variables and bins |
| [decision-trees.md](references/decision-trees.md) | decision trees and rule sets: nodes, branches, outputs |
| [vscode.md](references/vscode.md) | what the user has selected in VS Code, revealing code, debugging in the editor |
