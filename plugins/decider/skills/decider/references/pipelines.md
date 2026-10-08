# Pipelines: Python and flow files

Use this to understand or change a pipeline: what steps it has, where a value is
written and read, and which file to edit. The MCP tools read structure; you
edit the Python or YAML yourself, because no tool writes either.

## Two ways to write a pipeline

- **Python**: plain functions composed with `flow`, `dag`, `branch`, `loop`,
  `each`, `optimise`; `build()` returns the pipeline.
- **Flow file**: `flow.yaml` (or `.json`) holds the structure, `components`
  holds the functions. Serve one with `DECIDER_API__PIPELINE=flow.yaml`, or load
  it from `build()`:

```python
from pathlib import Path
from decider import load_flow
from flowapp import components

def build():
    return load_flow(Path(__file__).with_name("flow.yaml"), components=components)
```

```yaml
flow: lending
components: flowapp.components
steps:
  - step: ratio
  - step: affordable
  - branch: by_affordability
    when: {expr: "affordable"}
    modifies: [offer]
    then: [{step: full_offer}]
    else: [{step: no_offer}]
```

Flow-file rules: `modifies` on a `branch` and `carries` on a `loop` are required,
as in Python; an omitted `else` passes false rows through; an `expr` condition
is false where an input it reads is null; `dag:`, `each:` and `optimise:` nodes
take the Python constructors' arguments. `uv run decider guide`, section "A
pipeline from a flow file", has the rest.

## Read the structure first

An `entry` is `module:factory` (`flowapp.pipeline:build`). The tools build it
the way `decider serve` does: a `build(limit_band)` that takes config documents
gets the latest `configs/<version>/` documents, and runs use that version's
`params.json`. The module is imported fresh on every call, so your edits count.

1. `discover_flows(root=".")` lists the pipelines under `root`: `entry`, `file`,
   `line`, `kind`. It imports every candidate file, so top-level code in a
   script runs. A file that fails to import comes back with its `error`.
2. `describe_flow(entry)` returns `nodes` (each with `path`, `node_type`,
   `source`, `inputs`, `outputs`), `edges` and `value_slots`. Pass
   `subgraph="lending/by_affordability"` for one subtree.
3. `inspect_step(entry, path)` gives one node and the edges into and out of it.
4. `lineage(entry, name)` lists every writer and reader of a value.
5. `source_context(entry, path)` returns the file, line and a window of source.

For the flow file above, `describe_flow` lists these node paths:

```
lending                                   sequence
lending/ratio                             call   flowapp.components:ratio
lending/affordable                        call   flowapp.components:affordable
lending/by_affordability                  branch
lending/by_affordability/by_affordability_when   call   (the when expression)
lending/by_affordability/full_offer       call   flowapp.components:full_offer
lending/by_affordability/no_offer         call   flowapp.components:no_offer
```

`lineage(entry="flowapp.pipeline:build", name="offer")` answers
`writers: [".../full_offer", ".../no_offer", "lending/by_affordability"]` and
`readers: []`: the branch writes `offer` through its arms, and nothing reads it.

`source_context` on a function step returns its file and line. A composite
(`branch`, `flow`, `loop`) has no source of its own (`file: null`): find it in
the flow file by its name, `by_affordability`, or in the Python call that
builds it. `source` names the function as `module:name`.

## Change a pipeline

1. Read the structure (above) and find the file with `source_context` or by
   searching for the step name.
2. Edit the Python or YAML directly. Keep a step to one output; a branch's arms
   write the same names its `modifies` lists.
3. Re-read with `describe_flow` and check the wiring: the new step's `inputs`
   are written upstream (`lineage`), and its `outputs` are read where intended.
4. Prove the answers with a worked example (testing-and-checks.md).
5. Run `uv run decider check <entry>` and read every new finding.

Worked edit: to offer only below a 0.3 ratio, change the branch's `when` from
`affordable` to `affordable and ratio < 0.3`. On three loans with ratios 0.25,
0.35 and 0.8, `run_summary` goes from `offer` 3000.0 for two rows and 0.0 for one
to 3000.0 for one row and 0.0 for two.

## Pitfalls

- Tools run a pipeline with the latest version's `params.json` (the defaults in
  code when there is none). To try other values, change them with `params_set`
  or score through `pipeline.compile().score(record, params)` in a test.
- `discover_flows` reads `.py` files only and skips `.venv`, `node_modules` and
  directories starting with `.`. A flow file appears as the `build()` that
  loads it.
- Without durable ids, `flow_id` and `step_id` are `null` and a step is
  identified by `path`. Add ids with `generate_ids` (best-practices.md).
- A step reading `str`, `date` or a list of dicts runs in Python and says so
  with a `FallbackWarning`. That is a speed note, not an error
  (`uv run decider guide`, "What compiles, and what falls back").

## Tools

- `discover_flows(root=".")`: the pipelines under a directory, as `{name, entry, file, line, kind}`.
- `describe_flow(entry, subgraph=None)`: nodes, edges and value slots of a flow, or of one subtree.
- `inspect_step(entry, path)`: one node with its source location and the edges touching it.
- `lineage(entry, name)`: every slot, writer and reader of a value.
- `source_context(entry, path)`: file, line and a window of source around a step.
