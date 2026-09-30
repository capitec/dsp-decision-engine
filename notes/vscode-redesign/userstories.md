# VS Code redesign user stories

This is the working story map for redesigning `vscode-decider`. Add comments
inline using `>>> comment <<<`; unresolved decisions are collected at the end.

## Product framing

The extension has three primary user modes:

1. **Understand a flow**: orient in an unfamiliar project and reach the source
   behind a flow.
2. **Investigate execution**: understand one record or a selected data set by
   running and debugging it.
3. **Explore behaviour**: compare a population across parameters, versions,
   conditions, and reusable experiments.

These modes should have clear entry points and outcomes. In particular,
debugging answers “why did this execution behave this way?” while What-If
answers “how would behaviour differ under this controlled change?” They may
share data and visualisation, but neither should require users to infer which
mode they are in.

## Story 1: understand an unfamiliar flow

### Goal

As a developer newly assigned to a project, I want to discover its pipelines
and understand how their steps, values, and source code fit together, so that I
can safely begin working on the project.

### Starting point

1. I open the **decider** sidebar.
2. It discovers pipelines in the workspace and shows them in a navigable list.
3. I select a pipeline from the sidebar or its source file and open its flow.

### Desired experience

1. The flow opens with an overview that explains execution order and lets me
   expand or collapse nested nodes without losing my current location or zoom.
2. I can use familiar graph interaction: mouse-wheel plus a modifier to zoom,
   drag to pan, and a reliable action to fit the relevant selection or flow.
3. A step clearly exposes its inputs and outputs:
   - Selecting a **step** shows the values it reads and writes.
   - Selecting an **edge** shows the values carried by that dependency,
     including which step wrote each value.
   - The graph does not rely on crowded edge labels to convey this information.
4. Selecting a step from either the Structure view or graph navigates to its
   source. Source and graph selections stay visibly linked.
5. I can ask an agentic tool about the selected flow or steps. The tool can
   inspect a stable representation of the flow and ask the extension to
   highlight the relevant nodes, edges, values, or source.
6. When a flow is paused in a debug session, selecting a step also provides a
   concise execution-oriented value view for that step.

### Current pain points

- Input/output arrows are visually confusing; labels overlap and do not cleanly
  communicate reads and writes.
- Navigation is button-led. Zooming, panning, expansion, and preserving the
  viewport on large flows need to be direct and reliable.
- There is no MCP integration through which an agentic tool can understand or
  control the selected flow.

### Acceptance signals

- A newcomer can identify the execution order, a step's reads, and its writes
  without inspecting dense edge labels.
- Expanding or collapsing a node preserves the user's focal area.
- A selected graph item, Structure item, and source range agree on what is
  selected.
- An agent can retrieve and highlight flow context without screen scraping.

## Story 2: investigate execution on records

### Goal

As a developer investigating behaviour on real input data, I want to load a
data set, locate records, trace the relevant execution, and debug the Python
implementation, so that I can explain or change a specific outcome.

### Starting point

I have JSON, CSV, or Parquet input data. Usually I want to find and trace one
record. For frame-oriented steps, I may need the full frame available to
understand or reproduce behaviour.

### Desired experience

1. I load a supported data source and can see its schema, row count, and any
   loading errors before starting a run.
2. I can search, filter, and select one or more records using stable record
   identifiers and field values.
3. I choose an explicit execution scope:
   - **Selected record** for record-oriented investigation.
   - **Whole frame** when frame semantics must be preserved.
   - Potentially a third mode that traces the selected record while retaining
     the full frame for frame steps, if this can be made semantically clear.
4. The flow shows where the selected record travelled, its values at each
   relevant step, and the data available to the step.
5. I can set, see, disable, and remove breakpoints directly on graph steps as
   well as in source. The graph makes the current pause and breakpoint state
   unmistakable.
6. I can step into Python, inspect variables, use the normal debug console,
   and safely change a value for the current execution. The UI makes the scope
   and lifetime of that override clear.
7. I can reach the implementation of a step quickly and, where normal Python
   debugging permits, edit and rerun it.

### Current pain points

- Existing debug capabilities are strong but hard to discover; breakpoint
  placement is especially unclear.
- Debug-console and Variables changes do not make their downstream effects or
  lifetime sufficiently obvious.
- Single-record and frame-step behaviour needs an explicit model rather than
  an implicit compromise.
- Debugging and What-If currently overlap in ways that obscure their purpose.

### Acceptance signals

- A user can load data, find a target record, and begin a trace without
  learning the debug protocol first.
- A breakpoint set on a graph step is visible and behaves like its source
  counterpart.
- Every live override states whether it applies to this pause, this run, or a
  future rerun.
- Frame-step results are never silently calculated from an insufficient subset
  of the data.

## Story 3: explore behaviour across a population

### Goal

As a developer evaluating a flow's overall behaviour, I want to run and
compare controlled experiments across many records, so that I can understand
distributional effects, regressions, and decision-path changes.

### Starting point

I have a pipeline, input data, and one or more changes to investigate:
parameters, a flow revision, forced conditions or values at selected steps, or
a reusable experiment definition.

### Desired experience

1. I create a named experiment from a baseline data set and flow revision.
2. I define controlled changes:
   - parameter values;
   - input changes;
   - a selected flow revision;
   - scoped conditions or value overrides at a defined step;
   - scenario combinations where appropriate.
3. I run the baseline and variants and can compare:
   - record counts through paths and branches;
   - where executions first diverge;
   - changed, unchanged, and unique values;
   - value-level differences at selected steps;
   - aggregate summaries and suitable visualisations, such as Sankey diagrams.
4. I can filter an aggregate finding to its affected records and open one in
   the execution-investigation mode.
5. I can save, reload, share, and reproduce an experiment with its input
   provenance, flow revision, configuration, and results or result references.
6. A scenario result can launch debugging for a selected record without making
   the exploration mode itself behave like a debugger.

### Design direction

The core experiment model may belong in `decider` rather than VS Code. A
notebook and the extension should be adapters over the same experiment
interface, so results are reproducible and reusable outside the editor. VS
Code should focus on discovery, interactive definition, visualisation, and
drilling from aggregate results into source or a record trace.

### Acceptance signals

- An experiment can be rerun from its saved definition with an unambiguous
  baseline and input provenance.
- A user can move from an aggregate divergence to representative records and
  then to the exact flow/source location.
- Population comparisons remain useful for large data sets rather than
  rendering every record as graph state.

## Candidate stories not yet covered

### Story 4: prepare and validate an experiment

As a developer, I want the tool to validate data compatibility, parameter
constraints, version compatibility, and expected run cost before execution, so
that I can correct mistakes before spending time on a large experiment.

### Story 5: explain and share a finding

As a developer, I want to capture a linkable finding containing its selected
flow context, evidence, experiment configuration, and source revision, so that
another developer can reproduce and review it.

### Story 6: use an agent safely and precisely

As a developer using an agentic tool, I want it to query explicit flow,
execution, and experiment context and request focused highlighting or
navigation, so that its answers are grounded in the project rather than a
visual approximation.

This calls for a small, deep MCP interface rather than a mirror of every UI
action. Candidate high-leverage operations are: discover flows, describe a
selected flow subgraph, inspect a step or value lineage, retrieve a run or
experiment summary, and highlight or reveal a supplied flow selection.

### Story 7: govern and explain a flow

As a developer responsible for a decision flow, I want to see trace evidence,
governed checks, and revision-impact reports in the same source-aware tooling,
so that I can explain a decision and detect unsafe changes before release.

1. I can open an individual decision trace and navigate from a reason, matched
   rule, rounding operation, or value to the responsible flow step and source.
2. I can run or view governed checks that report concrete business paths and
   source locations, rather than only graph-implementation details.
3. I can compare revisions and see the appropriate confidence level:
   structural equivalence where it can be proven, or generated test evidence
   across thresholds and regimes where it cannot.
4. Stable flow/step/configuration identifiers make traces, parameter changes,
   comparisons, and report links survive ordinary source evolution.
5. I can distinguish decision evidence from operational telemetry. Links to
   OpenTelemetry traces may be useful, but they do not replace retained,
   decodable decision evidence.

The core tracing and governance interfaces belong in `decider`; VS Code
visualises their structured results and navigates them to source.

## Open questions for the redesign session

1. **Flow discovery:** What counts as a pipeline in a multi-package workspace,
   and how should ambiguity be presented? >>> I think the custom pipeline finder does the job well. either a function with no arguments that returns a pipeline or a pipeline global variable. I think you can choose the best approaches here <<<
2. **Graph interaction:** Which modifier gestures should be standard for zoom,
   pan, selection, and fit-to-selection? How should the viewport anchor when
   structure changes? >>> Try look for some standards on this? I know the xyflow gestures are quite nice but whatever makes the most sense<<<
3. **Values on edges:** Does an edge represent a control dependency, a data
   dependency, or both? Which value facts need to be visible without selecting
   it? >>> I think both but whatever again you think will be most useful. and without selecting again i think it needs to be made depending on how much information can be presented without being overwhelming. for dependencies i think when you select a node it should only show up like it does now. but maybe you want to go the other way and see for an internal state varaible or a final variable what are all the touch points that interact with it so then it whould show when you select that item rather than a node <<<
4. **Variable views:** Should static reads/writes and runtime values live in
   one step inspector with explicit tabs, or in separate views? >>> Its hard for me to determine what the ebst design would be but ideally whatever makes it easiest and most intuative for the user to naturally know what to do without being overwhelmed by too much info. <<<
5. **Execution scope:** What are the exact semantics for selected-record,
   whole-frame, and mixed record/frame execution? Which steps require a full
   frame, and must the tool refuse an invalid scope? >>> its only really the frame steps because those take in a while frame and might do stuff like col("test") / col("test").sum() then i think the whole frame makes more sense but for frame steps i think we need a way for the user to shoose the right execution mode for what they are trying to debug. <<<
6. **Input data:** Which JSON, CSV, and Parquet shapes are supported? How are
   schemas, IDs, missing values, and large files handled? >>> data is generally dumped from redshift so ideally support that to the best of your abilities. i think the framework already handles missing values etc we can always raise an error if the value isnt handled it would be good to raise that to the client to say value isnt provided and no defaults exist. <<<
7. **Overrides:** Which changes are valid in a live debug session, and which
   belong only to an experiment? How are their scope, persistence, and
   reproducibility communicated? >>> In debug it would be nice to be able to set parameters and edit the state values mid run similar to python debugging. for whatif its more the situation that you will kick off again in a debugger view liek reproduce the whatif exactly on one record and be able to run it same parameters, git commit and mid-flow forced overrides. <<<
8. **Breakpoints:** Is a graph breakpoint a source breakpoint, a logical flow
   breakpoint, or both? How should one map when a step has several source
   locations? >>> all breakpoint types. for several source locations i havent had an issue with the current mapping process not sure what that uses. maybe we only map when the source isnt vague like a step. the loop will be where that loop is defined hopefully or at least where the logic for the loop is defined. maybe it will be like the press on the box that is the condition will take you to the condition and the code will take you to the steps code no need for the loop itself to be a navigatable <<<
9. **Experiment persistence:** Should definitions live in versioned project
   files, a user workspace store, or both? What belongs under source control?
   >>> It would be nice to be perscriptive and have an experiments/ folder with like an experiment slug or name and maybe allow a readme or overview section the user can write to allow them to state the purpuse and use metadata like the date etc. then there can be assets like jsons for the graphs that must be presetn. either that or it can be a folder with just jsons or yamls describing experiments and all the graphs. Im kinda more leaning to yaml files one per experiment. <<<
10. **Experiment core:** What is the smallest `decider` experiment interface
    that both notebooks and VS Code can use? Which computation must remain in
    the core to ensure consistent results? >>> Im not sure on this maybe its best done through experementation and reserch with a subagent can look at some example projects and try to define an experimentation api. a way to define whatif scenarios (which parameters to set, which parameter sweeps to set, which git commit tags to compare (allowing stuff like HEAD and HEAD^-1 or something to compare head with the previous version.)) and what plots are needed at which steps. and maybe once you have defined the experiment the interface can be like results = exp.run(data, params={"prev_release_tag": prev_tag}); results.save("results/{exp_name}/{run_date}"); results["result_name"].interact() <- allows filtering and playing around with the specific result maybe using something like plotly and interactive controls.; restuls.report <- get some summary statistics maybe as a pd.dataframe. We could even add a set of data driven tests and have results.tests.all_pass() or results.test["test_name"] ... ideally driven py pytest or another framework under the hood. With that in mind it might require that the structure is experiments/{experiment_name}/index.yaml and then we can have experiments/{experiment_name}/tests/....py and experiments/{experiment_name}/graphs/....py to help with non-standard things <<<
11. **Scale:** What are the target data sizes, acceptable run times, and
    aggregation requirements? Which charts must be native versus supplied by
    an extension point?
    >>> It would be nice to cater to millions/billions of records depending on the graphs set up. 
12. **MCP control:** Which capabilities may an agent invoke automatically,
    which require confirmation, and what data must never be exposed through
    the MCP server?
    >>> Everyhting that doesnt cause persistance or run code i think can be done automatically. The team has adopted fastmcp so i would prefer any mcp tooling make use of taht. i think use your jusgement on that and we can always adjust the permissions post op <<<
13. **Decision traces:** What compact trace schema, retention model,
    redaction/deletion model, and trace-point-conservation check are required?
    How does an OpenTelemetry exporter map onto that schema without becoming
    the sole evidence store?
14. **Governance:** Which governed flow classifications select which
    centrally-owned checks? What report shape is shared by the Python API, CI,
    notebooks, and VS Code?
15. **Stable identifiers:** Which entities require durable IDs, and how are
    generated, user-specified, and derived identities reconciled across
    refactors and merges?

>>>
Another change that i would like is currently the debugger package lives tools/decider-bridge. its not really big so i think it would be best to relocate it to something like decider.debug_bridge or decider.ddb for decider debugger playing on pythons pdb.
<<<

## Review of inline comments

### Decisions now made

1. **Pipeline discovery:** retain the custom finder as the discovery mechanism.
   It should recognise a module-level pipeline value and a zero-argument
   factory that returns a pipeline. The finder should report ambiguous or
   invalid candidates with a useful reason, rather than silently guessing.
2. **Graph information density:** use progressive disclosure. A selected step
   highlights its immediate dependencies as it does today, while its inspector
   lists reads and writes. Selecting a value provides the inverse view: every
   step that reads, writes, or carries that value. Edge labels remain minimal
   and must not be the only way to understand a dependency.
3. **Variable inspection:** use one contextual inspector, not competing
   permanent views. Its default is a small overview; users can reveal static
   structure, lineage, and—when a run is available—runtime values without
   losing their selection.
4. **Frame execution:** selected-record and whole-frame execution are explicit
   choices. A frame step must receive the whole frame, since its calculation
   may depend on aggregates such as `col("test") / col("test").sum()`. The
   redesign should make the choice and its effect visible before execution.
5. **Input data:** prioritise the JSON, CSV, and Parquet forms produced by
   Redshift workflows. Validation should surface absent required values where
   no default exists. The framework's existing missing-value semantics remain
   authoritative; the extension reports incompatibility rather than silently
   inventing a value.
6. **Debugging versus What-If:** a debug session permits live parameter and
   state edits in the style of normal Python debugging. A What-If run is a
   reproducible scenario. A selected scenario must be launchable as a debugger
   run with its exact parameters, Git revision, input selection, and mid-flow
   forced overrides.
7. **Breakpoints and navigation:** expose every existing breakpoint type from
   the graph as well as source. Continue using the current source mapping where
   it is precise. Condition nodes navigate to condition code and step nodes to
   step code; loop containers do not need independent source navigation when
   their defining logic is already reachable.
8. **Experiment persistence:** experiments are project-owned, prescriptive
   assets under `experiments/<experiment-slug>/`. The default definition is a
   versioned YAML file, with optional human-authored overview, Python tests,
   and custom graphs. Generated run results remain separate from the
   definition and can be ignored or retained according to project policy.
9. **MCP:** use FastMCP. Read-only operations that neither run code nor persist
   data may be automatic. Running a flow, starting a debugger, writing files,
   or other persistent actions require explicit confirmation.
10. **Debugger packaging:** relocate `tools/decider-bridge` into the main
    package as `decider.debug_bridge`. This name is clearer and more
    discoverable than an abbreviation. The redesign must preserve its launch
    protocol, packaging, extension configuration, and debugger tests through
    the relocation.
11. **Tracing and governance:** implement compact decision tracing and
    governance reports in `decider`, not the extension. `decider` emits
    structured trace data but does not retain it; a client-selected
    post-record adapter handles OpenTelemetry or another sink, transformation,
    privacy policy, and retention. VS Code consumes trace and report results
    to provide source-aware explanations, path highlighting, and revision
    diagnostics.
12. **Durable IDs:** use deterministic derived IDs for initial discovery only.
    Provide a command to generate explicit IDs that are committed for
    production flows. It runs only when affected source is tracked and clean,
    and must preserve code readability.
13. **Checks:** provide generic diagnostics and check suites through a Python
    interface, for example
    `decider.check.run(pipeline, suites=[decider.checks.default_suite])`.
    Clients choose how to incorporate results into CI and may run the same
    suites from VS Code. Custom client checks are an extension point, not a
    prerequisite for the first release.
14. **Experiment execution and storage:** optimise local execution with
    Polars, without requiring users to provision remote compute. The practical
    record limit is determined by local resources; larger data sets require
    client sampling. `decider` saves results to a caller-chosen directory and
    does not own long-term storage; clients may use Git LFS or another policy.
15. **Scenario overrides:** constrain reproducible What-If overrides to
    declared override points or step outputs. This keeps scenarios powerful
    while avoiding arbitrary impossible internal states.
16. **MCP access:** make read capabilities as broad as practical so an agent
    can perform the workflow end-to-end and the UI can focus on visualisation.
    Avoid unnecessary per-read confirmation; retain confirmation for running
    code and persistence.

### Further answers to open questions

- **Trace privacy and retention:** emitted trace events should be rich by
  default. The client owns any filtering, redaction, deletion, retention, and
  delivery policy through a post-record adapter. The default adapter passes
  events through.
- **Governance scope:** start with general quality and correctness diagnostics:
  wall-clock reads, non-pure internal-state usage, potential numeric
  overflow/underflow, and floating-point sensitivity. Do not require
  regulated-domain classifications or centrally controlled policies.
- **Governance execution:** the Python check result is the shared mechanism;
  clients decide whether to call it in CI. VS Code can invoke the same check
  suites and render their reports.
- **Results:** experiment definitions are versioned project assets. Results
  are written to a caller-selected directory, outside `decider`'s ownership.
- **MCP data:** agent read access should include the contextual record and
  trace data needed to complete a workflow, subject to the client's chosen
  trace adapter and data-access environment.

### Recommendations

- **Graph gestures:** adopt conventional canvas behaviour: wheel zooms around
  the pointer, drag on empty canvas pans, click selects, and a visible
  fit-to-flow/fit-to-selection control recentres deliberately. Expansion and
  collapse should retain the clicked node's screen position where possible,
  preserving both viewport and selection. Before implementation, confirm the
  interaction contract supported by the selected graph library rather than
  building custom gestures.
- **Experiment layout:** start with this small, inspectable project shape:

  ```text
  experiments/
    <experiment-slug>/
      experiment.yaml
      README.md                 # optional purpose and interpretation
      tests/                    # optional data-driven checks
      graphs/                   # optional custom Python visualisations
  ```

  `experiment.yaml` should name the flow, baseline revision, input data,
  scenario definitions, expected outputs, and requested built-in summaries.
  It should refer to custom tests/graphs rather than embed arbitrary code.
- **Scale:** support for millions or billions of records means experiments must
  produce aggregate and sampled/drill-down data at the computation layer.
  Neither the extension nor MCP should attempt to materialise per-record graph
  state. Required aggregates, sampling rules, storage location, and execution
  backend remain open design work.

### Work that needs a dedicated design/research pass

The experiment interface should be explored before committing to an API. It
needs to cover declarative scenarios, parameter sweeps, relative Git revisions
such as `HEAD^`, step-level forced overrides, reusable plots, interactive
result filtering, summaries, persisted results, and data-driven tests. A
future investigation should compare established experiment/evaluation projects
and then design at least two candidate `decider` interfaces for depth,
locality, notebook usability, VS Code integration, and pytest compatibility.

The illustrative shape from the comments is a useful target to evaluate:

```python
results = experiment.run(data, params={"previous_release_tag": previous_tag})
results.save("results/<experiment-name>/<run-date>")
results["result-name"].interact()
report = results.report
assert results.tests.all_pass()
```

It is not yet a proposed public interface: persistence, lazy versus eager
evaluation, data scale, result ownership, and how YAML connects to custom
Python graphs/tests must be decided first.

## Initial assessment

Your separation is sound: orientation, execution investigation, and
population-level experimentation have different goals, data scales, and
interaction models. The principal redesign risk is trying to make a single
graph/debug panel satisfy all three. The highest-leverage direction is to keep
one shared flow model and source-navigation model, while giving each user mode
its own clear starting point, state, and result view.

The experiment capability is the strongest candidate for a deep module in
`decider`: a small interface for defining, running, persisting, and querying
experiments can hide execution, comparison, aggregation, and caching. The VS
Code extension and notebook tooling would then be adapters at that seam rather
than independently implementing experiment semantics.
