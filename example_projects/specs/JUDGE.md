# Judge brief: one project, two implementations

You judge one example project, implemented independently by a Haiku agent
(`example_projects/<NN-name>/haiku/`) and a Sonnet agent
(`example_projects/<NN-name>/sonnet/`) with the `decider` framework. Be
evidence-based: every claim cites a file (and line where useful) or a command
you ran. Never trust an implementation's own NOTES.md or report without
checking the code. Do not modify either implementation or anything under
`decider/`; write only your judgement file and scratch files.

Context to read first: `example_projects/specs/BRIEF.md` (what implementers
were asked to do), the project's spec and its slice in `SCOPE.md`, `DEPS.md`.
Protocol caveat: Haiku projects 00, 01, 02, 03 and 10 were briefed with the
plain BRIEF only; every other project (and every Sonnet project) was also told
explicitly that verifying `decider build` is mandatory and which built-ins to
look at. Mention it where it matters.

## Assess

1. **Servable unchanged.** For each implementation, follow its `SERVE.md`
   exactly as written (paths rewritten only from a scratchpad location to its
   repo location, if SERVE.md points at the scratchpad — say so). Use
   `uv run --project <repo>` and never `--python`. Run `decider build`, score
   `sample_request.json` through the handler as SERVE.md describes (or via
   `decider.serving.handler.construct_handler_from_settings()` → `stage()` →
   `activate()` → `live.executable.score(record)`), and run its tests.
   Record PASS/FAIL per check with the exact error. No edits allowed.
2. **Similarity.** Did the two reach similar solutions? Compare architecture
   (how the slice is decomposed into steps/configs), key design decisions (e.g.
   how rules, tables, nesting, search, overlays, evidence were done), scope
   covered, and behaviour on the spec's own worked examples where both can run
   them. Note where they diverge and which is closer to the spec.
3. **Maintainability.** Read the code as a new engineer would. Then read the
   code-only summaries a low-effort Haiku agent wrote without the spec or
   notes: `example_projects/evaluation/summaries/<NN>-haiku.md` and
   `<NN>-sonnet.md`. Score each summary's accuracy against the code and the
   spec (1–5), and list what the summariser got wrong or couldn't work out —
   that is a proxy for how understandable the code is with little context.
4. **Reuse.** For each implementation: what it reused from its own model's
   earlier projects (named in DEPS.md; verify real calls, not just imports or
   stubs), which `decider` built-ins it used (TreeConfig, DecisionTableConfig,
   ScorecardConfig, param/missing_as, flow/dag, branch, loop, frame_step,
   sessions, config store, testing helpers) versus what it wrote from scratch
   that a built-in covers, and use of the standard library versus hand-rolled
   code. Verify quantitative claims (e.g. "67% of decision points reused").
5. **Struggles.** What each agent struggled with, from its NOTES.md and the
   code (workarounds, stubs, private-API use, monkeypatches), classified as
   framework problem / spec problem / agent capability.

## Output

Write `example_projects/evaluation/judgements/<NN>.md` starting with this
exact block (fill in values), then a concise section per assessment item:

```yaml
project: "<NN-name>"
haiku:
  build: pass|fail
  sample_scores: pass|fail
  tests: "<passed>/<total>"
  summary_accuracy: 1-5
  reuse_of_earlier_projects: none|stubbed|partial|real
  builtins_used: [list]
  spec_slice_coverage: 1-5
  code_quality: 1-5
sonnet:
  build: pass|fail
  sample_scores: pass|fail
  tests: "<passed>/<total>"
  summary_accuracy: 1-5
  reuse_of_earlier_projects: none|stubbed|partial|real
  builtins_used: [list]
  spec_slice_coverage: 1-5
  code_quality: 1-5
similarity: 1-5   # 5 = essentially the same design
closer_to_spec: haiku|sonnet|tie
```

Keep the file under ~2 pages. Report back the YAML block and three sentences
on the most important finding.
