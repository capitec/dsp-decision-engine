# Progress log

One entry per merged task: date, task, model, commit, outcome, tests, follow-ups.

## 2026-09-23

- **T0.1** (opus) `1bd587b`: `CLAUDE.md` conventions plus `tests/test_conventions.py` (500-line limit with allowlist; bans `doc NN`, `§`, `EXPERIMENTS`). 2 passed. Follow-up: `CLAUDE.md` points at `Design.md` §8 and `IR.md` §9 for the layout; repoint when those docs are removed (T7.1). Worktree came off `main` and was fast-forwarded to `feature/decider-v2`; briefs now tell agents to check their base.
- **T0.2** (opus) `a4bc8aa`, merge `e950786`: `import decider, decider.serving` works; the handler's `module_fn` and `DeciderConfigSettings.get()` raise `NotImplementedError` until serving and the config store are rebuilt (T6.1/T6.2); `get_default_executor` removed; legacy tests moved to `tests/_legacy/` and ignored. 3 passed. Follow-up: `serving/servers/{starlette,sanic}.py` still import `decider.initialization` at server startup (T6.2).
- **T0.4** (opus) `b046335`: 10 decision notes plus `notes/README.md` index. Findings that need a decision or care later: fusion default (Design says one kernel per sequence, evidence says fusing many heavy steps loses; decide before T3.2/T3.3); the fallback also has to catch `UnsupportedBytecodeError` (not a `NumbaError`); the compile content key must cover constants, not just `co_code` (T3.2); decider2 shipped `nogil` opt-in, the new rule is unconditional for serving.

**Phase 0 complete.** `uv run pytest`: 3 passed. Conventions test green.
