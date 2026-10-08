# Can screenshot-scored user stories steer the editor UI?

Date: 2026-09-23 to 2026-09-24 · Script: spike, not kept (Playwright `_electron` stories against VS Code, removed with the old extension)

## Purpose
Find usability defects an assertion cannot see, and learn how far a judge model scoring screenshots can be trusted as a stop signal.

## Method
Stories run the real editor (headless, `--ozone-platform=headless`) and caption each screenshot with what the user just did. A fresh judge with no other context scores each story 1 to 10 from the screenshots alone (a story whose goal is not visibly met caps at 5), lists the worst problems, and the UI is fixed between rounds. Two loops: steering and tracing a run (5 stories, 1,000-step bank flow plus a loop example) and a large project (7 stories, about 1,400 params, 14 lookup tables).

## Result
| loop | rounds | first | best | stop |
|---|---|---|---|---|
| steering and tracing | 8 | 6.4 | 7.4 (round 4) | three rounds without a new best |
| large project | 22 | 3.3 | 8.3 (round 20; second judge 7.9) | above 8, confirmed by two judges |

- Judges differ by about 0.3 to 0.4 on the same screenshots, so a score that crosses a threshold needs a second judge.
- Fixes it drove: a find box and fold-by-default for 1,000 steps; formula trees filled with the record's values; "go back" and "will re-run" history; money and percent formatting everywhere; branch controls shown only on the condition step. Layout bugs found only in images: flex children that would not shrink, data edges that made the layout sprawl.
- Bugs found by the large flow: stale bytecode after a same-second, same-size edit (now a fresh `sys.pycache_prefix` per load) and a custom DAP request with undefined `args`.
- Launch traps: use the Electron binary, not the `codium` wrapper; on macOS not the `code` CLI, with a short absolute `--user-data-dir`; point `decider.python` at the worktree (`["uv", "run", "python"]`); webview content is two iframes deep; a copied, not symlinked, `ms-python.debugpy`.

## Decision
Keep stories that read as a person would, judged by a fresh agent per round, stopping after three rounds without a new best, and confirm a high score with a second judge. Only the unit and in-editor smoke tests remain in the repo. See [debugger and editors](../decisions/debugger-and-editors.md).
