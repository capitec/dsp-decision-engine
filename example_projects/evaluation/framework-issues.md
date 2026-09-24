# Framework issues reported by the example-project implementers

Triage of every framework finding in `notes/progress.md` ("Example-project experiment") and in the "Framework friction" section of each `example_projects/*/*/NOTES.md`. Twelve Haiku projects and ten Sonnet projects were covered. Sonnet 09 and 11 were still being built and had no NOTES.md yet. Each issue was reproduced against `decider` on `feature/decider-v2` with a script of at most 20 lines, run with `uv run --project <repo> python`. The scripts are in `/tmp/claude-1000/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/8b7ba2ad-c1f7-41bd-bdc5-974c7120b22c/scratchpad/framework-issues/` (`r_*.py`, `r_codepath.sh`). No file under `decider/` or `tests/` was changed.

Project codes: `00s` means project 00, Sonnet; `06h` means project 06, Haiku.

Classes:
- **BUG**: confirmed bug.
- **MISLEADING**: the user made a mistake, but the error message didn't help.
- **DESIGN**: works as designed, but the design doesn't fit the use case.
- **DOCS**: a gap in the documentation.
- **NOT REPRO**: not reproduced.
- **USER**: user error.

## What actually stopped the Haiku builds

The progress notes say 6 of the 12 Haiku projects stopped at a "params-document path error". Re-running each Haiku build shows that the params code is almost never the cause:

| Haiku project | Reported | Actual cause |
|---|---|---|
| 02, 05, 07 | `no step with params at '<name>'` | Their SERVE.md puts 00's directory, which has its own `pipeline.py`, ahead of their own directory on `PYTHONPATH`. `_warm` doesn't move `code_path` to the front, so **project 00's pipeline was built** against their params document (F2). With the path fixed, 02 then fails in warm-up (F1). 05 and 07 then fail on `from decider import RequestHandler` in `inference.py` (F32). |
| 06 | `'float' object is not iterable` in params loading | Wrong `pipeline.py` as above (F2). Once fixed, the warm-up fed `1.0` to a `dict` input (F1). |
| 08 | `param(ge=, le=)` → `float() argument ... not 'dict'` | Not related to constraints: the build fails the same way with none. A step annotated `-> dict` runs as a Python fallback in fused mode, and the fallback writes into a float64 buffer (F4). Deleting the constraints didn't fix it: the build still fails today. |
| 09 | `no step with params at 'operation'` | User error: `params.json` is flat. The message gives no list of valid keys (F12). |
| 00 | (reported OK) | SERVE.md names `pipeline:build`, but `params.json` belongs to `pipeline_simple.py` (F12, user error). |
| 04 | "strict type checking on string literals" | The compiled-mode `str` rule (F5). |
| 03, 10 | "decider CLI missing" / not run | The CLI works via `uv run --project` (F23). `inference.py` imports a name `decider` doesn't export (F32). |

All ten Sonnet projects monkeypatch `decider.serving.handler._warm` in `inference.py` (F1). All ten set `DECIDER_API__MODE=interpreted` (F5, with F4 waiting behind it).

## Summary

Sorted by severity. "Shared root" means several rows share one fix.

| # | Issue | Class | Severity | Projects hit | Fix (root cause) |
|---|---|---|---|---|---|
| F1 | Serving warm-up feeds `1.0` to `date`/`list`/`dict` inputs, so `decider build` fails | BUG | blocker | all 10 Sonnet (monkeypatched), 02h, 06h | `serving/handler.py:198-206`: add `date`/`datetime` dummies; warm from `<code_path>/sample_request.json` when present; turn a user-code exception during warm-up into a warning |
| F2 | `code_path` isn't moved to `sys.path[0]` when `PYTHONPATH` already lists it, so the wrong project's `pipeline.py` is **silently built and served** | BUG | blocker | 02h, 05h, 07h, 06h (the "params" blockers), 08s, 10s; 05s, 06s, 07s worked around it | `serving/handler.py:213-215`: `if code_path in sys.path: sys.path.remove(code_path)` before the insert; print the pipeline module's `__file__` in `decider build` |
| F3 | Nested/temporal columns round-trip through `Series.to_numpy()` (shared root of F3a–F3g) | BUG | blocker | 00s, 01s, 02s, 05s, 06s, 07s, 10s | `engine/run/state.py:187-196` `from_series`: keep List/Struct/Array/temporal dtypes as Python objects (`s.to_list()` into an object array). Tested: fixes F3a–F3g; the suite still passes (1510 passed, 1 xfailed) |
| F3a | `list[struct]` frame-step output crashes materialisation (`cannot parse numpy data type dtype('O')`) | BUG | high | 00s, 02s, 10s | F3 |
| F3b | `list[struct]` from one frame step into a second one crashes | BUG | high | 02s | F3 |
| F3c | `list[list]` frame-step output comes back as `Array(Object)` of numpy arrays | BUG | high (silent) | 05s | F3 |
| F3d | `list[date]` becomes `Array(Int64)` epoch days between frame steps | BUG | high (silent) | 05s | F3 |
| F3e | `list[str]` frame output with empty and non-empty rows → `PanicException` in `run()` | BUG | high | 01s | F3 |
| F3f | `list[struct]` input read by a plain step on the frame path becomes a float ndarray (dates → epoch floats, keys lost). 06s's exact numpy dtype panic was not reproduced; this is the same code path, silently corrupting | BUG | high (silent) | 06s | F3 |
| F3g | A `list` input is an `ndarray` under `run()` but a `list` under `score()`, so `x or []` raises | BUG | medium | 00s, 05s | F3 |
| F4 | Compiled modes: a Python-fallback step returning `dict`/`list`/`date` writes into a float64 buffer (`float() argument ... dict`) | BUG | high | 08h (misattributed to `ge=/le=`); latent in every Sonnet project that leaves interpreted mode | `engine/compile/units.py:98-101`: object dtype for any non-numeric, non-Literal output in `Fallback.writes`. Tested |
| F5 | Compiled modes refuse a step reading two `str` inputs, a `str` literal, or a `str` input without a `str` param | DESIGN | high | 00s, 01s, 07s (all 10 Sonnet serve interpreted, ~294 rows/s in 07s), 04h, 09h | `engine/run/runners/stepped.py:179-204`: rather than raise, run that one call through the existing `Fallback` unit (the message becomes its `reason`) |
| F6 | Single record with a frame step anywhere: `[]` or a lone `None` infers `Null`/`List(Null)` → `ArrowImportError` | BUG | high | 00s, 02s, 06s, 07s, 08s | `engine/run/engine.py:115-117` (`prepare`, via `score` at :169): drop `pl.Null` columns and cast `List(Null)` to the declared list dtype. Tested |
| F7 | `missing_as([])` on an absent list column fills zero-width, giving `IndexError` (score) or `ShapeError` (frame step) | BUG | high | 00s, 02s, 07s | `runners/interpreted.py:197` and `runners/stepped.py:97`: `np.where(valid, values, [])` broadcasts to shape (n, 0); fill list/dict/tuple row by row into an object array. Tested |
| F8 | `list[float]` outputs are materialised with an inferred dtype: `[[1, 2], [1.5]]` → `[[1, 2], [1]]` (**silent truncation**); `[9, 1.5]` → `TypeError` | BUG | high (silent) | 03s | `engine/run/state.py:180-182` `_series`: pass the polars dtype of the declared output annotation |
| F9 | `missing_as(False)`/`param(False)` are truthy when the function is called directly | BUG | high (silent) | 07s, 08s | `engine/params/declare.py:26-40`: `bool` isn't a carrier type; add `__bool__` to `MissingAs` (fill) and `ParamSpec` (default). Tested |
| F10 | An input read as `float` by one step and `int` by another: the first reader wins, and the `int` reader silently gets a float | BUG | high (silent) | 10s | `engine/wiring/resolve.py:145-147` (the "first reader's declaration" ponytail): raise `WiringError` on conflicting base annotations, or cast per reader in `_argument` |
| F11 | Chained `.relabel(writes=)` keyed by the new name is a silent no-op | BUG | high (silent) | 08s | `steps/base.py:75-78`: compose through the existing mapping, and raise on a key that is neither an original nor a current name |
| F12 | Params-document errors: no list of valid paths, and an empty `{}` for a step without params is rejected | MISLEADING | medium | 09h, 00h, 02h, 05h, 06h, 07h | `engine/run/params.py:87-99`: when `hint()` is empty, append "valid keys here: [...]"; accept an empty mapping at a path with no params |
| F13 | `frame_step` can't take `param()`: its default is used silently, and a params entry errors | DESIGN | medium | 01s, 03s, 06s | `runners/interpreted.py:90-91`, `steps/frame.py:30-33`: harvest params after `df` in the signature and pass the bundle as kwargs in `_frame` |
| F14 | `.relabel()` refused on frame steps | DESIGN | medium | 03s | `engine/ir/context.py:270`: support it by renaming reads before `fn(df)` and writes after, in `_frame` |
| F15 | Typo guard hard-fails a legitimate new input that is ≥0.8 similar to an earlier output | DESIGN | medium | 05s, 07s | `engine/wiring/resolve.py:136-143` (`TYPO_CUTOFF` at `registry/resolve.py:8`): only guess a typo within ≤2 edits of the same length, or accept the input when the reader declares its type |
| F16 | `TreeConfig.path_output` gives only the leaf id, not the path | DESIGN | medium | 04s (and 04h wrote its own tree engine) | `steps/trees/__init__.py:36-38`: add a `trace_output` (`list[str]` of visited node ids); both walkers already visit them |
| F17 | Built-in `ConfigurableStep` tags (`tree`, `decision_table`) resolve only after their module is imported | BUG | medium | 01s | `registry/core.py:118-126` / `steps/__init__.py`: import the built-in configurable modules eagerly, or map built-in aliases to import paths |
| F18 | `DecisionTableConfig` open edges and contiguity are checked table-wide, not per `eq` group; `None` edges fail, `±inf` also fails contiguity, and only `±inf` with `allow_gaps` works | DESIGN + DOCS | medium | 00s, 05s | `steps/tables/schema.py:105-124`: run `check_rows` per group of rows sharing their `eq`/`in` values; meanwhile document `±inf` + `allow_gaps` |
| F19 | `.emit()` of an input nothing reads raises "not a declared input column", yet the column passes through anyway | MISLEADING | medium | 01s, 02s, 04s | `engine/wiring/resolve.py:245-261`: treat an unknown name as a pass-through no-op, or say that unread inputs already pass through |
| F20 | A param's name sent as a key of the request record is silently ignored | MISLEADING | medium | 08s | `engine/run/engine.py:119-127` (`_check_shadowing`): raise when a record key names a param of the pipeline and isn't an input column |
| F21 | Tree params are per call, not per row | DESIGN + DOCS | low | 04s | Document in `TreeConfig`: params are per run; a per-row threshold is a column inside a `ComputedFeature` (01s's pattern) |
| F22 | `python -m decider.cli` / `python -m decider` don't run | BUG | low | 06s | Add `decider/__main__.py` (and `decider/cli/__main__.py`): `from decider.cli import cli; cli()` |
| F23 | "decider CLI missing" (Haiku) | USER | low | 03h | None: `uv run --project <repo> decider --help` works; bare `decider` isn't on `PATH` outside the venv |
| F24 | `Engine().bind()` input classification depends on import order | NOT REPRO | low | 03s | 3 runs of each ordering and 8 hash seeds gave identical `_produced`/inputs; no action |
| F25 | `flow` runs in written order, unlike `dag` | DOCS | low | 07s | The error is clear; add one line to `flow`'s docstring |
| F26 | `dag` duplicate writers: reported one at a time, only at build, and always suggesting `flow` | MISLEADING | low | 01s, 08s, 10s | `steps/dag.py:33-42`: collect every collision into one error, and suggest "or relabel one" |
| F27 | `apply_stack_step` output-name collisions | USER (project 00 API, not decider) | low | 07s, 10s | In `credit_core.adjustments`: namespace `adjustment_set_id`/`adjustments_applied` by target; F26 covers decider's side |
| F28 | `missing_as(None)` is refused | DOCS | low | 02s, 09h, 01h | The message is right; cross-reference `T \| None` in the `missing_as` docstring |
| F29 | `Executable.run(dict)` fails with an `AttributeError`; `Engine(mode=)` | MISLEADING / USER | low | 00s, 10s | `engine/run/engine.py:140`: when `df` is a `Mapping`, say "use score() for one record" |
| F30 | A rule id with a hyphen can't be a param name | DOCS | low | 01s | Note in `param()`/`TreeConfig` docs that param names must be identifiers |
| F31 | No construct to run a sub-pipeline per element of a nested collection | DESIGN | low | 05s (05h, 04h asked for loops over collections) | Out of scope; the nested-`Engine`-in-`frame_step` pattern works (05s) |
| F32 | `inference.py` does `from decider import RequestHandler`, which is an `ImportError` | USER | low | 03h, 05h, 07h, 10h (found during this triage) | Optional: export `RequestHandler` from `decider.serving`, and show that import in `decider template` |

The Haiku "framework friction" lists are mostly feature requests, not defects: evidence tagging, table versioning, bi-temporal queries, reduce/population steps, overlay stacks, reuse instrumentation and state threading. They aren't triaged here.

---

## F1. Warm-up feeds `1.0` to `date`/`list`/`dict` inputs (BUG, blocker)

```python
import datetime as dt
from decider import Engine, flow
from decider.serving.handler import _warm
def age(decision_date: dt.date, history: list[float]) -> int:
    return decision_date.year - 2000 + len(history)
exe = Engine().bind(flow(age, name="p"))
print(exe.score({"decision_date": dt.date(2026, 1, 1), "history": [1.0]}))
try: _warm(exe, None)
except Exception as e: print(type(e).__name__, str(e).splitlines()[0])
```
```
{'decision_date': datetime.date(2026, 1, 1), 'history': [1.0], 'age': 27}
AttributeError 'float' object has no attribute 'year'
```
Root cause: `decider/serving/handler.py:198-206`. `_DUMMY` covers only `bool/int/str/bytes`. Everything else gets `1.0`, and `stage()` calls `_warm` with no extension point. Almost every spec has a `decision_date: date`. Fix: add `date: date(2000, 1, 1)` and `datetime`. When `<code_path>/sample_request.json` exists, warm from it (all ten Sonnet projects did exactly this by monkeypatching). If user code raises during warm-up, emit a warning naming the step instead of failing the build.

## F2. `code_path` loses to an earlier `PYTHONPATH` entry, so the wrong pipeline is served (BUG, blocker)

```bash
# two projects, each with pipeline.py; code_path = a
for PP in "$T/b:$T/a" "$T/b"; do
  cd $T/a && PYTHONPATH=$PP DECIDER_API__CODE_PATH=$T/a DECIDER_API__PIPELINE=pipeline:build \
  DECIDER_CONFIG__BASEPATH=$T/a/configs uv run -q --project $R decider build; done
```
```
PYTHONPATH=b:a  CODE_PATH=a
  imported project b pipeline.py
  built config version 0.0.0 (pipeline pipeline:build, mode fused)
PYTHONPATH=b  CODE_PATH=a
  imported project a pipeline.py
  built config version 0.0.0 (pipeline pipeline:build, mode fused)
```
Root cause: `decider/serving/handler.py:213-215` runs `if code_path not in sys.path: sys.path.insert(0, code_path)`. When the project directory is already on `PYTHONPATH` behind a dependency, the dependency's `pipeline.py` wins, and the build **succeeds** with the wrong logic. The build only fails when the params document happens not to match. This caused the Haiku 02/05/07/06 "no step with params" blockers: `build()` of project 00 was checked against their params. Fix: `if code_path in sys.path: sys.path.remove(code_path)`, then insert at 0. Also have `decider build` print the resolved pipeline module's `__file__`.

## F3. Nested and temporal columns round-trip through `Series.to_numpy()` (BUG, blocker; shared root of F3a–F3g)

`decider/engine/run/state.py:187-196` (`from_series`) calls `s.to_numpy()` on every non-numeric column: each frame-step output (`interpreted.py:244`) and each input column on the frame path (`state.py:58`). Numpy turns `List(Struct)` into 2-D floats, `List(Date)` into epoch ints, `List(List)` into arrays of arrays, and `List(String)` into object arrays. `_series` (`state.py:180-182`) then either can't rebuild a polars Series or rebuilds the wrong type.

Candidate fix, tested by monkeypatch (`fixplugin.py`):
```python
def from_series(s):
    if not (s.dtype.is_nested() or s.dtype.is_temporal()):
        return orig(s)                       # unchanged numeric/bool/str path
    values = np.empty(len(s), object)
    for i, x in enumerate(s.to_list()):
        values[i] = x
    return values, (s.is_not_null().to_numpy() if s.null_count() else None)
```
With it, every F3a–F3g repro below gives the right answer. `uv run pytest tests -p fixplugin` gives **1510 passed, 1 xfailed**, the same as the baseline.

### F3a/F3c. `list[struct]` and `list[list]` frame-step outputs
```python
@frame_step(reads=["x"], writes=["z"])
def fs(df): return df.with_columns(pl.Series("z", [[{"a": 1}]] * len(df)))
@frame_step(reads=["x"], writes=["z"])
def fl(df): return df.with_columns(pl.Series("z", [[[1, 2], [3]]] * len(df)))
def ps(x: float) -> list[dict]: return [{"a": x}]          # plain step, for comparison
```
```
frame list[struct] score ValueError cannot parse numpy data type dtype('O') into Polars data type
frame list[struct] run ValueError cannot parse numpy data type dtype('O') into Polars data type
frame list[list] score {'x': 1.0, 'z': [array([1, 2]), array([3])]}      # schema: Array(Object, shape=(2,))
step list[dict] score {'x': 1.0, 'ps': [{'a': 1.0}]}
step list[dict] run {'x': 1.0, 'ps': [{'a': 1.0}]}
```
10s reported that a *plain* step's `list[dict]` output crashes. In interpreted mode this was not reproduced, including when the value feeds a frame step. In fused mode it fails through F4.

### F3b. `list[struct]` passed from one frame step to another
```python
@frame_step(reads=["x"], writes=["accts"])
def merge(df): return df.with_columns(pl.Series("accts", [[{"amt": 1.0}, {"amt": 2.0}]] * len(df)))
@frame_step(reads=["accts"], writes=["total"])
def total(df): return df.with_columns(total=pl.col("accts").list.eval(pl.element().struct.field("amt")).list.sum())
Engine().bind(flow(merge, total, name="p")).run(pl.DataFrame({"x": [1.0]}))
```
```
ValueError cannot parse numpy data type dtype('O') into Polars data type
```

### F3d. `list[date]` degrades to integers between frame steps (silent)
```python
@frame_step(reads=["x"], writes=["ev_date"])
def s1(df): return df.with_columns(pl.Series("ev_date", [[dt.date(2026, 1, 1), dt.date(2025, 6, 1)]] * len(df)))
@frame_step(reads=["ev_date"], writes=["seen"])
def s2(df):
    print("  s2 sees", df.schema["ev_date"], df["ev_date"].to_list()[0])
    return df.with_columns(seen=pl.lit(1))
```
```
  s2 sees Array(Int64, shape=(2,)) [20454, 20240]
```

### F3e. `list[str]` output mixing empty and non-empty rows panics
```python
@frame_step(reads=["x"], writes=["y"])
def f(df): return df.with_columns(pl.Series("y", [["a", "b"], [], ["c"]], dtype=pl.List(pl.Utf8)))
Engine().bind(f).run(pl.DataFrame({"x": [1, 2, 3]}))
```
```
PanicException called `Result::unwrap()` on an `Err` value: SchemaMismatch(ErrString("invalid series dtype: expected `String`, got `object` for series with name ``"))
```

### F3f. `list[struct]` input read by a plain step on the frame path (silent)
```python
def first(accounts: list[dict]) -> str: return repr(accounts[0])
@frame_step(reads=["first"], writes=["n"])
def f(df): return df.with_columns(n=pl.col("first").str.len_chars())
exe = Engine().bind(flow(first, f, name="p").emit("first"))
exe.score({"accounts": [{"opened": dt.date(2020, 1, 1), "q": 1.0}, ...]})["first"]
```
```
same keys -> array([1.8262e+04, 1.0000e+00])
key missing -> array([1.8262e+04, 1.0000e+00])
date + str -> array([datetime.date(2020, 1, 1), 'A'], dtype=object)
datetime + missing float -> array([1.5778368e+15, 1.0000000e+00])
```
06s's `DTypePromotionError` panic, for records with different key sets, came from this `to_numpy()` call. The exact panic wasn't reproduced with small records, but the same call silently turns each struct into a row of floats.

### F3g. The Python type of a `list` input depends on the entry point
```python
def kind(hist: list[float]) -> str: return type(hist).__name__
exe = Engine().bind(flow(kind, name="p"))
print("score:", exe.score({"hist": [1.0, 2.0]})["kind"], "| run:", exe.run(pl.DataFrame({"hist": [[1.0, 2.0]]}))["kind"][0])
def falsy(hist: list[float]) -> int: return len(hist or [])
```
```
score: list | run: ndarray
`hist or []` on run(): ValueError The truth value of an array with more than one element is ambiguous.
```

## F4. Compiled-mode fallback stores `dict`/`list`/`date` outputs in a float64 buffer (BUG, high)

```python
from decider import Engine, flow
def d(x: float) -> dict: return {"a": x}
def l(x: float) -> list[float]: return [x]
def t(x: float) -> dt.date: return dt.date(2026, 1, 1)
for f in (d, l, t):
    for mode in ("interpreted", "fused"):
        print(f.__name__, mode, Engine().bind(flow(f, name="p"), mode=mode).score({"x": 1.0}))
```
```
d interpreted {'x': 1.0, 'd': {'a': 1.0}}
d fused TypeError float() argument must be a string or a real number, not 'dict'
l fused ValueError setting an array element with a sequence.
t fused TypeError float() argument must be a string or a real number, not 'datetime.date'
```
Root cause: `decider/engine/compile/units.py:98-101`. `Fallback.writes` uses `output_dtype()`, which is float64 for anything that isn't `float/int/bool/str`, and the write fails at `:123`. This is 08h's "`ge=/le=` bug": its one step returns `-> dict`, and removing the constraints never fixed the build. Fix, tested: `np.dtype(object) if base_annotation(o.annotation) not in (float, int, bool) and literal_choices(o.annotation) is None else output_dtype(o.annotation)`. The default serving mode is fused, so every Sonnet project would hit this as soon as F5 allowed it to leave interpreted mode.

## F5. Compiled modes reject `str`-vs-`str` comparisons and `str` literals (DESIGN, high)

```python
def pick(a_version: str, b_version: str, use_a: bool) -> str: return a_version if use_a else b_version
def same(a_version: str, b_version: str) -> bool: return a_version == b_version
def lit(band: str) -> bool: return band != "complete"
for f in (pick, same, lit):
    Engine().bind(flow(f, name="p"), mode="fused").score({"a_version": "v1", "b_version": "v2", "use_a": True, "band": "partial"})
```
```
pick ValueError p/pick: reads several `str` inputs ['a_version', 'b_version']; compiled modes compare a `str` input only with a `str` param, so split the step or run it in interpreted mode
same ValueError ... (same)
lit ValueError p/lit: `str` input 'band' enters a compiled kernel as a code, so a literal in the function body would never match it; declare the literal as a `str` param
```
Root cause: `decider/engine/run/runners/stepped.py:179-204` (`_str_params`). The message is clear, but it arrives at the first run rather than at bind time, and it forces the whole pipeline into interpreted mode. One inherited step (00's `norm_table_version`) put all ten Sonnet projects there. 07s measured 294 rows/s: 3.9 h for the 4.1M book against a 3 h window. `pick` doesn't even compare the two strings. Fix: route just that call through the existing `Fallback` Python unit (reason = this message), and keep the error behind an opt-in strict flag.

## F6. `[]` or a lone `None` infers `Null` on the one-record frame path (BUG, high)

```python
def n(hist: list[float], last: dt.date | None = None, v: float | None = None) -> int:
    return len(hist) + (last is not None) + (v is not None)
@frame_step(reads=["n"], writes=["m"])
def fm(df): return df.with_columns(m=pl.col("n") + 1)
# score() on flow(n) and on flow(n, fm) with each record below
```
```
no frame step {'hist': [1.0], 'last': None, 'v': 1.0} -> 2
no frame step {'hist': [], 'last': datetime.date(2026, 1, 1), 'v': 1.0} -> 2
with frame step {'hist': [1.0], 'last': None, 'v': 1.0} -> ArrowImportError sm_import_frame rc=-6: Expected array with 0 buffer(s) but found 1 buffer(s)
with frame step {'hist': [], 'last': datetime.date(2026, 1, 1), 'v': 1.0} -> ArrowImportError ...
```
Root cause: `decider/engine/run/engine.py:168-169`. With any frame step (`_record_path` is False, `:103`), `score()` builds `pl.DataFrame([record])`. A `None` date then infers `Null`, and `[]` infers `List(Null)`, which the Arrow boundary rejects. The typed path without frame steps works. Fix, tested in `Executable.prepare` (it covers `run()` too): drop `pl.Null` columns, since absent means null, and cast `List(Null)` columns to the declared list dtype. Workarounds seen: placeholder accounts (02s), sentinel dates `"2000-01-01"` (06s, 08s), filtering `None` keys (02s).

## F7. `missing_as([])` on an absent list column fills zero-width (BUG, high)

```python
def total(hist: list[float] = missing_as([])) -> float: return float(sum(hist))
@frame_step(reads=["total"], writes=["t2"])
def t2(df): return df.with_columns(t2=pl.col("total") * 2)
# score({"x": 1.0}) on flow(total) and flow(total, t2)
```
```
no frame step IndexError list index out of range
with frame step ShapeError unable to add a column of length 0 to a DataFrame of height 1
```
Root cause: `decider/engine/run/runners/interpreted.py:197` and `stepped.py:97`. `np.where(valid, values, [])` broadcasts the empty list to shape `(n, 0)`, so the step gets zero rows. This happens even without a frame step (00s thought one was needed). Fix, tested: for a list/dict/tuple fill, copy `values` to an object array and assign `decl.fill` to each missing row.

## F8. `list[float]` output dtype is inferred, not declared (BUG, high, silent)

```python
def chain(x: float) -> list[float]: return [9, 1.5] if x > 1 else [1, 2]
def rates(x: float) -> list[float]: return [1, 2] if x < 2 else [1.5]
Engine().bind(flow(chain, name="p")).run(pl.DataFrame({"x": [2.0]}))
Engine().bind(flow(rates, name="p")).run(pl.DataFrame({"x": [1.0, 2.0]}))["rates"].to_list()
```
```
run [2.0] TypeError unexpected value while building Series of type Int64; found value of type Float64: 1.5
silent: [[1, 2], [1]]
```
Root cause: `decider/engine/run/state.py:180-182`. `pl.Series(name, values.tolist())` infers `List(Int64)` from the first element, then either raises or **truncates 1.5 to 1**. `score()` is unaffected. 03s's `int`/`float` mix message came from here, and the error names neither the step nor the column. Fix: pass the polars dtype of the declared output annotation to `pl.Series`.

## F9. `missing_as(False)` and `param(False)` are truthy on a direct call (BUG, high, silent)

```python
from decider import Engine, flow, missing_as, param
def excluded(deceased: bool = missing_as(False), review: bool = param(False)) -> str:
    return f"deceased={bool(deceased)} review={bool(review)}"
print("direct call :", excluded())
print("through eng :", Engine().bind(flow(excluded, name="p")).score({"deceased": None})["excluded"])
```
```
direct call : deceased=True review=True
through eng : deceased=False review=False
```
Root cause: `decider/engine/params/declare.py:12,37-40`. `bool` can't be subclassed, so `_mark` returns a bare `MissingAs()`/`ParamSpec()`, and objects are truthy by default. This is wider than reported: `param(False)` is affected too. Fix, tested: `MissingAs.__bool__ = lambda s: bool(s.fill)` and `ParamSpec.__bool__ = lambda s: bool(s.default)`. `is False` still won't hold, so add a docstring line to that effect.

## F10. First reader's annotation types an input column (BUG, high, silent)

```python
def as_float(dependants: float) -> float: return dependants * 1.5
def as_int(dependants: int) -> str: return type(dependants).__name__
for order in ((as_float, as_int), (as_int, as_float)):
    exe = Engine().bind(flow(*order, name="p"))
    print(..., exe.score({"dependants": 2})["as_int"], exe.run(pl.DataFrame({"dependants": [2]}))["as_int"][0])
```
```
['as_float', 'as_int'] score: float run: float
['as_int', 'as_float'] score: int run: int
```
Root cause: `decider/engine/wiring/resolve.py:145-147` (`ponytail: the first reader's declaration describes the column`). Fix: raise a `WiringError` naming both readers when their base annotations differ, or cast to each reader's declared type in `_argument`.

## F11. Chained `.relabel(writes=)` is a silent no-op (BUG, high, silent)

```python
def band(score: float) -> int: return int(score // 100)
s = step(band).relabel(writes={"band": "band_a"})
print(Engine().bind(flow(s.relabel(writes={"band_a": "band_b"}), name="p")).score({"score": 250.0}))
print(Engine().bind(flow(s.relabel(writes={"band": "band_b"}), name="p")).score({"score": 250.0}))
```
```
twice by new name  : {'score': 250.0, 'band_a': 2}
twice by orig name : {'score': 250.0, 'band_b': 2}
```
Root cause: `decider/steps/base.py:75-78` merges dicts keyed by the *original* names, and ignores unknown keys. Fix: rewrite the entry whose current value equals the key, and raise on a key that is neither an original nor a current name. The same applies to `reads`.

## F12. Params-document errors don't say what is valid (MISLEADING, medium)

```python
def a(x: float) -> float: return x
def b(a: float, k: float = param(2.0)) -> float: return a * k
exe = Engine().bind(flow(a, b, name="p"))
for doc in ({"p": {"a": {}}}, {"operation": "replay"}):
    exe.score({"x": 1.0}, doc)
```
```
ParamsError params document: no step with params at 'p/a'.
ParamsError params document: no step with params at 'operation'.
```
Root cause: `decider/engine/run/params.py:87-99` (`_walk`). A hint appears only for a close match, so a flat document (09h), another pipeline's document (00h) or the wrong project's pipeline (F2) get no clue. An empty `{}` for a step without params, which `parameters().defaults()` style templates produce, is also rejected. Fix: when `hint()` is empty, append the valid keys at that level. Treat an empty mapping at a known step path as fine.

## F13. `frame_step` has no `param()` support (DESIGN, medium)

```python
@frame_step(reads=["x"], writes=["y"])
def g(df, k: float = param(3.0)): return df.with_columns(y=pl.col("x") * k)
Engine().bind(flow(g, name="p")).score({"x": 1.0}, {"p": {"g": {"k": 10.0}}})
```
```
param: ParamsError params document: no step with params at 'p'.
```
Root cause: `decider/engine/run/runners/interpreted.py:90-91` calls `_frame`, which calls `node.fn(df)` with no bundle. `steps/frame.py:30-33` declares no params. Without a params document the default is used silently. This blocks runtime overlay toggles in search-shaped flows (06s, where the workaround is a second `Engine`). Fix: harvest params from the signature after `df`, validate them like any node, and pass the bundle as keyword arguments in `_frame`.

## F14. `.relabel()` refused on frame steps (DESIGN, medium)

```python
@frame_step(reads=["x"], writes=["y"])
def f(df): return df.with_columns(y=pl.col("x") * 2)
Engine().bind(flow(f.relabel(writes={"y": "y2"}), name="p")).score({"x": 1.0})
```
```
relabel: IRError p/f: relabel can't rename a frame step's columns; rename them in the function
```
Root cause: `decider/engine/ir/context.py:270`. Fix: rename in `_frame`, applying `df.rename(reverse reads)` before the call and `out.rename(writes)` after.

## F15. The typo guard rejects legitimate inputs (DESIGN, medium)

```python
def entity_base_score() -> float: return 600.0
def total(entity_base_score: float, entity_bureau_score: float) -> float: return entity_base_score + entity_bureau_score
Engine().bind(flow(entity_base_score, total, name="t")).score({"entity_bureau_score": 50.0})
```
```
WiringError t/total: input 'entity_bureau_score' is not produced by any earlier step and is not a declared input column. Did you mean 'entity_base_score' (produced by 't/entity_base_score')? ...
```
Root cause: `decider/engine/wiring/resolve.py:136-143`, with `TYPO_CUTOFF = 0.8` in `registry/resolve.py:8`. Long shared prefixes (`applicant1_declared_income` vs `_expenses`, 07s) clear the cutoff. There is no escape hatch short of renaming. Fix: guess a typo only within ≤2 edits of a similar length, or allow a per-step `inputs=` declaration that marks a name as a genuine input.

## F16. `path_output` holds only the leaf id (DESIGN, medium)

```python
tree = TreeConfig.load({..., "path_output": "path", "tree": {root → mid → hi|lo}})
tree.run(pl.DataFrame({"a": [0.9, 0.9], "b": [0.9, 0.1]}))
```
```
[{'a': 0.9, 'b': 0.9, 'band': 1, 'path': 'hi'}, {'a': 0.9, 'b': 0.1, 'band': 2, 'path': 'lo'}]
```
This is documented in `decider/steps/trees/__init__.py:36-38` as "the id of the leaf that answered". Spec 04 needs the ordered node path (04s wrote a parallel Python walker; 04h wrote its own engine). Fix: an opt-in `trace_output` that writes `list[str]` of the visited node ids from both walkers.

## F17. Built-in `ConfigurableStep` tags need their module imported (BUG, medium)

```python
from decider import ConfigurableStep
ConfigurableStep.resolve("decision_table")
import decider.steps.tables
ConfigurableStep.resolve("decision_table")
```
```
before import: RegistryError 'decision_table' is not a registered ConfigurableStep type.
after import : DecisionTableConfig
```
Root cause: `decider/registry/core.py:118-126` resolves only registered tags, and `decider/steps/__init__.py` doesn't import `trees`/`tables`/`scorecard`. A `build(tree: ConfigurableStep)` signature, as in the handler docstring and template, fails unless something else imports the module. Fix: import the built-in configurable modules in `decider/steps/__init__.py`, or keep a built-in alias-to-import-path map that `resolve` imports lazily. The error should also say "import the module that defines it".

## F18. Decision-table band checks are table-wide (DESIGN + DOCS, medium)

```python
# two grades, each with bands (lo_open, 10) and (10, hi_open); expression = eq(grade) AND between(amount)
None edges -> ...
```
```
None edges -> Value error, Row 1: upper bound unresolvable — only row 3 may have an open upper edge.
+-inf edges -> Value error, Row 1 upper (inf) != row 2 lower (-inf): ranges are not contiguous. Set allow_gaps=True to permit this.
+-inf + allow_gaps -> [0.1, 0.2]
```
Root cause: `decider/steps/tables/schema.py:105-124` (`check_rows`) treats all rows as one ladder. This is documented on `BetweenExpression` ("only the first row may leave its lower edge open"). `±inf` alone does not work: `allow_gaps=True` is also needed, which disables the contiguity check that was the point. Fix: group rows by their `eq`/`in` column values and check each ladder separately.

## F19. `.emit()` of an unread input raises, though it passes through (MISLEADING, medium)

```python
def score(x: float) -> float: return x * 2
p = flow(score, name="p")
print(Engine().bind(p).score({"x": 1.0, "client_id": 7}))
Engine().bind(p.emit("client_id")).score({"x": 1.0, "client_id": 7})
```
```
no emit  : {'x': 1.0, 'client_id': 7, 'score': 2.0}
emit     : WiringError p: emit('client_id'): no step produces 'client_id' and it is not a declared input column.
```
Root cause: `decider/engine/wiring/resolve.py:245-261`. Unread inputs never enter scope, so the error says the column is lost when it isn't. An identity step `def client_id(client_id: int) -> int` works (01s reported otherwise; not reproduced). Fix: make emitting an unknown bare name a pass-through no-op, or say "input columns nothing reads already pass through".

## F20. A param's name in the request record is silently ignored (MISLEADING, medium)

```python
def intensity(base: int, stack_enabled: bool = param(True)) -> int: return base + 1 if stack_enabled else base
exe = Engine().bind(flow(intensity, name="p"))
print(exe.score({"base": 2, "stack_enabled": False}))
print(exe.score({"base": 2}, {"p": {"intensity": {"stack_enabled": False}}}))
```
```
record key : {'base': 2, 'stack_enabled': False, 'intensity': 3}
params doc : {'base': 2, 'intensity': 2}
```
Root cause: `decider/engine/run/engine.py:119-127`. Only produced columns are checked for shadowing; param names aren't. Fix: in `_check_shadowing`, raise (or warn) when a record/frame key equals a param name of the pipeline and no step reads it as an input.

## F21. Tree params are per call (DESIGN + DOCS, low)

The same `r_tree_path.py` run with `params={"t": {"cut": 2.0}}` changes every row. A params document is per run by design, so a per-row threshold must be a column. 01s showed the working pattern: a `ComputedFeature` expression such as `"amount - 8000.0 * mule_scam_amount_multiplier"`. Fix: say this in the `TreeConfig` docstring.

## F22. `python -m decider.cli` fails (BUG, low)

```
$ uv run --project <repo> python -m decider.cli --help
No module named decider.cli.__main__; 'decider.cli' is a package and cannot be directly executed
```
`pyproject.toml:49-50` defines only the console script. Fix: add a `decider/__main__.py` that calls `cli()`, and a matching `decider/cli/__main__.py`.

## F23. "decider CLI missing" (USER, low)

`uv run --project <repo> decider --help` prints the CLI usage. `which decider` finds nothing because the script lives in the repo venv. 03h didn't use the command in the brief.

## F24. Import order changes input classification (NOT REPRO)

Two scripts differing only in where `from decider import Engine` sits. Each builds 03s's pipeline with its rate card and prints `'accounts_in_arrears_count' in exe._produced`, `len(_produced)` and `len(plan.inputs)`. Results: 3 runs of each ordering plus 8 `PYTHONHASHSEED` values all gave `False 116 64`. The resolver iterates lists and dicts in insertion order, and I found no set-order dependence. No action.

## F25–F30. Low-severity wiring and documentation items

```python
try: Engine().bind(flow(b, a, name="f"))                       # F25
d = dag(step(set_id).relabel(writes={"set_id": "s"}), step(set_id2).relabel(writes={"set_id2": "s"}),
        step(a).relabel(writes={"a": "t"}), step(b).relabel(writes={"b": "t"}), name="d")
d.parameters()                                                  # F26
Engine().bind(flow(a, name="f")).run({"x": 1.0})               # F29
```
```
flow b,a : WiringError f/b reads 'a' as an input column, but f/a, which runs later, writes 'a'. Order is execution order: ...
dag built without error: DagStep
dag at .parameters(): WiringError dag 'd': set_id and set_id2 both write 's'; use flow(...) to apply them in written order, the later one winning
run(dict): AttributeError 'dict' object has no attribute 'columns'
```
- F25: the `flow` error is good. Add a docstring line.
- F26 (`steps/dag.py:33-42`): the first collision raises, the second (`t`) is never reported, and the fix offered is always `flow`, even where a rename is right (01s).
- F27: `credit_core.adjustments.apply_stack_step` is project 00's API, not decider.
- F28: the `missing_as(None)` message already says "annotate the input `T | None`"; cross-reference it in the docstring.
- F29: `Executable.run` should detect a `Mapping` and point to `score()`.
- F30: param names become namedtuple fields, so `CF-0412_amount_thresh` fails with `ValueError: Type names and field names must be valid identifiers`; document it.

## F31. No per-element sub-pipeline construct (DESIGN, low)

05s's nested `Engine` inside a `frame_step` works in both `score` and `run`, including zero entities. Recorded as a design gap for spec 05 §13 Q1; no repro needed.

## F32. `inference.py` imports `RequestHandler` from `decider` (USER, low)

Found while rebuilding the Haiku projects with a corrected `PYTHONPATH` (03h, 05h, 07h, 10h):
```
Error: config version latest failed to build: ImportError: cannot import name 'RequestHandler' from 'decider'
```
The class lives in `decider.serving.handler`. Optionally re-export it from `decider.serving` and show that import in the template's `inference.py`.
