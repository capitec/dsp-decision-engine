# Where does a one-record run spend its time outside the steps?

Date: 2026-10-10 · Script: spike, not kept (in-process A/B of old and new functions on the same pipeline, plus `cProfile`)

## Purpose
The twelve examples run 12 to 34 Python-run steps per pipeline. After the declared-output and not-integral checks were added, a one-record `score()` was about 7% slower than before them. Find which per-step work costs, and cut it without weakening a check.

## Method
- Attribution: the old and new version of one function were swapped inside one process while the same pipeline scored the same records, in blocks of 30 records alternating over 60 rounds (the machine is shared, and separate processes drifted by 10 to 20%). The p50 of the best round is compared.
- Call counts per record from `cProfile` told which functions were new.
- Whole-example numbers: one process per ref (`r3/integrated` and this branch), 4 alternating processes, the first one cold; `Executable.score` and `RequestHandler.process_fn` from raw JSON bytes.

## Result
Old code, per record, on shared-credit-core (12 Python-run steps, 15 declared outputs checked), affordability (15 steps) and loan-granting-pricing (34 steps):

| Cost | shared-credit-core | affordability | loan-granting-pricing |
|---|---|---|---|
| `check_output` (15 calls in shared-credit-core; pydantic alone is 1.1 us per call) | 77 us | 57 us | 221 us |
| not-integral check in `Fallback.run` | under 5 us | under 5 us | under 5 us |
| the rest of `Fallback.run`'s new per-call work (`dtype` flags and a zipped list rebuilt per call, a mask made and `.all()`-tested per output) | 76 us | 130 us | 192 us |
| `without_absent` | 0 (not on the `score` path; no example declares a `NotRequired` key) | 0 | 0 |

The rest of the gain came from work every example pays, older than the checks:
- `score` called `record_value` for each of the ~66 outputs, and it resolved the annotation (two `typing.get_origin` calls) for each: now only a struct output does, the rest go through `despan`. 80 us on shared-credit-core.
- A run nobody steps through still built a frozen-dataclass `Checkpoint` before and after each of ~96 nodes: `Engine._iterate` now asks the built-in runners for `None`s. 83 to 96 us.
- `mask.all()` costs 1.6 us on a one-row mask, `np.count_nonzero(mask) == mask.size` 0.5 us; `np.ones(1, bool)` costs 1.3 us against 0.3 us for `np.empty`. A `Fallback` now makes a mask only when a step returns a null. About 24 us.
- `without_absent` resolved the TypedDict's hints on every value (20 us a record on a list of records); now once.

Tried and dropped: a hand-written check for lists of numbers and text and for TypedDicts, ahead of pydantic. It saved 0.3 us on `list[int]` and cost 0.7 us more on a TypedDict, because pydantic's strict validation is compiled; 1.1 to 1.6 us per output stays.

All 12 examples, per record, `r3/integrated` to this branch (median of 3 alternating processes):

- `score` p50: shared-credit-core -19.2%, affordability -21.2%, loan-granting-pricing -11.6%, business-credit-e2e -13.6%, business-nested -12.8%, campaign-trees -12.5%, collections -9.6%, consolidation -11.4%, credit-limit-management -24.8%, retail-credit-e2e -24.6%, transaction-fraud -2.3%, governance-replay +1.3%. Geometric mean -13.9% (mean -13.8%, p99 -13.3%).
- `process_fn` from raw bytes: geometric mean of p50 -14.1%.
- The last two are dominated by the step bodies (14 ms and 27 ms), so the framework's share is below the noise (3%).
- The response bytes of all 12 examples are identical, interpreted and fused.

## Decision
Per-step work in `Fallback.run` and `Executable.score` is kept to what the step's declaration needs: flags decided at bind, masks made on demand, no objects built for a caller that does not read them. `check_output` stays strict; a cheaper check is only worth adding if it beats pydantic's own, which none of the shapes tried did. See [engine](../decisions/engine.md).
