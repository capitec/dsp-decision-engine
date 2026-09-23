# Python fallback: compile failures only, at kernel boundaries

**Decision:**
- A step numba can't compile runs in Python. The compiler splits the kernel
  around that step and keeps the steps before and after it compiled.
- Fallback is triggered only by a compile-failure exception: `NumbaError`, plus
  `UnsupportedBytecodeError` (see below).
- A runtime error always propagates, compiled or not.
- The fallback decision is cached per node and named in the build report.

**Why:**
- **One bad node takes down the whole driver.** In all six compile-failure modes
  tested, the whole fused driver failed; never just one node.
- **Per-row `objmode` is worse than plain Python.** 200k rows × 4 steps, median:

  | variant | time | vs all-njit |
  |---|---|---|
  | all-njit | 7.13 ms | 1.0× |
  | whole driver in plain Python | 167.3 ms | 23.5× |
  | one node via `objmode` per row | 548.4 ms | 77.0× |
  | `objmode` inside `prange` | 4380.6 ms | 614.7× |

  The last one compiles silently. Per-row `objmode` costs about 2.7 µs per row.
- **Catching bare `Exception` hides bugs.** It would turn a real bug (such as a
  `ZeroDivisionError`) into a silent "needed a fallback".
- **Division by zero differs by mode.** Over numpy columns it gives `inf` in
  Python, raises under njit serial, and gives `inf` again under `prange`. Tests
  must pin this per mode.

**What we tried:**
- **Per-node fallback inside a fused kernel** (the original decider2 promise).
  It can't exist: nopython code can't call back into Python except through
  `objmode`, which is slower than giving up.
- **Better escapes than per-row `objmode`,** both about 2.5× better:
  - hoist the escape to one `objmode` call per batch: 29.7×;
  - compute the column in Python first, then keep the kernel pure: 31.3×.
- **`NumbaError` alone:**
  - Most compile failures subclass `NumbaError`: `TypingError`,
    `UnsupportedError`, `LoweringError` and others.
  - The exception on numba 0.67.0 is `UnsupportedBytecodeError`, which subclasses
    plain `Exception`.
  - That is exactly what a step with an `import` inside its body raises. decider2
    therefore catches the tuple `(NumbaError, UnsupportedBytecodeError)`, still
    never bare `Exception`.

**Source:** `decider2/docs/EXPERIMENTS.md` (§B);
`decider2/docs/02-architecture.md` (§3.2);
`decider2/docs/05-boundary-and-compilation.md` (§6);
`experimentation/objmode-fallback-blast-radius/`;
`decider2/src/decider2/compile/driver.py`
