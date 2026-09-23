# Origins stay out of cache keys and generated identifiers

**Decision:**
- An IR node's `Origin` (path plus source import path) is metadata for events,
  traces and the debugger only.
- Compiled code is keyed by content: source bytes, including constants, plus
  argument types.
- Generated file, module and class names come from that content hash. They are
  never built from an origin path, a step name, `id()`, an address, a timestamp
  or a PID.

**Why:**
- **Paths change for reasons that don't change the code.** Renaming a step,
  nesting it one level deeper or reusing it at a second path changes its
  origin, and none of these changes its content.
- **numba's cache is sensitive to identity.** A same-content file that is renamed
  or moved to another directory misses (4.6–4.8 s recompile at 200 args, vs
  0.18 s warm).
- **Names aren't content.** In decider2, a NamedTuple's fingerprint covered only
  `__name__` and its field names and types. Two different classes with the same
  name therefore collided in dispatch. On numba 0.67 this was a correctness
  defect, not a speed one (1.03×).
- **Content is the right key.** The same content-hash key fixes the
  stale-constant cache bug; see `numba-compile-cache.md`. It also lets unchanged
  steps keep their compiled code when only one step is edited.

**What we tried:**
- decider2 derived generated names from module ids. That tied the cache to
  naming, and it opened the same-name collision above.
- Params bundle classes made with `collections.namedtuple` inside a function are
  pickled *by value*, with a fresh class per process. The warm process's index
  lookups all missed, and every save appended a new entry. For a 3-band table,
  14 specialisations were re-saved on every start, and one helper's index grew
  by 12 entries per process. The fix was to register each bundle class under a
  deterministic qualname derived from its fields, so it pickles by reference.
- **Caution:** keying on `co_code` alone is not enough, because it excludes
  `co_consts`. The content key must include constants, or hash the source text.

**Source:** `decider2/docs/EXPERIMENTS.md` (§C, §D, §M);
`decider2/docs/05-boundary-and-compilation.md` (§4.2);
`decider2/src/decider2/runtime/bundles.py`;
`experimentation/namedtuple-dispatch-trap/`; `IR.md` (§4.1, §8)
