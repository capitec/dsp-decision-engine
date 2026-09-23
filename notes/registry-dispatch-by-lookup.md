# Registry: dispatch by lookup, `SerializeAsAny` on nested steps

**Decision:**
- Nested `ConfigurableStep` fields resolve their type tag by a dictionary lookup
  at validation time (`StepRef`: a `WrapValidator` plus `SerializeAsAny`).
- No discriminated union is rebuilt when a class registers.

**Why:**
- **Stale unions.** decider_old's registry (`decider_old/_ext.py`,
  `create_extendable_model`) calls `_rebuild()` on every `register_provider`.
  That rebuild regenerates a `RootModel` over a `Union[...]` with
  `model_rebuild(force=True)`.
  - Each rebuild moves nested unions forward by only one level. With native
    recursion, `model_validate` then fails with about 100 spurious errors.
  - The fix under that design is "build once, after all registration", which
    makes plugin load order a correctness constraint. A late registration left
    a stale union with no error.
  - Registration was also superlinear (exponent 1.39; 333 ms for 175 types).
  - Nested errors lost the step index: a bad id three levels down reported loc
    `('sequential','steps','sequential','steps','sequential','steps')`.
- **Lookup vs union cost.** Measured on a 63-node, 3-level config: union 74.1 µs,
  plain dict registry 73.5 µs, native-recursion hybrid 58.5 µs. So Python-side
  lookup costs about 15 µs over the fastest option. That doesn't matter, because
  configs are validated once and cached.
- **Better errors.** The union's `union_tag_invalid` message is 1332 characters
  and lists every tag. A lookup gives a short did-you-mean.
- **Subclass fields silently dropped.** Without `SerializeAsAny`, pydantic dumps a
  nested subclass as its declared base type. Checked on pydantic 2.13.4: a
  `Sub(Base)` with `extra_field=7` in a field typed `Base` dumps as
  `{'type': 'sub'}`, with no warning. `SerializeAsAny[Base]` dumps
  `{'type': 'sub', 'extra_field': 7}`.

**What we tried:**
- decider2 picked the native-recursion hybrid, with one union sealed at
  finalisation. It is fastest, but it keeps the "all registration before first
  validation" ordering rule.
- The current `decider/registry/` already dispatches by lookup: `resolve(tag)`
  over a per-root dict, filled in `__pydantic_init_subclass__`, with a
  did-you-mean. It lacks import-path tags, aliases, same-name replacement,
  `StepRef` and `SerializeAsAny`; that is T2.1.
- `extra="forbid"` is required on params models. Without it, all three designs
  silently ignored a misspelled param and reported only the real field as
  missing.

**Source:** `decider2/docs/01-motivation-and-evidence.md` (§5.7);
`decider2/docs/02-architecture.md` (§2.2); `decider_old/_ext.py`;
`decider/registry/core.py`. The `SerializeAsAny` behaviour was checked directly;
the decider2 docs don't cover it.
