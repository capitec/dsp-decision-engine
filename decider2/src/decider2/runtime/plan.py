"""The single-record plan — BOUNDARY-REWORK.md Stage 4.

About 90 % of real calls are `score()` (doc 02 §3.5), whose spec is 60 µs,
and before this module almost none of a call's cost was the decision: the
kernel is ~1.4 µs (EXPERIMENTS.md §N1) and the rest was **schema-invariant
Python redone on every call** — the interface re-walk with difflib
(`graph/pipeline.py:_walk`), `flatten_for_runtime` with its topological
sort, `resolve_params`' structural lookups, `np.array([value])` per input,
and inside each segment `inspect.signature(eval_str=True)` per output
(`compile/driver.py:_return_dtype`). Measured on the flagship: 421 µs p50,
of which the three kernels were under 5 µs.

`ScorePlan` is that schema-invariant work, done once and held as one
explicit object a reviewer can read: what is cached is exactly its fields,
and what a call still does per record is exactly `run()`.

**What a plan is keyed on / what invalidates it.** A plan is a pure
function of the arguments `build()` receives — the flattened `steps`,
`group_ids`, `owners`, the `interface`, `emit`, `param_spaces`, the
`MissingInputPolicy`, `mode` and `build_dir`. It caches nothing that
depends on a call's `record`, `params=` or `shared=` (those are `run()`'s
arguments and are merged fresh every call, so a retune takes effect on the
next call and, because the kernel takes params as arguments, without a
compile). The plan itself is cached only by `graph.pipeline.Pipeline.
score_plan()`, which holds it against the identity of the frozen
`Pipeline` fields it was built from (`elements`, `emits`,
`missing_input_policy`) and rebuilds on any mismatch — see that method for
the rule and its test.

**Thread safety.** Serving runs concurrent `score()` calls on one
`Pipeline` (EXPERIMENTS.md §N4). The pooled 1-row input buffers are
therefore **per thread** (`threading.local`), never shared: two threads
marshalling into one buffer would read each other's records — the cross-
talk `tests/test_score_plan.py` drives 16 threads to detect. A pool that
is already `busy` on its own thread (a re-entrant call — a `FallbackSegment`
step calling `score()` — or a generator suspended mid-call) is never
shared either; that call gets a throwaway pool. The output arrays are
allocated fresh per call by each `Segment.run` (`compile/driver.py`), so
they are never shared by construction; nothing in the returned dict
aliases a pooled buffer (`.item()` copies every terminal out).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from decider2.compile.driver import numpy_dtype
from decider2.runtime import modes
from decider2.runtime.invoke import ParamsPlan, ParamSpace, _default_build_dir, params_plan
from decider2.types import Decision, Input, Interface, MissingInputPolicy, NullPolicy, Step

__all__ = ["ScorePlan", "InputSlot"]


@dataclass(frozen=True)
class InputSlot:
    """One declared input, as `run()` marshals it: everything decided from
    the `Input` alone, once. `dtype` is `compile.driver.numpy_dtype(inp.
    annotation)` — the same resolution `apply()` uses, so an `int` input
    stays int64 (doc 03 §1) and a `str` one is an int32 code (doc 05
    §1.5). `raise_if_missing` is `MissingInputPolicy.raise_for` membership,
    resolved here rather than per call."""

    name: str
    dtype: np.dtype
    null_policy: NullPolicy
    fill: Any
    is_str: bool
    raise_if_missing: bool
    valid_key: str


class _RowPool:
    """One thread's pooled 1-row buffers: per slot `(slot, values, valid,
    code)` — the value array in the slot's own dtype, a bool validity
    array (registered only for an OPTIONAL input, `boundary.nulls` tier 3),
    and, for a `str` input, the constant int32 code-0 array a single record
    is its own dictionary for (see `ScorePlan.run`). `busy` guards against
    re-entrant use on the same thread."""

    __slots__ = ("entries", "busy")

    def __init__(self, slots: tuple[InputSlot, ...]) -> None:
        self.entries = tuple(
            (
                slot,
                np.empty(1, dtype=slot.dtype),
                np.empty(1, dtype=np.bool_),
                np.zeros(1, dtype=np.int32) if slot.is_str else None,
            )
            for slot in slots
        )
        self.busy = False


@dataclass(frozen=True)
class ScorePlan:
    """Everything a single-record call needs that does not depend on the
    record: built by `build()` once, run by `run()` per record."""

    steps: tuple[Step, ...]
    group_ids: tuple[int, ...]
    owners: tuple[str, ...]
    interface: Interface
    slots: tuple[InputSlot, ...]                # one per `interface.inputs`, in order
    params: ParamsPlan                          # the structural half of resolve_params
    policy: MissingInputPolicy
    terminal_names: frozenset                   # interface.terminals | emit — the driver key's
    terminal_order: tuple[str, ...]             # the same names, iterated once, for read-back
    mode: type                                  # the `runtime.modes.Mode` class, looked up once
    build_dir: Path
    _local: threading.local = field(default_factory=threading.local, init=False, repr=False, compare=False)

    @classmethod
    def build(
        cls,
        steps: Sequence[Step],
        *,
        interface: Interface,
        group_ids: Sequence[int],
        owners: Sequence[str] | None = None,
        mode: str = "fused",
        emit: Sequence[str] = (),
        param_spaces: Sequence[ParamSpace] | None = None,
        policy: MissingInputPolicy | None = None,
        build_dir: "str | Path | None" = None,
    ) -> "ScorePlan":
        """The same arguments `runtime.invoke.score` takes (its contract
        with the graph layer, see that module's docstring); this is what
        that function now builds and runs."""
        steps_t = tuple(steps)
        owners_t = tuple(owners) if owners is not None else tuple(s.name for s in steps_t)
        policy = policy or MissingInputPolicy()
        try:
            mode_cls = modes.MODES[mode]
        except KeyError:
            raise ValueError(f"unknown mode {mode!r}; expected one of {tuple(modes.MODES)}") from None
        terminal_names = frozenset(interface.terminals) | frozenset(emit)
        slots = tuple(
            InputSlot(
                name=inp.name,
                dtype=numpy_dtype(inp.annotation),
                null_policy=inp.null_policy,
                fill=inp.fill,
                is_str=inp.annotation is str,
                raise_if_missing=inp.name in policy.raise_for,
                valid_key=f"__valid__{inp.name}",
            )
            for inp in interface.inputs
        )
        return cls(
            steps=steps_t,
            group_ids=tuple(group_ids),
            owners=owners_t,
            interface=interface,
            slots=slots,
            params=params_plan(steps_t, param_spaces=param_spaces, owners=owners_t),
            policy=policy,
            terminal_names=terminal_names,
            terminal_order=tuple(terminal_names),
            mode=mode_cls,
            build_dir=_default_build_dir(build_dir),
        )

    # --- per call ------------------------------------------------------------

    def _pool(self) -> _RowPool:
        local = self._local
        pool = getattr(local, "pool", None)
        if pool is None:
            pool = _RowPool(self.slots)
            local.pool = pool
        elif pool.busy:
            # Re-entered on this thread while a call is in flight: a fresh,
            # unshared pool for this call, and the resident one stays put.
            return _RowPool(self.slots)
        return pool

    def run(
        self, record: Mapping[str, Any], *, params: Mapping[str, Any] | None = None,
        shared: Mapping[str, Any] | None = None,
    ) -> dict:
        """One record in, one dict out (doc 02 §3.5). What happens per
        call, and nothing else: marshal the record into this thread's
        pooled row (one loop over `slots`), merge this call's params
        (`ParamsPlan.resolve`), run the mode's segments over `n == 1`, read
        the terminals back."""
        pool = self._pool()
        pool.busy = True
        try:
            return self._run(pool, record, params, shared)
        finally:
            pool.busy = False

    def _run(self, pool: _RowPool, record: Mapping[str, Any], params: Any, shared: Any) -> dict:
        registry: dict[str, Any] = {}
        routed_reason: str | None = None
        record_categories: dict[str, tuple[str, ...]] | None = None
        get = record.get
        for slot, values, valid, code in pool.entries:
            # Doc 03 §1 (review finding 4): absent and null share one path —
            # `record.get` folds "absent" and "present but None" into the same
            # `value`, so `missing_as`/`not_applicable_as` fill an ABSENT key
            # exactly as they fill a present null, and a REQUIRED input that
            # is absent routes through `MissingInputPolicy` exactly like a
            # REQUIRED null does — naming the column, never a raw KeyError
            # from three frames away.
            value = get(slot.name)
            if value is None:
                null_policy = slot.null_policy
                if null_policy is NullPolicy.REQUIRED:
                    if slot.raise_if_missing:
                        raise ValueError(
                            f"step argument {slot.name!r} is declared required (no `| "
                            f"None`) but the record has no usable value for "
                            f"{slot.name!r} (absent or null), and {slot.name!r} is in "
                            "raise_for (doc 03 §1)."
                        )
                    if routed_reason is None:
                        routed_reason = slot.name
                    continue
                if null_policy is NullPolicy.OPTIONAL:
                    values[0] = 0
                    valid[0] = False
                    registry[slot.name] = values
                    registry[slot.valid_key] = valid
                    continue
                values[0] = slot.fill          # MISSING_AS / NOT_APPLICABLE_AS
                registry[slot.name] = values
                continue
            if slot.is_str and isinstance(value, str):
                # A str input enters the kernel as a dictionary code (doc 05
                # §1.5). apply() takes the dictionary from the column; a single
                # record has no column, so the record IS its own dictionary: this
                # value is code 0, and _resolve_str_param_code then gives the
                # literal 0 when it matches and -1 when it does not. `a == b` over
                # one row is exactly `code(a) == code(b)` under that mapping, so
                # score() and apply() agree without score() needing a declared
                # vocabulary it has no way to know. (Stage 2 replaces this block
                # with a span over the encoded bytes.)
                if record_categories is None:
                    record_categories = {}
                record_categories[slot.name] = (value,)
                registry[slot.name] = code
                continue
            values[0] = value
            registry[slot.name] = values
            if slot.null_policy is NullPolicy.OPTIONAL:
                valid[0] = True
                registry[slot.valid_key] = valid

        out = dict(record)
        if routed_reason is not None:
            # Doc 03 §1: "a null must be able to produce a decision, not an
            # exception." Rendering the decision fully is observe/'s job (not
            # built); surfaced here rather than silently computing nothing.
            policy = self.policy
            default = policy.default
            out["decision"] = default.value if isinstance(default, Decision) else default
            out["reason"] = policy.reason
            out["routed_on"] = routed_reason
            return out

        resolved = self.params.resolve(params, shared_overrides=shared, categories=record_categories)
        registry = self.mode.run(
            self.steps, self.group_ids, self.owners, registry, resolved, 1,
            terminal_names=self.terminal_names, build_dir=self.build_dir,
        )

        for name in self.terminal_order:
            arr = registry.get(name)
            if arr is not None:
                value = arr[0]
                out[name] = value.item() if hasattr(value, "item") else value
        return out
