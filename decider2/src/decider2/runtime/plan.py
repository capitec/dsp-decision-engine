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
    resolved here rather than per call.

    A `bytes` input is a string SPAN (a tree's string feature, BOUNDARY-
    REWORK.md §2.1, Stage 2): it does not cross as a scalar at all but as
    one `(address, byte length)` pair pointing into the record's own
    encoded text, so `is_span` gives it a `(1, 2)` int64 pooled buffer and
    an int64 `dtype` rather than `numpy_dtype`'s float64 default."""

    name: str
    dtype: np.dtype
    null_policy: NullPolicy
    fill: Any
    is_str: bool
    raise_if_missing: bool
    valid_key: str
    is_span: bool = False


class _RowPool:
    """One thread's pooled 1-row buffers: per slot `(slot, values, valid,
    code, null_span)` — the value array in the slot's own dtype (`(1, 2)`
    int64 for a SPAN slot), a bool validity array (registered only for an
    OPTIONAL input, `boundary.nulls` tier 3), for a `str` input the
    constant int32 code-0 array a single record is its own dictionary for
    (see `ScorePlan.run`), and for a SPAN slot the constant `(0, -1)` null
    span — the same thing `sm_gather_row` writes for a null string
    (`_arrow/frame.py`). `busy` guards against re-entrant use on the same
    thread."""

    __slots__ = ("entries", "busy")

    def __init__(self, slots: tuple[InputSlot, ...]) -> None:
        self.entries = tuple(
            (
                slot,
                np.empty((1, 2), dtype=np.int64) if slot.is_span else np.empty(1, dtype=slot.dtype),
                np.empty(1, dtype=np.bool_),
                np.zeros(1, dtype=np.int32) if slot.is_str else None,
                np.array([[0, -1]], dtype=np.int64) if slot.is_span else None,
            )
            for slot in slots
        )
        self.busy = False


def _slot(inp: Input, policy: MissingInputPolicy) -> InputSlot:
    """One declared input's `InputSlot`. A `bytes` input is a string span,
    so its buffer is `(1, 2)` int64 and a null is the span `(0, -1)` — the
    same refusal `_arrow.frame.FramePlan` makes for a fill on a STR column
    is made here, so `score()` and `apply()` cannot disagree about what a
    missing string is."""
    is_span = inp.annotation is bytes
    if is_span and inp.fill is not None:
        raise ValueError(f"{inp.name}: a STR feature has no fill; a null is length -1")
    return InputSlot(
        name=inp.name,
        dtype=np.dtype(np.int64) if is_span else numpy_dtype(inp.annotation),
        null_policy=inp.null_policy,
        fill=inp.fill,
        is_str=inp.annotation is str,
        raise_if_missing=inp.name in policy.raise_for,
        valid_key=f"__valid__{inp.name}",
        is_span=is_span,
    )


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
        slots = tuple(_slot(inp, policy) for inp in interface.inputs)
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
        # The UTF-8 bytes of every `bytes` (string-span) input, kept alive
        # until the kernel has run: the span in `registry` is an ADDRESS
        # into them. Empty unless the pipeline declares a `bytes` input.
        keepalive: list[bytes] = []
        get = record.get
        for slot, values, valid, code, null_span in pool.entries:
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
                    if slot.is_span:
                        registry[slot.name] = null_span
                    else:
                        values[0] = 0
                        registry[slot.name] = values
                    valid[0] = False
                    registry[slot.valid_key] = valid
                    continue
                # MISSING_AS / NOT_APPLICABLE_AS. A span slot has no fill
                # (refused in `_slot`, as `FramePlan` refuses it), so its
                # null is the null span the gather would have written.
                if slot.is_span:
                    registry[slot.name] = null_span
                    continue
                values[0] = slot.fill
                registry[slot.name] = values
                continue
            if slot.is_span:
                # A string SPAN input (a tree's string feature, docs/
                # BOUNDARY-REWORK.md §2.1, Stage 2): the record's text is
                # encoded once and the span is its address + byte length —
                # no polars, no Arrow, no dictionary. This is the ladder's
                # INDEPENDENT producer (§1.6): `apply()` reads the same
                # bytes out of polars' own memory through nanoarrow, and
                # `assert_equivalent`'s fourth rung checks the two agree.
                # The encoded bytes are kept alive until the kernel has
                # run (`keepalive`); the span is an ADDRESS into them.
                if not isinstance(value, str):
                    raise TypeError(
                        f"input {slot.name!r} is a string feature, but the record holds "
                        f"{type(value).__name__} {value!r}; pass a str."
                    )
                encoded = value.encode("utf-8")
                keepalive.append(encoded)
                buf = np.frombuffer(encoded, dtype=np.uint8)
                values[0, 0] = buf.ctypes.data
                values[0, 1] = buf.size
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
                # vocabulary it has no way to know. A tree's `bytes` feature
                # takes the span branch above instead; this block is the
                # hand-written-step `str` convention, Stage 7's surface.
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
        del keepalive   # the kernel has run; no span in `registry` is read after this

        for name in self.terminal_order:
            arr = registry.get(name)
            if arr is not None:
                value = arr[0]
                out[name] = value.item() if hasattr(value, "item") else value
        return out
