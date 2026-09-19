"""E11 — does param() in a step signature survive the compiler?

Tests doc 03 section 4.4 ("Declaring a param in the signature") and the five
items listed in doc 06 E11.

Run:  .venv/bin/python experimentation/e11-param-in-signature/e11.py
      .venv/bin/python experimentation/e11-param-in-signature/e11.py --part 5
      .venv/bin/python experimentation/e11-param-in-signature/e11.py --child <variant>

Part 5 variants run in FRESH SUBPROCESSES on purpose: the failure mode under
test (doc 01 section 4c) is a contamination of numba's per-process argument-type
cache, so measuring two variants in one interpreter would let the first poison
the second.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import inspect
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

import numpy as np
from numba import njit

# --------------------------------------------------------------------------
# A minimal but real implementation of doc 03 section 4.4's mechanism.
# --------------------------------------------------------------------------


class ParamSpec:
    """The sentinel that sits in the signature as a default value."""

    __slots__ = ("default", "field_kwargs")

    def __init__(self, default: Any, **field_kwargs: Any) -> None:
        self.default = default
        self.field_kwargs = field_kwargs

    def __repr__(self) -> str:
        return f"param({self.default!r}, {self.field_kwargs})"


def param(default: Any, **field_kwargs: Any) -> ParamSpec:
    """`param()` takes exactly what `Field()` takes (doc 03 section 4.4)."""
    return ParamSpec(default, **field_kwargs)


def harvest(fn) -> tuple[dict[str, tuple[type, Any]], list[str]]:
    """Split a step signature into (declared params, plain inputs)."""
    sig = inspect.signature(fn)
    params: dict[str, tuple[type, Any]] = {}
    inputs: list[str] = []
    for name, p in sig.parameters.items():
        if isinstance(p.default, ParamSpec):
            ann = p.annotation if p.annotation is not inspect.Parameter.empty else float
            params[name] = (ann, p.default)
        else:
            inputs.append(name)
    return params, inputs


def model_for(fn, *, model_name: str | None = None):
    """Harvest a signature into a generated pydantic model."""
    from pydantic import ConfigDict, Field, create_model

    declared, _ = harvest(fn)
    fields = {
        name: (ann, Field(spec.default, **spec.field_kwargs))
        for name, (ann, spec) in declared.items()
    }
    # doc 01 section 5.7 / E0 constraint: extra="forbid" on params models.
    return create_model(
        model_name or f"{fn.__name__}_Params",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


_BUNDLE_CACHE: dict[int, type] = {}


def bundle_type(model, *, memoise: bool = True, class_name: str | None = None) -> type:
    """The NamedTuple type a params model becomes (doc 03 section 4).

    `memoise=False` simulates regenerating the bundle on every pipeline build,
    which is the residual case doc 01 section 4c's implementation rule closes
    with an lru_cache.
    """
    key = id(model)
    if memoise and key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[key]
    name = class_name or model.__name__
    nt = collections.namedtuple(name, list(model.model_fields))
    if memoise:
        _BUNDLE_CACHE[key] = nt
    return nt


def bundle(model, nt: type, **overrides):
    """Validate through pydantic, then freeze into the NamedTuple."""
    validated = model(**overrides)
    return nt(**{f: getattr(validated, f) for f in model.model_fields})


def step(fn):
    """@step: keeps the function directly callable (doc 03 section 11).

    Substitutes declared defaults so a plain Python call needs no params
    bundle, and accepts `params=` to override.
    """
    declared, _ = harvest(fn)
    model = model_for(fn)

    def caller(*args, params=None, **kwargs):
        if params is None:
            values = {n: spec.default for n, (_, spec) in declared.items()}
        elif isinstance(params, dict):
            values = {n: getattr(model(**params), n) for n in declared}
        else:
            values = {n: getattr(params, n) for n in declared}
        values.update({k: v for k, v in kwargs.items() if k in declared})
        plain = {k: v for k, v in kwargs.items() if k not in declared}
        return fn(*args, **plain, **values)

    caller.__name__ = fn.__name__
    caller.__wrapped__ = fn
    caller.params_model = model
    caller.declared_params = declared
    return caller


STEP_BODY = "    return min(term_cap, cap) if min_net_salary < 5000 else term_cap\n"


def emit_step_source(*, keep_default: bool) -> str:
    """Emit a step two ways: sentinel default kept, or stripped by codegen."""
    default = " = _SENTINEL" if keep_default else ""
    return (
        "from numba import njit\n"
        "class ParamSpec:\n"
        "    def __init__(self, default, **kw):\n"
        "        self.default = default\n"
        "        self.field_kwargs = kw\n"
        "_SENTINEL = ParamSpec(48.0, ge=6, le=60)\n"
        "@njit\n"
        "def cap_by_income_band(term_cap: float, min_net_salary: float, "
        f"cap: float{default}) -> float:\n"
        + STEP_BODY
    )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def timeit_us(call, repeats: int) -> dict[str, float]:
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        call()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return {
        "median_us": round(statistics.median(ts), 3),
        "mean_us": round(statistics.fmean(ts), 3),
        "p10_us": round(ts[int(0.10 * len(ts))], 3),
        "p90_us": round(ts[int(0.90 * len(ts))], 3),
        "n": repeats,
    }


def build_module(src: str, name: str):
    """Write generated source to a real file and import it.

    Doc 01 section 4d / E9: real files rather than exec, for compile cost.
    """
    d = tempfile.mkdtemp(prefix="e11_")
    path = os.path.join(d, f"{name}.py")
    with open(path, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


# --------------------------------------------------------------------------
# PART 1 — does njit tolerate a sentinel default it never uses?
# --------------------------------------------------------------------------


def cap_by_income_band(
    term_cap: float,
    min_net_salary: float,
    cap: float = param(48.0, ge=6, le=60, description="Term cap in months"),
) -> float:
    """Cap term at 48 months below the income floor."""
    return min(term_cap, cap) if min_net_salary < 5000 else term_cap


def part1() -> dict:
    out: dict[str, Any] = {}

    # (a) njit the function AS WRITTEN, sentinel default and all.
    try:
        jit_raw = njit(cap_by_income_band)
        t0 = time.perf_counter()
        v = jit_raw(60.0, 4000.0, 48.0)
        out["a_sentinel_default_kept"] = {
            "compiled": True,
            "result": float(v),
            "compile_s": round(time.perf_counter() - t0, 3),
            "signatures": len(jit_raw.signatures),
        }
    except Exception as exc:  # noqa: BLE001
        out["a_sentinel_default_kept"] = {
            "compiled": False,
            "error": f"{type(exc).__name__}: {exc}"[:400],
        }

    # (a2) same dispatcher, called WITHOUT the param — does the sentinel reach
    #      the compiler then?
    try:
        v = jit_raw(60.0, 4000.0)
        out["a2_called_without_param"] = {"ok": True, "result": float(v)}
    except Exception as exc:  # noqa: BLE001
        out["a2_called_without_param"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {str(exc).strip().splitlines()[0]}"[:300],
        }

    # (b) both emissions, from REAL generated source files: sentinel default
    #     kept vs stripped by codegen. Same body, same call.
    for label, keep in (("b_default_kept_generated", True),
                        ("b_default_stripped_generated", False)):
        try:
            m, path = build_module(emit_step_source(keep_default=keep),
                                   f"e11_part1_{label}")
            t0 = time.perf_counter()
            v = m.cap_by_income_band(60.0, 4000.0, 48.0)
            out[label] = {
                "compiled": True,
                "result": float(v),
                "compile_s": round(time.perf_counter() - t0, 3),
                "signatures": [str(x) for x in m.cap_by_income_band.signatures],
                "source_line": [l for l in open(path) if l.startswith("def cap_")][0].strip(),
            }
        except Exception as exc:  # noqa: BLE001
            out[label] = {"compiled": False,
                          "error": f"{type(exc).__name__}: {exc}"[:400]}

    # (c) does a sentinel-defaulted step compile when CALLED FROM another
    #     njit function that passes the param explicitly? This is the real
    #     driver situation.
    try:
        src = (
            "from numba import njit\n"
            "import numpy as np\n"
            "from e11_support import SENTINEL_STEP as _s\n"
        )
        # inline the step source instead, so the sentinel literally appears.
        src = (
            "import numpy as np\n"
            "from numba import njit\n"
            "class _P:\n"
            "    def __init__(self, d): self.d = d\n"
            "_SENT = _P(48.0)\n"
            "@njit\n"
            "def inner(term_cap, min_net_salary, cap=_SENT):\n"
            "    return min(term_cap, cap) if min_net_salary < 5000 else term_cap\n"
            "@njit\n"
            "def driver(x, out, cap):\n"
            "    for i in range(x.shape[0]):\n"
            "        out[i] = inner(x[i], 4000.0, cap)\n"
        )
        mod, _ = build_module(src, "e11_part1c")
        x = np.array([60.0, 50.0])
        o = np.zeros(2)
        t0 = time.perf_counter()
        mod.driver(x, o, 48.0)
        out["c_driver_calls_sentinel_step"] = {
            "compiled": True,
            "result": o.tolist(),
            "compile_s": round(time.perf_counter() - t0, 3),
        }
    except Exception as exc:  # noqa: BLE001
        out["c_driver_calls_sentinel_step"] = {
            "compiled": False,
            "error": f"{type(exc).__name__}: {str(exc).strip().splitlines()[0]}"[:300],
        }

    return out


# --------------------------------------------------------------------------
# PART 2 — generated model vs hand-written model
# --------------------------------------------------------------------------


def part2() -> dict:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError

    class cap_by_income_band_Params(BaseModel):  # noqa: N801  (matches generated name)
        model_config = ConfigDict(extra="forbid")
        cap: float = Field(48.0, ge=6, le=60, description="Term cap in months")

    hand = cap_by_income_band_Params
    gen = model_for(cap_by_income_band)

    res: dict[str, Any] = {
        "generated_name": gen.__name__,
        "hand_name": hand.__name__,
        "names_equal": gen.__name__ == hand.__name__,
        "fields_equal": list(gen.model_fields) == list(hand.model_fields),
        "json_schema_equal": gen.model_json_schema() == hand.model_json_schema(),
    }
    res["generated_schema"] = gen.model_json_schema()
    res["hand_schema"] = hand.model_json_schema()

    # validators: same accept/reject behaviour on the declared bounds
    probes = [
        ("in_range", {"cap": 48.0}),
        ("at_lower_bound", {"cap": 6.0}),
        ("below_lower_bound", {"cap": 5.0}),
        ("above_upper_bound", {"cap": 61.0}),
        ("extra_field", {"cap": 48.0, "bogus": 1.0}),
        ("default_only", {}),
    ]
    behaviour = {}
    for label, kw in probes:
        def run(m):
            try:
                return ("ok", m(**kw).cap)
            except ValidationError as exc:
                return ("error", exc.errors()[0]["type"])
        behaviour[label] = {"generated": run(gen), "hand": run(hand),
                            "agree": run(gen) == run(hand)}
    res["validator_behaviour"] = behaviour
    res["validators_equal"] = all(v["agree"] for v in behaviour.values())

    # NamedTuple type equality, as numba sees it
    import numba

    nt_gen = bundle_type(gen)
    nt_hand = bundle_type(hand)
    b_gen = bundle(gen, nt_gen)
    b_hand = bundle(hand, nt_hand)
    t_gen = numba.typeof(b_gen)
    t_hand = numba.typeof(b_hand)
    res["numba_type_generated"] = str(t_gen)
    res["numba_type_hand"] = str(t_hand)
    res["numba_types_equal"] = bool(t_gen == t_hand)
    res["namedtuple_classes_identical"] = nt_gen is nt_hand
    res["bundle_values_equal"] = tuple(b_gen) == tuple(b_hand)
    return res


# --------------------------------------------------------------------------
# PART 3 — retuning does not recompile; retyping does
# --------------------------------------------------------------------------


def part3(retunes: int = 15) -> dict:
    gen = model_for(cap_by_income_band)
    nt = bundle_type(gen)

    src = (
        "import numpy as np\n"
        "from numba import njit\n"
        "@njit\n"
        "def driver(x, out, p):\n"
        "    for i in range(x.shape[0]):\n"
        "        out[i] = min(x[i], p.cap)\n"
    )
    mod, path = build_module(src, "e11_part3")
    x = np.array([60.0, 50.0, 40.0])
    o = np.zeros(3)

    mod.driver(x, o, bundle(gen, nt, cap=48.0))  # warm / compile
    after_warm = len(mod.driver.signatures)

    values = []
    for i in range(retunes):
        v = 6.0 + (i * 54.0 / max(retunes - 1, 1))
        mod.driver(x, o, bundle(gen, nt, cap=v))
        values.append(round(v, 2))
    after_retunes = len(mod.driver.signatures)

    # negative control: change the DECLARED TYPE of the param, float -> int
    def cap_by_income_band_int(
        term_cap: float,
        min_net_salary: float,
        cap: int = param(48, ge=6, le=60),
    ) -> float:
        return min(term_cap, cap) if min_net_salary < 5000 else term_cap

    gen_int = model_for(cap_by_income_band_int)
    nt_int = bundle_type(gen_int, class_name="cap_by_income_band_Params")
    mod.driver(x, o, bundle(gen_int, nt_int, cap=48))
    after_retype = len(mod.driver.signatures)

    return {
        "retunes": retunes,
        "retuned_values": values,
        "signatures_after_warmup": after_warm,
        "signatures_after_retunes": after_retunes,
        "signatures_after_type_change": after_retype,
        "retune_adds_signature": after_retunes != after_warm,
        "retype_adds_signature": after_retype > after_retunes,
        "signature_strings": [str(s) for s in mod.driver.signatures],
        "module_path": path,
    }


# --------------------------------------------------------------------------
# PART 4 — direct Python call without a params bundle
# --------------------------------------------------------------------------


def part4() -> dict:
    from pydantic import ValidationError

    s = step(cap_by_income_band)
    res: dict[str, Any] = {}

    # plain call, no params at all -> declared default substituted
    res["plain_call"] = {
        "call": "s(term_cap=60.0, min_net_salary=4000.0)",
        "result": s(term_cap=60.0, min_net_salary=4000.0),
        "expected_from_default_48": 48.0,
    }
    res["plain_call"]["matches_default"] = (
        res["plain_call"]["result"] == 48.0
    )

    # above threshold -> default irrelevant
    res["plain_call_above_threshold"] = s(term_cap=60.0, min_net_salary=9000.0)

    # test override via params= (bundle)
    gen = s.params_model
    nt = bundle_type(gen)
    res["params_bundle_override"] = {
        "cap": 24.0,
        "result": s(term_cap=60.0, min_net_salary=4000.0,
                    params=bundle(gen, nt, cap=24.0)),
    }
    # test override via params= (dict, validated)
    res["params_dict_override"] = {
        "cap": 12.0,
        "result": s(term_cap=60.0, min_net_salary=4000.0, params={"cap": 12.0}),
    }
    # an out-of-bounds override must be rejected by the harvested validator
    try:
        s(term_cap=60.0, min_net_salary=4000.0, params={"cap": 999.0})
        res["out_of_bounds_override_rejected"] = False
    except ValidationError as exc:
        res["out_of_bounds_override_rejected"] = True
        res["out_of_bounds_error"] = exc.errors()[0]["type"]

    # the UNWRAPPED function, called plainly, receives the sentinel — this is
    # the failure @step exists to prevent.
    try:
        raw = cap_by_income_band(60.0, 4000.0)
        res["unwrapped_plain_call"] = {
            "raised": False,
            "returns": repr(raw)[:160],
            "is_float": isinstance(raw, float),
        }
    except Exception as exc:  # noqa: BLE001
        res["unwrapped_plain_call"] = {
            "raised": True,
            "error": f"{type(exc).__name__}: {exc}"[:200],
        }
    return res


# --------------------------------------------------------------------------
# PART 5 — THE RISK: one bundle per function, 20 functions with a param 'cap'
# --------------------------------------------------------------------------

N_FUNCS = 20
DISPATCH_REPEATS = 2000

VARIANTS = {
    # label: (class_name(i), field_name(i))
    "naive_same_name": (lambda i: "Params", lambda i: "cap"),
    "unique_class_name": (lambda i: f"step{i}_Params", lambda i: "cap"),
    "unique_field_name": (lambda i: "Params", lambda i: f"cap_{i}"),
    "unique_both": (lambda i: f"step{i}_Params", lambda i: f"cap_{i}"),
    "one_bundle_baseline": (lambda i: "Params", lambda i: "cap"),
}


def part5_child(variant: str, n: int | None = None) -> dict:
    """Build a driver taking `n` params bundles and time per-call dispatch."""
    scalar = variant == "scalar_args_control"
    name_of, field_of = VARIANTS.get(variant, (lambda i: "Params", lambda i: "cap"))
    if n is None:
        n = 1 if variant == "one_bundle_baseline" else N_FUNCS

    lines = ["import numpy as np", "from numba import njit", ""]
    for i in range(n):
        lines.append(
            f"@njit\ndef step{i}(x, cap):\n"
            f"    return min(x, cap) if x > 0 else x\n"
        )
    args = ", ".join(f"p{i}" for i in range(n))
    if scalar:
        body = "\n".join(f"        v = step{i}(v, p{i})" for i in range(n))
    else:
        body = "\n".join(f"        v = step{i}(v, p{i}.{field_of(i)})" for i in range(n))
    lines.append(
        f"@njit\ndef driver(x, out, {args}):\n"
        f"    for i in range(x.shape[0]):\n"
        f"        v = x[i]\n{body}\n        out[i] = v\n"
    )
    mod, path = build_module("\n".join(lines), f"e11_part5_{variant}")

    if scalar:
        classes = []
        bundles = [float(60 - i) for i in range(n)]
    else:
        classes = [collections.namedtuple(name_of(i), [field_of(i)]) for i in range(n)]
        bundles = [classes[i](float(60 - i)) for i in range(n)]

    x = np.array([50.0])
    o = np.zeros(1)

    t0 = time.perf_counter()
    mod.driver(x, o, *bundles)
    compile_s = round(time.perf_counter() - t0, 3)
    for _ in range(50):  # extra warm
        mod.driver(x, o, *bundles)

    timing = timeit_us(lambda: mod.driver(x, o, *bundles), DISPATCH_REPEATS)

    import numba

    distinct_classes = len({id(c) for c in classes})
    distinct_names = len({c.__name__ for c in classes})
    numba_types = [str(numba.typeof(b)) for b in bundles]
    distinct_type_strings = len(set(numba_types))
    # do the numba types of two distinct same-named classes compare equal?
    type_eq = None
    if n > 1 and not scalar:
        ta, tb = numba.typeof(bundles[0]), numba.typeof(bundles[1])
        type_eq = {
            "type0": str(ta),
            "type1": str(tb),
            "equal": bool(ta == tb),
            "hash_equal": hash(ta) == hash(tb),
        }

    return {
        "variant": variant,
        "n_bundles": n,
        "us_per_bundle": None,
        "compile_and_first_call_s": compile_s,
        "distinct_python_classes": distinct_classes,
        "distinct_class_names": distinct_names,
        "distinct_numba_type_strings": distinct_type_strings,
        "type_compare": type_eq,
        "driver_signatures": len(mod.driver.signatures),
        "result": o.tolist(),
        "dispatch": timing,
        "module_path": path,
    }


def part5_regen_child(same: bool) -> dict:
    """The memoisation wrinkle (doc 01 section 4c note 2).

    One function. Its bundle class is either memoised (one class) or
    regenerated (two distinct classes, identical name and fields). The driver
    is called alternately with instances of each.
    """
    src = (
        "import numpy as np\n"
        "from numba import njit\n"
        "@njit\n"
        "def driver(x, out, p):\n"
        "    for i in range(x.shape[0]):\n"
        "        out[i] = min(x[i], p.cap)\n"
    )
    mod, path = build_module(src, "e11_part5_regen")
    A = collections.namedtuple("cap_by_income_band_Params", ["cap"])
    B = A if same else collections.namedtuple("cap_by_income_band_Params", ["cap"])
    ba, bb = A(48.0), B(36.0)
    x = np.array([50.0])
    o = np.zeros(1)
    mod.driver(x, o, ba)
    mod.driver(x, o, bb)
    for _ in range(50):
        mod.driver(x, o, ba)
        mod.driver(x, o, bb)

    flip = [ba, bb]
    ctr = {"i": 0}

    def call():
        ctr["i"] += 1
        mod.driver(x, o, flip[ctr["i"] & 1])

    timing = timeit_us(call, DISPATCH_REPEATS)
    return {
        "variant": "regen_memoised" if same else "regen_not_memoised",
        "classes_identical": A is B,
        "driver_signatures": len(mod.driver.signatures),
        "dispatch": timing,
        "module_path": path,
    }


def part5_contaminate() -> dict:
    """Does one contaminated call poison an UNRELATED later call?

    doc 01 section 4c: "permanently, for every call involving that name".
    Driver B is a clean single-bundle driver whose class is named `Params`
    with field `cap`. It is timed before and after driver A is called with 20
    distinct classes all named `Params` with field `cap`.
    """
    src_b = ("import numpy as np\nfrom numba import njit\n@njit\n"
             "def driver_b(x, out, p):\n"
             "    for i in range(x.shape[0]):\n"
             "        out[i] = min(x[i], p.cap)\n")
    mod_b, _ = build_module(src_b, "e11_contam_b")
    Clean = collections.namedtuple("Params", ["cap"])
    pb = Clean(48.0)
    x = np.array([50.0]); o = np.zeros(1)
    mod_b.driver_b(x, o, pb)
    for _ in range(50):
        mod_b.driver_b(x, o, pb)
    before = timeit_us(lambda: mod_b.driver_b(x, o, pb), 1000)

    n = N_FUNCS
    lines = ["import numpy as np", "from numba import njit", ""]
    for i in range(n):
        lines.append(f"@njit\ndef s{i}(x, cap):\n    return min(x, cap)\n")
    args = ", ".join(f"p{i}" for i in range(n))
    body = "\n".join(f"        v = s{i}(v, p{i}.cap)" for i in range(n))
    lines.append(f"@njit\ndef driver_a(x, out, {args}):\n"
                 f"    for i in range(x.shape[0]):\n        v = x[i]\n{body}\n"
                 f"        out[i] = v\n")
    mod_a, _ = build_module("\n".join(lines), "e11_contam_a")
    dirty = [collections.namedtuple("Params", ["cap"])(float(60 - i)) for i in range(n)]
    mod_a.driver_a(x, o, *dirty)
    for _ in range(20):
        mod_a.driver_a(x, o, *dirty)

    after = timeit_us(lambda: mod_b.driver_b(x, o, pb), 1000)
    return {
        "variant": "contamination",
        "driver_b_before_us": before["median_us"],
        "driver_b_after_us": after["median_us"],
        "ratio": round(after["median_us"] / before["median_us"], 2),
        "driver_b_signatures": len(mod_b.driver_b.signatures),
        "detail": {"before": before, "after": after},
    }


def part5_bundling_child(mode: str, n: int = N_FUNCS) -> dict:
    """Is the per-NamedTuple-argument cost avoidable by bundling?

    flat   : one NamedTuple with n fields, passed as ONE argument.
    nested : one outer NamedTuple of n inner per-step bundles (keeps doc 03
             section 4.1 namespacing) -- inner classes uniquely named, or all
             named `Params` to test whether nesting re-triggers the collision.
    """
    lines = ["import numpy as np", "from numba import njit", ""]
    for i in range(n):
        lines.append(f"@njit\ndef s{i}(x, cap):\n    return min(x, cap) if x > 0 else x\n")
    if mode == "flat":
        body = "\n".join(f"        v = s{i}(v, p.cap_{i})" for i in range(n))
    else:
        body = "\n".join(f"        v = s{i}(v, p.s{i}.cap)" for i in range(n))
    lines.append(f"@njit\ndef driver(x, out, p):\n"
                 f"    for i in range(x.shape[0]):\n        v = x[i]\n{body}\n"
                 f"        out[i] = v\n")
    mod, path = build_module("\n".join(lines), f"e11_bundling_{mode}")

    if mode == "flat":
        Outer = collections.namedtuple("PipelineParams", [f"cap_{i}" for i in range(n)])
        p_arg = Outer(*[float(60 - i) for i in range(n)])
    else:
        naive = mode == "nested_naive"
        inners = [collections.namedtuple("Params" if naive else f"s{i}_Params", ["cap"])
                  for i in range(n)]
        Outer = collections.namedtuple("PipelineParams", [f"s{i}" for i in range(n)])
        p_arg = Outer(*[inners[i](float(60 - i)) for i in range(n)])

    x = np.array([50.0]); o = np.zeros(1)
    t0 = time.perf_counter()
    mod.driver(x, o, p_arg)
    compile_s = round(time.perf_counter() - t0, 3)
    for _ in range(50):
        mod.driver(x, o, p_arg)
    timing = timeit_us(lambda: mod.driver(x, o, p_arg), DISPATCH_REPEATS)
    return {
        "variant": f"bundled_{mode}",
        "n_bundles": n,
        "n_driver_args_for_params": 1,
        "compile_and_first_call_s": compile_s,
        "driver_signatures": len(mod.driver.signatures),
        "result": o.tolist(),
        "dispatch": timing,
        "module_path": path,
    }


def part5(python: str, script: str) -> dict:
    results = {}
    jobs = (list(VARIANTS) + ["scalar_args_control", "contamination",
            "regen_memoised", "regen_not_memoised"]
            + [f"sweep:{v}:{n}" for v in ("naive_same_name", "unique_class_name")
               for n in (2, 5, 10, 40)]
            + ["bundling:flat", "bundling:nested_unique", "bundling:nested_naive",
               "bundling:flat:40", "bundling:flat:5"])
    for v in jobs:
        proc = subprocess.run(
            [python, script, "--child", v],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if proc.returncode != 0:
            results[v] = {"could_not_run": proc.stderr[-1500:]}
            continue
        try:
            results[v] = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            results[v] = {"could_not_run": proc.stdout[-1500:]}
    return results


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0, help="run one part only")
    ap.add_argument("--child", type=str, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.child:
        if a.child == "regen_memoised":
            out = part5_regen_child(True)
        elif a.child == "regen_not_memoised":
            out = part5_regen_child(False)
        elif a.child == "contamination":
            out = part5_contaminate()
        elif a.child.startswith("bundling:"):
            _parts = a.child.split(":")
            out = part5_bundling_child(_parts[1],
                                       n=int(_parts[2]) if len(_parts) > 2 else N_FUNCS)
        elif a.child.startswith("sweep:"):
            _, v, n = a.child.split(":")
            out = part5_child(v, n=int(n))
            out["variant"] = f"{v}@n={n}"
        else:
            out = part5_child(a.child)
        if "dispatch" in out and out.get("n_bundles"):
            out["us_per_bundle"] = round(
                out["dispatch"]["median_us"] / out["n_bundles"], 3)
        print(json.dumps(out))
        return

    script = os.path.abspath(__file__)
    python = sys.executable
    report: dict[str, Any] = {}
    parts = [a.part] if a.part else [1, 2, 3, 4, 5]
    for p in parts:
        t0 = time.perf_counter()
        try:
            if p == 1:
                report["part1_njit_tolerates_sentinel_default"] = part1()
            elif p == 2:
                report["part2_generated_vs_handwritten_model"] = part2()
            elif p == 3:
                report["part3_retune_vs_retype"] = part3()
            elif p == 4:
                report["part4_direct_python_call"] = part4()
            elif p == 5:
                report["part5_per_function_bundle_dispatch"] = part5(python, script)
        except Exception:  # noqa: BLE001
            report[f"part{p}"] = {"could_not_run": traceback.format_exc()[-2000:]}
        report[f"part{p}_wall_s"] = round(time.perf_counter() - t0, 2)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
