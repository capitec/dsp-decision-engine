"""Drive decider2's REAL tree codegen path end to end, from a schema `Tree`
document to a compiled, callable batch kernel — mirroring what
`decider2.compile.driver._try_njit` does to every step it compiles:

    fn = njit(cache=True)(step.fn)
    fn.compile(tuple(signature))   # explicit signature, no lazy dispatch

`decider2.trees.codegen.emit_tree()` and `decider2.compile.cache.get_or_build()`
are called completely unmodified, straight out of `decider2/src/`. The only
thing this module adds that production doesn't have here is a tiny
row-loop wrapper, because decider2's own `compile.driver` normally supplies
that (by fusing a tree step into a pipeline's compiled row loop) and we are
deliberately not standing up a whole pipeline just to drive one tree over
one column matrix. That wrapper is timed and reported SEPARATELY
(`wrapper_compile_s`) from the tree's own compile time, so it never hides
inside the headline "codegen compile" number.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from numba import float64, int32, int64, njit

from tree_shapes import TreeShape, collect_string_patterns, to_schema_tree

BUILD_DIR = Path(__file__).parent / "_generated"


@dataclass
class CodegenCompiled:
    emitted: object
    row_loop: object
    timings: dict = field(default_factory=dict)
    emitted_lines: int = 0


def compile_codegen_tree(shape: TreeShape, *, line_cap: int = 100_000) -> CodegenCompiled:
    """Full cold-start path: schema doc -> source -> file -> import -> njit
    -> explicit-signature compile -> a callable batch row loop.

    `line_cap` defaults far above `decider2.trees.codegen.LINE_CAP` (500)
    so the depth-7/9 full-binary shapes can be measured at all — doc 05 §7's
    own cap would reject them outright in production (`TreeTooLarge`); that
    refusal is itself reported by run_experiment.py, not hidden by raising
    the cap silently.
    """
    from decider2.compile import cache as decider_cache
    from decider2.trees import emit_tree

    timings: dict = {}

    t0 = time.perf_counter()
    schema_tree = to_schema_tree(shape)
    emitted = emit_tree(schema_tree, name=shape.name, line_cap=line_cap)
    timings["emit_source_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cached = decider_cache.get_or_build(emitted.source, BUILD_DIR)
    timings["write_and_import_s"] = time.perf_counter() - t0

    string_patterns = collect_string_patterns(shape)

    # -- matcher fns: (feature_code: int32, pattern_codes...: int32) -> int64
    matcher_fns: dict = {}
    t0 = time.perf_counter()
    for feature, fn_name in zip(emitted.string_features, emitted.matcher_fn_names):
        raw = getattr(cached.module, fn_name)
        patterns = string_patterns[feature]
        dispatcher = njit(cache=True)(raw)
        dispatcher.compile(tuple([int32] * (1 + len(patterns))))
        matcher_fns[feature] = dispatcher
    matcher_compile_s = time.perf_counter() - t0

    # -- path fn: one arg per feature (int64 if string-derived, else
    # float64), then one float64 per threshold, in emitted signature order.
    t0 = time.perf_counter()
    path_raw = getattr(cached.module, emitted.path_fn_name)
    path_fn = njit(cache=True)(path_raw)
    path_sig = []
    for feature in emitted.features:
        path_sig.append(int64 if feature in emitted.string_features else float64)
    float_thresholds = [p.default for p in emitted.params if p.annotation == "float"]
    path_sig += [float64] * len(float_thresholds)
    path_fn.compile(tuple(path_sig))
    path_compile_s = time.perf_counter() - t0

    # -- output fn: (path_result: int64) -> float64
    t0 = time.perf_counter()
    (output_fn_name,) = emitted.output_fn_names
    output_raw = getattr(cached.module, output_fn_name)
    output_fn = njit(cache=True)(output_raw)
    output_fn.compile((int64,))
    output_compile_s = time.perf_counter() - t0

    timings["matcher_compile_s"] = matcher_compile_s
    timings["path_compile_s"] = path_compile_s
    timings["output_compile_s"] = output_compile_s
    timings["tree_compile_s"] = matcher_compile_s + path_compile_s + output_compile_s

    # -- the row-loop wrapper: harness glue, not part of the tree's own
    # emitted source. Bakes threshold defaults and pattern codes as
    # generated-source constants purely to keep this wrapper's own
    # signature fixed (`(feat_matrix, str_matrix, out)`); the tree's OWN
    # kernel (path_fn) still takes every threshold as a real argument,
    # which is the property doc 05 §4.2 actually cares about.
    t0 = time.perf_counter()
    row_loop = _build_row_loop(emitted, path_fn, output_fn, matcher_fns, float_thresholds, string_patterns, shape)
    timings["wrapper_build_and_compile_s"] = time.perf_counter() - t0

    return CodegenCompiled(emitted=emitted, row_loop=row_loop, timings=timings, emitted_lines=emitted.emitted_lines)


def _build_row_loop(emitted, path_fn, output_fn, matcher_fns, float_thresholds, string_patterns, shape: TreeShape):
    ns: dict = {"path_fn": path_fn, "output_fn": output_fn}
    lines = ["def row_loop(feat_matrix, str_matrix, out):"]
    lines.append("    n = feat_matrix.shape[0]")
    lines.append("    for i in range(n):")

    fcol = 0
    scol = 0
    call_args = []
    for feature in emitted.features:
        if feature in emitted.string_features:
            mfn_name = f"_m{scol}"
            ns[mfn_name] = matcher_fns[feature]
            pat_codes = [shape.string_categories[feature].index(p) for p in string_patterns[feature]]
            args = ", ".join([f"str_matrix[i, {scol}]"] + [str(c) for c in pat_codes])
            var = f"_v{scol}"
            lines.append(f"        {var} = {mfn_name}({args})")
            call_args.append(var)
            scol += 1
        else:
            call_args.append(f"feat_matrix[i, {fcol}]")
            fcol += 1
    for j, value in enumerate(float_thresholds):
        ns[f"_thr{j}"] = float(value)
        call_args.append(f"_thr{j}")

    lines.append(f"        _path = path_fn({', '.join(call_args)})")
    lines.append("        out[i] = output_fn(_path)")
    src = "\n".join(lines) + "\n"
    exec(compile(src, f"<row_loop:{shape.name}>", "exec"), ns)
    raw = ns["row_loop"]
    # No cache=True here: this wrapper is harness glue with no real file on
    # disk (decider2 itself never does this — every generated driver it
    # compiles goes through `cache.get_or_build` first, which is exactly
    # what the tree's own path/matcher/output functions above went
    # through). On-disk caching only affects a FUTURE process's cold
    # start, not this run's ns/row, so its absence here doesn't distort
    # anything being measured.
    return njit(raw)
