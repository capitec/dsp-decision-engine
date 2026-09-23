"""The graph->runtime params seam: doc 03 §4.1, §4.3, §10.

These are regression tests for four defects the parallel build left at that
seam, every one of which returned a *wrong answer with no signal* — doc 03
§2.1's "no error, different decisions ... the worst failure mode the design
can have". They are grouped in their own file because the bug was never in
one layer: `graph` knew which module owned which knob, `runtime` did the
validating, and the tuple passed between them carried step names, so the
module-instance namespace doc 03 §10 promises did not survive the crossing.

The flagship corpus cannot catch any of this: every module in it is a
bare-function module whose single step is named after the module (§5.3), so
step-name and module-name namespacing coincide exactly.
"""
import polars as pl
import pytest

from decider2 import flow, module, param


FRAME = pl.DataFrame({"term_cap": [60.0, 60.0]})


# Steps live at module scope out of habit from the generated-source kernel,
# which imported each step by `<fn.__module__>.<fn.__name__>` and so could
# not fuse a step defined inside a factory function. `decider2.compile.
# kernel` closes over the step's own dispatcher instead, so that limitation
# is gone (tests/test_compile_kernel.py covers the closure case); nothing
# here depends on module scope any more.


def cap(term_cap: float, cap: float = param(48.0, ge=6, le=60)) -> float:
    """Cap the term."""
    return min(term_cap, cap)


def net_of_floor(term_cap: float, floor: float = param(10.0)) -> float:
    """Subtract the floor."""
    return term_cap - floor


def plus_headroom(net_of_floor: float, headroom: float = param(3.0)) -> float:
    """Add headroom."""
    return net_of_floor + headroom


def _capper():
    return cap


def _band_module():
    """A genuine multi-step module: neither step is named after the module,
    and the two knobs are declared on different steps but share one flat
    namespace (doc 03 §4.1)."""
    return module(net_of_floor, plus_headroom, name="band")


# --- doc 03 §4.1 / §10: namespaced by module instance, not by step ---------


def test_a_multi_step_module_is_tuned_through_its_module_name():
    pipe = flow(_band_module())
    assert pipe.params_schema() == {"band": {"floor": 10.0, "headroom": 3.0}}
    assert pipe.apply(FRAME)["plus_headroom"].to_list() == [53.0, 53.0]

    tuned = pipe.apply(FRAME, params={"band": {"floor": 20.0}})
    assert tuned["plus_headroom"].to_list() == [43.0, 43.0]


def test_a_knob_reaches_the_step_that_declared_it_not_its_sibling():
    """Both steps sit in one namespace; each must still receive only its own
    declared fields (doc 03 §4.1), or numba sees an argument the function
    does not take."""
    pipe = flow(_band_module())
    out = pipe.apply(FRAME, params={"band": {"floor": 20.0, "headroom": 5.0}})
    assert out["plus_headroom"].to_list() == [45.0, 45.0]


def test_score_is_namespaced_the_same_way_as_apply():
    pipe = flow(_band_module())
    assert pipe.score({"term_cap": 60.0}, params={"band": {"floor": 20.0}})[
        "plus_headroom"
    ] == 43.0


# --- doc 03 §10: a misspelling is a hard error, never silence --------------


def test_an_unknown_params_namespace_is_a_hard_error_with_a_suggestion():
    pipe = flow(_capper())
    with pytest.raises(ValueError, match="no module instance 'kap'") as exc:
        pipe.apply(FRAME, params={"kap": {"cap": 12.0}})
    assert "Did you mean: cap?" in str(exc.value)


def test_a_misspelled_field_inside_a_known_namespace_is_a_hard_error():
    pipe = flow(_capper())
    with pytest.raises(Exception, match="extra_forbidden|Extra inputs"):
        pipe.apply(FRAME, params={"cap": {"capp": 12.0}})


# --- doc 03 §4.3: bind() freezes a runtime value, it does not vanish ------


def test_a_bound_value_changes_the_answer():
    """§4.3: "A bound value stays a runtime value with a fixed default."
    Previously `bound` lived only on the Module and never reached the kernel,
    so `.bind(cap=12.0)` reported correctly and computed 48.0."""
    bound = module(_capper(), name="capper").bind(cap=12.0)
    assert flow(bound).apply(FRAME)["cap"].to_list() == [12.0, 12.0]


def test_a_bound_value_leaves_the_caller_facing_interface():
    """§4.3: "`cap` leaves the caller-facing params interface." """
    bound = module(_capper(), name="capper").bind(cap=12.0)
    assert bound.params_schema() == {}


def test_overriding_a_bound_value_is_an_error_not_a_silent_win():
    """A caller naming a frozen knob is working from a stale interface
    (§4.3), so neither answer — honouring it or ignoring it — is safe to
    give silently."""
    bound = module(_capper(), name="capper").bind(cap=12.0)
    with pytest.raises(ValueError, match="frozen by capper.bind"):
        flow(bound).apply(FRAME, params={"capper": {"cap": 36.0}})


def test_a_bound_value_is_still_validated_by_the_model():
    """It is a params field like any other: `le=60` still applies."""
    with pytest.raises(Exception):
        flow(module(_capper(), name="capper").bind(cap=999.0)).apply(FRAME)


def test_binding_one_knob_leaves_its_siblings_tunable():
    bound = _band_module().bind(floor=20.0)
    assert bound.params_schema() == {"headroom": 3.0}
    out = flow(bound).apply(FRAME, params={"band": {"headroom": 5.0}})
    assert out["plus_headroom"].to_list() == [45.0, 45.0]


# --- doc 03 §4.3: binding must not trigger a recompile --------------------


def test_binding_does_not_change_the_compiled_signature_count(tmp_path):
    """§4.3, verbatim: "binding never triggers recompilation and remains
    cheap" — a bound value is a kernel argument, not a literal in generated
    source (doc 05 §4.2)."""
    from decider2.compile.driver import build_driver

    def build(mod):
        steps, group_ids, _, _ = flow(mod).flatten_for_runtime()
        return build_driver(
            list(steps), list(group_ids), build_dir=tmp_path,
            terminal_names=frozenset({"cap"}),
        )

    plain = build(module(_capper(), name="capper"))
    bound = build(module(_capper(), name="capper").bind(cap=12.0))
    assert [s.plan.group_name for s in plain.segments if s.plan] == [
        s.plan.group_name for s in bound.segments if s.plan
    ]
