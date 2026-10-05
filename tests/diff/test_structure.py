"""structure: steps added, removed, renamed and changed; params; wiring."""
from decider import diff, flow, param, step


def net(a: float, b: float) -> float:
    return a - b


def scaled(net: float) -> float:
    return net * 2.0


def tripled(net: float) -> float:
    return net * 3.0


def capped(scaled: float, cap: float = param(10.0)) -> float:
    return min(scaled, cap)


def capped_higher(scaled: float, cap: float = param(20.0)) -> float:
    return min(scaled, cap)


def test_identical_pipelines_have_an_empty_diff():
    s = diff.structure(flow(net, scaled, name="t"), flow(net, scaled, name="t"))
    assert s.identical and not (s.added or s.removed or s.renamed or s.moved or s.changed_steps or s.params or s.edges)


def test_an_added_step_is_reported_with_its_wiring():
    s = diff.structure(flow(net, name="t"), flow(net, scaled, name="t"))
    assert [r.path for r in s.added] == ["t/scaled"]
    assert not s.identical
    assert [(e.change, e.edge.from_path, e.edge.to_path) for e in s.edges] == [("added", "t/net", "t/scaled")]


def test_a_removed_step_is_reported():
    s = diff.structure(flow(net, scaled, name="t"), flow(net, name="t"))
    assert [r.path for r in s.removed] == ["t/scaled"]


def test_a_changed_constant_is_reported_as_changed_code():
    s = diff.structure(flow(net, scaled, name="t"), flow(net, scaled.__class__ and tripled, name="t"))
    assert not s.identical
    assert s.added or s.changed_steps


def test_a_step_with_a_durable_id_is_matched_through_a_rename():
    def doubled(net: float) -> float:
        return net * 2.0

    old = flow(step(output="x", id="0123abcdef45")(scaled), name="t")
    new = flow(step(output="x", id="0123abcdef45", name="renamed")(scaled), name="t")
    s = diff.structure(old, new)
    assert [(m.old.name, m.new.name) for m in s.renamed] == [("scaled", "renamed")]
    assert not s.added and not s.removed and s.identical


def test_a_changed_param_default_is_reported_per_param():
    old = flow(net, scaled, step(output="out", id="0123abcdef45")(capped), name="t")
    new = flow(net, scaled, step(output="out", id="0123abcdef45")(capped_higher), name="t")
    s = diff.structure(old, new)
    assert [(p.name, p.field, p.old, p.new) for p in s.params] == [("cap", "default", "10.0", "20.0")]
    assert [c.params for c in s.changed_steps] == [True]
