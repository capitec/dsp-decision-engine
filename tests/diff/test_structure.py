"""structure: steps added, removed, renamed and changed; params; wiring."""
from decikit import flow, param, step
from decikit.lab import diff
from decikit.steps.tables import DecisionTableConfig
from decikit.steps.trees import TreeConfig


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


def grade_table(rate=2.0):
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "grade", "columns": {"g": "String", "rate": "Float64"},
        "rows": [{"g": "a", "rate": 1.0}, {"g": "b", "rate": rate}],
        "expression": {"type": "eq", "variable": "grade", "value_column": "g"},
        "outputs": ["rate"], "default": [0.0]})


def test_the_same_table_pipeline_built_twice_is_identical():
    s = diff.structure(flow(grade_table(), name="t"), flow(grade_table(), name="t"))
    assert s.identical and not s.changed_steps


def test_a_changed_table_row_is_reported_as_changed_consts():
    s = diff.structure(flow(grade_table(), name="t"), flow(grade_table(rate=3.0), name="t"))
    assert not s.identical
    assert [c.consts for c in s.changed_steps] == [True]


def _split_tree(rule_ids, name="rules"):
    leaf = {"type": "leaf", "result_idx": 0}
    rules = [{"meta": {"name": f"R{k}"}, "rule": {"type": "unary", "then": leaf,
                                                   "condition": {"op": ">", "feature": "x", "threshold": k}}} for k in rule_ids]
    return TreeConfig(name=name, id="0123456789ab", part_outputs=4, tree={
        "type": "prioritized_flat_rule", "mode": "all", "rules": rules,
        "output": {"data": [{"hit": True}], "default": {"hit": False}, "dtypes": [["hit", "Boolean"]]}})


def test_a_split_tree_with_a_durable_id_is_matched_part_by_part():
    old = flow(_split_tree(range(10)), name="p")
    same = diff.structure(old, flow(_split_tree(range(10)), name="p"))
    assert (same.added, same.removed, same.changed_steps) == ((), (), ())
    grown = diff.structure(old, flow(_split_tree([*range(10), 99]), name="p"))
    assert (grown.added, grown.removed) == ((), ()) and [c.ref.path for c in grown.changed_steps] == ["p/rules/part_2"]
    renamed = diff.structure(old, flow(_split_tree(range(10), name="checks"), name="p"))
    assert (renamed.added, renamed.removed) == ((), ()) and (len(renamed.renamed), len(renamed.moved)) == (1, 3)
