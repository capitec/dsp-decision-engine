import numpy as np
import pytest

from decider.engine.compile import compile_plan, default_bundle


def run_plan(plan, inputs, *, fuse=True, params=None, valid=None, units=None):
    """Run every compiled unit of `plan` in order; returns (outputs by name, values by version id)."""
    units = compile_plan(plan, fuse=fuse) if units is None else units
    leaves = {v.name: v for v in plan.versions if v.producer is None}
    values = {leaves[k].id: np.asarray(a) for k, a in inputs.items() if k in leaves}
    masks = {leaves[k].id: np.asarray(m) for k, m in (valid or {}).items()}
    bundles = {c.id: default_bundle(c.node)._replace(**(params or {}).get(c.node.origin.path, {}))
               for c in plan.calls if c.node.params}
    n = len(next(iter(inputs.values())))
    for call in plan.calls:
        unit = units.get(call.id)
        if unit is not None and unit.calls[0] is call:
            unit.run(values, masks, bundles, n)
    return {name: values[v.id] for name, v in plan.outputs.items() if v.id in values}, values


@pytest.fixture
def run():
    return run_plan
