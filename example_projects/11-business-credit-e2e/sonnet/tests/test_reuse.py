"""The importlib-by-path loader (business_credit_e2e/reuse.py) resolves each
sibling project's pipeline.py without colliding on the shared `pipeline`
module name -- the central mechanism the whole project depends on."""
from __future__ import annotations

from business_credit_e2e import reuse


def test_project02_pipeline_loads():
    module = reuse.load_project02_pipeline()
    assert callable(module.evidence_unit)
    assert callable(module.capacity_unit)


def test_project05_pipeline_loads_and_builds():
    module = reuse.load_project05_pipeline()
    pipeline = module.build()
    assert pipeline is not None


def test_project07_pipeline_loads_and_builds():
    module = reuse.load_project07_pipeline()
    from limit_mgmt.matrix import build_matrix_table
    pipeline = module.build(build_matrix_table())
    assert pipeline is not None


def test_project06_pipeline_loads():
    """06 is `not_reached` in this slice (reuse_inventory.py), but the loader
    itself must work -- proving the gap is a scope choice, not a broken import."""
    module = reuse.load_project06_pipeline()
    assert callable(module.build)


def test_loaders_are_cached_not_reimported_each_call():
    a = reuse.load_project05_pipeline()
    b = reuse.load_project05_pipeline()
    assert a is b
