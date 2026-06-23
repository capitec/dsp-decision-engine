"""Tests for config save/load round-trip via JsonFileConfigManager."""

import asyncio
import tempfile
from pathlib import Path

import polars as pl
import pytest
from pydantic import BaseModel

from decider.config.file import JsonFileConfigManager
from decider.modules import GraphModule, register_graph_module
from decider.modules.functional import generate_from_functions


# ── helpers ───────────────────────────────────────────────────────────────────

class _RtConfig(BaseModel):
    weight: float = 1.0


def rt_score(amount: pl.Expr, config: _RtConfig) -> pl.Expr:
    return amount * config.weight


_RtScorer = generate_from_functions("rt_scorer", rt_score)
register_graph_module(_RtScorer)


def _make_scorer(weight: float = 2.0):
    return _RtScorer(name="scorer", weight=weight)


# ── save / load ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_creates_json_file():
    with tempfile.TemporaryDirectory() as tmp:
        mgr = JsonFileConfigManager(basepath=tmp)
        scorer = _make_scorer(weight=3.0)
        versioned = await scorer.asave("main", mgr)
        await mgr.save_version(overwrite=True)

        config_path = Path(tmp) / str(versioned.version) / "main.json"
        assert config_path.exists()


@pytest.mark.asyncio
async def test_roundtrip_preserves_config_values():
    with tempfile.TemporaryDirectory() as tmp:
        scorer = _make_scorer(weight=7.0)
        mgr = JsonFileConfigManager(basepath=tmp)
        await scorer.asave("main", mgr)
        await mgr.save_version(overwrite=True)

        fresh = JsonFileConfigManager(basepath=tmp)
        loaded = await fresh.get_latest()
        module = GraphModule.model_validate(loaded.config["main"]).root

        assert module.weight == pytest.approx(7.0)


@pytest.mark.asyncio
async def test_roundtrip_produces_correct_output():
    df = pl.DataFrame({"amount": [1.0, 2.0, 4.0]})

    with tempfile.TemporaryDirectory() as tmp:
        scorer = _make_scorer(weight=5.0)
        mgr = JsonFileConfigManager(basepath=tmp)
        await scorer.asave("main", mgr)
        await mgr.save_version(overwrite=True)

        fresh = JsonFileConfigManager(basepath=tmp)
        loaded = await fresh.get_latest()
        module = GraphModule.model_validate(loaded.config["main"]).root

        result = module({"input": df})
        assert result["rt_score"].to_list() == pytest.approx([5.0, 10.0, 20.0])


@pytest.mark.asyncio
async def test_versioning_increments():
    with tempfile.TemporaryDirectory() as tmp:
        scorer = _make_scorer()
        mgr = JsonFileConfigManager(basepath=tmp)

        v1 = await scorer.asave("main", mgr)
        await mgr.save_version(overwrite=True)

        mgr2 = JsonFileConfigManager(basepath=tmp)
        v2_versioned = await mgr2.create_version()
        scorer2 = _make_scorer(weight=9.0)
        await scorer2.asave("main", mgr2)
        await mgr2.save_version(overwrite=True)

        latest = await JsonFileConfigManager(basepath=tmp).get_latest()
        module = GraphModule.model_validate(latest.config["main"]).root
        assert module.weight == pytest.approx(9.0)


# ── sync wrapper ──────────────────────────────────────────────────────────────

def test_sync_save():
    with tempfile.TemporaryDirectory() as tmp:
        scorer = _make_scorer(weight=4.0)
        mgr = JsonFileConfigManager(basepath=tmp)
        scorer.save("main", mgr)
        asyncio.run(mgr.save_version(overwrite=True))

        fresh = JsonFileConfigManager(basepath=tmp)
        loaded = asyncio.run(fresh.get_latest())
        module = GraphModule.model_validate(loaded.config["main"]).root
        assert module.weight == pytest.approx(4.0)
