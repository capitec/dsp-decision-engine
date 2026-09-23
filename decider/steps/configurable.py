from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pydantic import ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from decider.registry import BaseRegistryModule, ref
from decider.steps.base import Step


class ConfigurableStep(Step, BaseRegistryModule, root=True):
    """A serialisable, config-driven step (trees, tables, scorecards): subclass it and implement `to_ir`.

    Every subclass loads by its import path (`"credit.steps:Scaled"`), or by a
    short alias it declares as `type: Literal["scaled"] = "scaled"`. Loading
    an import path only ever returns a registered `ConfigurableStep`. Configs
    are frozen: changing one means a new object, whose IR is built afresh.

    A `Value[T]` field holds a literal (`2.0`) or a `ParamRef`
    (`{"param": "factor", "default": 2.0}`). `ctx.call` passes a literal as
    is and turns a ref into a param, so retuning it never rebuilds anything.
    To build a config from other steps instead, return
    `ctx.expand(self, helper_step)`.

    Example::

        from typing import Literal
        from decider import ConfigurableStep, Value

        def scale(x: float, factor: float) -> float:
            return x * factor

        class Scaled(ConfigurableStep):
            type: Literal["scaled"] = "scaled"
            column: str
            factor: Value[float] = 1.0

            def to_ir(self, ctx):
                return ctx.call(self, scale, inputs={"x": self.column}, values={"factor": self.factor})

        cfg = ConfigurableStep.load({"type": "scaled", "name": "doubled", "column": "income",
                                     "factor": {"param": "factor", "default": 2.0}})
        cfg.run(df)                                                # writes column "doubled"
        cfg.run(df, params={"doubled": {"factor": 3.0}})           # retuned, nothing rebuilt
        ConfigurableStep.load(cfg.model_dump_json()) == cfg        # True
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    # Relabels are placement, set in Python like the pipeline itself, so never dumped.
    reads: SkipJsonSchema[tuple[tuple[str, str], ...]] = Field((), exclude=True, repr=False)
    writes: SkipJsonSchema[tuple[tuple[str, str], ...]] = Field((), exclude=True, repr=False)

    @classmethod
    def load(cls, source: str | Path | Mapping[str, Any]) -> Any:
        """A config from a JSON file path, JSON text or a parsed dict, loaded as the class its `type` names.

        On a subclass the tag may be left out, and must name that subclass (or one of its own) if given.

        Example::

            rules = ConfigurableStep.load("rules/risk.json")
            rule = ConfigurableStep.load({"type": "threshold_rule", "name": "hi", ...})
            same = ConfigurableStep.load(rule.model_dump_json())
        """
        if isinstance(source, Mapping):
            doc = source
        elif isinstance(source, str) and source.lstrip().startswith("{"):
            doc = json.loads(source)
        else:
            doc = json.loads(Path(source).read_text())
        tag = doc.get("type")
        return (cls if tag is None else cls.resolve(tag)).model_validate(doc)

    def _replace(self, **changes: Any) -> Any:
        return self.model_copy(update=changes)


StepRef = ref(ConfigurableStep)
"""A pydantic field type holding any `ConfigurableStep`, chosen by its `type` tag.

Example::

    class RuleSet(ConfigurableStep):
        rules: list[StepRef]

        def to_ir(self, ctx):
            return ctx.expand(self, flow(*self.rules))
"""
