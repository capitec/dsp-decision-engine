from __future__ import annotations

from typing import Any

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

    Example::

        class Scaled(ConfigurableStep):
            type: Literal["scaled"] = "scaled"
            factor: Value[float]

            def to_ir(self, ctx):
                factor = ctx.value(self.factor, float)
                ...

        cfg = ConfigurableStep.resolve("scaled").model_validate_json(text)
        pipeline = prepare | cfg
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    # Relabels are placement, set in Python like the pipeline itself, so never dumped.
    reads: SkipJsonSchema[tuple[tuple[str, str], ...]] = Field((), exclude=True)
    writes: SkipJsonSchema[tuple[tuple[str, str], ...]] = Field((), exclude=True)

    def _replace(self, **changes: Any) -> Any:
        return self.model_copy(update=changes)


StepRef = ref(ConfigurableStep)
"""A pydantic field type holding any `ConfigurableStep`, chosen by its `type` tag."""
