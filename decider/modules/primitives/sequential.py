import typing as t

import polars as pl
from pydantic import field_validator

from decider.types import TInputType, TOutputType
from decider.modules.core import BaseModule, BaseExecuteModule


if t.TYPE_CHECKING:
    from decider.executor import Executor


class ConfigRef(BaseExecuteModule):
    """Resolves a module from the current VersionedConfig by key and executes it.

    Enables composition without inlining: store each component as a separate
    file under configs/{version}/{key}.json, then reference it by key:

        {"type": "config_ref", "key": "bureau_source"}

    The VersionedConfig is read from the context variable set by the config
    manager.  This means ConfigRef only works when a versioned config has been
    loaded (e.g. via JsonFileConfigManager).

    Typical use inside a sequential chain:
        {
          "type": "sequential",
          "name": "01_init_afs",
          "steps": [
            {"type": "config_ref", "name": "step1", "key": "initialise_afs"},
            {"type": "config_ref", "name": "step2", "key": "bureau_source"},
            ...
          ]
        }
    """

    type: t.Literal["config_ref"]
    key: str

    def _resolve(self) -> BaseModule:
        from decider.config.versioned import get_current_versioned_config
        from decider.modules._ext import GraphModule

        versioned_config = get_current_versioned_config()
        if versioned_config is None:
            raise RuntimeError(
                f"ConfigRef(key={self.key!r}): no VersionedConfig in context. "
                "Load a config via a ConfigManager before executing."
            )
        config_data = versioned_config.config.get(self.key)
        if config_data is None:
            available = list(versioned_config.config.keys())
            raise KeyError(
                f"ConfigRef(key={self.key!r}): key not found in VersionedConfig. "
                f"Available keys: {available}"
            )
        return GraphModule.model_validate(config_data).root

    def execute(self, inputs: TInputType, executor: "Executor") -> TOutputType:
        return self._resolve()(inputs, executor=executor)


class SequentialModule(BaseExecuteModule):
    """Chains modules so each step receives the previous step's output as 'input'.

    Created via the | operator:  mod_a | mod_b | mod_c
    """

    type: t.Literal["sequential"]
    steps: t.List[t.Any]  # BaseModule; use Any to allow discriminated deserialisation

    @field_validator("steps", mode="before")
    @classmethod
    def _deserialise_steps(cls, v: t.Any) -> t.List[BaseModule]:
        from decider.modules._ext import GraphModule
        result = []
        for item in v:
            if isinstance(item, BaseModule):
                result.append(item)
            elif isinstance(item, dict):
                result.append(GraphModule.model_validate(item).root)
            else:
                result.append(item)
        return result

    def _compute_input_frame_keys(self) -> t.List[str]:
        return self.steps[0].get_input_frame_keys() if self.steps else ["input"]

    def execute(self, inputs: TInputType, executor: "Executor") -> TOutputType:
        frames: t.Dict[str, pl.LazyFrame] = {
            k: v.lazy() if isinstance(v, pl.DataFrame) else v
            for k, v in inputs.items()
        }
        _input = frames.get("input")
        current = _input if _input is not None else next(iter(frames.values()))

        for step in self.steps:
            frames["input"] = current
            current = step(frames, executor=executor)

        return current

    def __or__(self, other: BaseModule) -> "SequentialModule":
        return SequentialModule(name=self.name, steps=self.steps + [other])
