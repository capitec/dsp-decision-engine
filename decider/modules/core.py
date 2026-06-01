import typing as t
from abc import ABC, abstractmethod
from pydantic import PrivateAttr
from decider.types import TInputType, TOutputType
from decider._ext import TypeDiscriminatedBaseModule

if t.TYPE_CHECKING:
    from decider.executor import Executor, FrameNode, CompiledFrameGraph
    from decider.config.base import BaseConfig
    from decider.config.versioned import VersionedConfig


class BaseModule(TypeDiscriminatedBaseModule, ABC):
    name: str
    _compiled_graph: t.Optional["CompiledFrameGraph"] = PrivateAttr(default=None)


    def compile(self, executor: "Executor") -> None:
        if self._compiled_graph is None:
            frame_nodes = self.get_frame_nodes(executor)
            self._compiled_graph = executor.compile_frame_graph(frame_nodes)
        return self._compiled_graph

    @abstractmethod
    def get_frame_nodes(self, executor: "Executor") -> t.List["FrameNode"]:
        ...

    def __call__(
        self,
        inputs: TInputType,
        executor: t.Optional["Executor"] = None,
    ) -> TOutputType:
        from decider.settings import get_default_executor
        executor = executor or get_default_executor()
        return executor.execute(self, inputs)

    def __or__(self, other: "BaseModule") -> "BaseModule":
        from decider.modules.primitives.sequential import SequentialModule
        if isinstance(self, SequentialModule):
            return SequentialModule(name=self.name, steps=self.steps + [other])  # type: ignore[attr-defined]
        return SequentialModule(name=self.name, steps=[self, other])

    def to_config(self, config_key: str) -> "BaseConfig[t.Self]":
        from decider.config.base import ConfigModule
        config_class = ConfigModule.for_module_class(type(self))
        return config_class.from_model(model=self, config_key=config_key)

    async def save(self, root_key: str, config_manager=None) -> "VersionedConfig":
        from decider.config.base import DUMP_TRIGGER_KEY
        from decider.config.versioned import with_versioned_config
        if config_manager is None:
            from decider.settings import settings
            config_manager = settings.config.get()
        versioned_conf = await config_manager.get()
        with with_versioned_config(versioned_conf):
            config_mod = self.to_config(config_key=root_key)
            config_mod.model_dump(context={DUMP_TRIGGER_KEY: True})
        return versioned_conf


class BaseExecuteModule(BaseModule, ABC):
    name: str

    @abstractmethod
    def execute(self, inputs: TInputType, executor: "Executor") -> TOutputType:
        ...

    def get_frame_nodes(self, executor: "Executor") -> t.List["FrameNode"]:
        from decider.executor import FrameNode
        return [FrameNode(
            name=self.name,
            callable=self.execute,
            depends_on=[],
        )]
