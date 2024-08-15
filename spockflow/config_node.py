import typing
from spockflow.nodes import AliasedVariableNode, VariableNode
from pydantic import Field, ConfigDict

if typing.TYPE_CHECKING:
    from hamilton import node


class ConfigCacheStrategy:
    def __init__(self, max_size: int = 1):
        from threading import RLock
        from cachetools import LRUCache
        from collections import defaultdict

        self.lock = RLock()
        self.cache = defaultdict(lambda: LRUCache(max_size))

    def query(self, name: str, params: typing.Any):
        h = self.hash(params)
        try:  # Better to use try catch to avoid time of check time of read error
            return self.cache[name][h]
        except KeyError:
            return None

    def update(self, name, params, value):
        h = self.hash(params)
        with self.lock:
            if h not in self.cache[name]:
                self.cache[name][h] = value

    def hash(self, params): ...


class ShallowConfigCache(ConfigCacheStrategy):
    def hash(self, params):
        return id(params)


class NopConfigCache(ConfigCacheStrategy):
    def __init__(self, max_size: int = 1):
        pass

    def query(self, name: str, params: typing.Any):
        return None

    def update(self, name, params, value):
        pass


class InherentConfigCache(ConfigCacheStrategy):
    def hash(self, params):
        return hash(params)


T = typing.TypeVar("T", bound=VariableNode)


class ConfigVariableNode(VariableNode, typing.Generic[T]):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    node_class: typing.Type[T]
    config_path: str
    cache: ConfigCacheStrategy = Field(
        default_factory=ShallowConfigCache, alias="caching_strategy"
    )

    def _generate_nodes(
        self, name: str, config: "typing.Dict[str, typing.Any]"
    ) -> "typing.List[node.Node]":
        g = self.cache.query(name, config)
        if g is None:
            g = (
                self.load(config)
                ._set_module(self._module)
                ._set_name(self._name)
                ._generate_nodes(name, config)
            )
            self.cache.update(name, config, g)
        return g

    def load(self, config: "typing.Dict[str, typing.Any]"):
        from spockflow._util import index_config_path

        return self.node_class(**index_config_path(config, self.config_path))

    def alias(self) -> "AliasedVariableNode":
        # This cannot be done at this point in time
        raise NotImplementedError()
