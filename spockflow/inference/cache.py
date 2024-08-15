import typing
from time import monotonic
from threading import RLock
from collections import defaultdict

from cachetools import LRUCache

from .settings import get_settings

if typing.TYPE_CHECKING:
    from spockflow.inference.model_loader import VersionedModel, ModelLoader
    from spockflow.core import Driver


class LatestVersionTTLCache:
    class _Unset:
        pass

    def __init__(self, ttl):
        self.ttl = ttl
        self.item = self._Unset
        self.version = self._Unset
        self.expires = float("inf")

    def get(self):
        if self.item == self._Unset:
            raise KeyError()
        if monotonic() > self.expires:
            raise KeyError()
        return self.item

    def set(self, value, version):
        old_item, old_version = self.item, self.version
        self.item = value
        self.version = version
        self.expires = monotonic() + self.ttl
        if old_item == self._Unset:
            return None
        return old_version, old_item


T = typing.TypeVar("T")


class ModelCache(LRUCache, typing.Generic[T]):
    class _Unset:
        pass

    def __init__(self, ttl, maxsize, getsizeof=None):
        super().__init__(maxsize, getsizeof)
        self.latest_item_cache = LatestVersionTTLCache(ttl)
        self.lock = RLock()

    def __getitem__(self, key) -> T:
        if key is None:
            return self.latest_item_cache.get()
        return super().__getitem__(key)

    def update(self, model: "VersionedModel"):
        value, version = model.model, model.version
        if model.is_latest:
            old_model = self.latest_item_cache.set(value, version)
            if old_model is None:
                return
            version, value = old_model
        self[version] = value


class ModelCacheManager:
    def __init__(
        self, model_loader: "ModelLoader", max_size=None, latest_ttl=None
    ) -> None:
        if max_size is None:
            max_size = get_settings().config_cache_capacity
        if latest_ttl is None:
            latest_ttl = get_settings().config_cache_latest_ttl

        self.models: typing.Dict[str, ModelCache["Driver"]] = defaultdict(
            lambda: ModelCache(latest_ttl, max_size)
        )
        self.model_loader = model_loader

    def get(self, model_name: str = None, version: str = None):
        if model_name not in self.model_loader:
            if model_name is None:
                raise ValueError(
                    f"No default model specified please specify model to run with {get_settings().server_model_name_header}"
                )
            raise ValueError(f"Could not load model {model_name}")
        try:
            return self.models[model_name][version]
        except KeyError:
            pass  # Cache miss
        model_cache = self.models[model_name]
        with model_cache.lock:
            # Extra check to see if was added by a different thread
            if version not in model_cache:
                versioned_model = self.model_loader.load_model(model_name, version)
                model_cache.update(versioned_model)
                return versioned_model.model
        return self.models[model_name][version]

    def refresh_latest_models(self):
        for model_name, model_cache in self.models.items():
            versioned_model = self.model_loader.load_model(model_name, None)
            model_cache.update(versioned_model)
