import typing as t
import warnings
from enum import StrEnum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from .cache import ParameterCache
from .sources import ParameterSource



class AbstractParameterSourceProvider(ABC):
    @abstractmethod
    def with_values_set(self, **kwargs) -> 't.Self':
        ...

    @abstractmethod
    def requires_refresh(self, **kwargs) -> bool:
        ...

    @abstractmethod
    def resolve(self, source_name: str, key: str, **kwargs) -> t.Any:
        ...

class VersionMismatchStrategy(StrEnum):
    WARN = 'warn'
    ERROR = 'error'
    IGNORE = 'ignore'


@dataclass
class ParameterSourceProvider(AbstractParameterSourceProvider):
    sources: dict[str, t.Union[ParameterCache, ParameterSource]]
    request: t.Any = None
    versions: dict[str, t.Any] = field(default_factory=dict)
    kwargs: t.Dict[str, t.Any] = field(default_factory=dict)
    version_mismatch_strategy: VersionMismatchStrategy = VersionMismatchStrategy.ERROR

    def with_values_set(self, **kwargs) -> 't.Self':
        return replace(
            self, 
            request=kwargs.pop('request', self.request), 
            versions=kwargs.pop('requested_versions', self.versions), 
            kwargs=kwargs or self.kwargs
        )
    
    def requires_refresh(self) -> bool:
        return any(
            source.requires_refresh(
                request=self.request,
                requested_version=self.versions.get(source_name),
                **self.kwargs,
            ) 
            for source_name, source in self.sources.items()
        )

    def resolve(self, source_name: str, key: str) -> t.Any:
        if source_name not in self.sources:
            raise ValueError(f"Source '{source_name}' not found in provider.")
        requested_version = self.versions.get(source_name)
        result_version, result = self.sources[source_name].get(
            key, 
            request=self.request,
            requested_version=requested_version,
            **self.kwargs,
        )
        if requested_version is None:
            # We pin the version now so the rest of the parameters in the same request will use the same version.
            # This makes use of the fact that the context will always make a copy of the provider so we are safe to 
            # mutate the versions here without affecting other requests.
            self.versions[source_name] = result_version
        elif (
            self.version_mismatch_strategy != VersionMismatchStrategy.IGNORE and 
            result_version != requested_version
        ):
            warning_text = (
                f"Version mismatch for source '{source_name}': "
                f"requested {requested_version}, got {result_version}"
            )
            # This is important because if we find drift in the versions
            # That means for a single request there is a possibility that half of the 
            # values are using an old version and half are using a new version 
            # which can lead to very subtle and hard to debug issues. 
            if self.version_mismatch_strategy == VersionMismatchStrategy.ERROR:
                raise ValueError(warning_text)
            elif self.version_mismatch_strategy == VersionMismatchStrategy.WARN:
                warnings.warn(warning_text)
        return result
