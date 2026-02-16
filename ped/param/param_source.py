import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace



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


@dataclass
class ParameterSourceProvider(AbstractParameterSourceProvider):
    sources: dict[str, t.Union['CashedParameterSource']]
    request: t.Any = None
    versions: dict[str, t.Any] = field(default_factory=dict)
    kwargs: t.Dict[str, t.Any] = field(default_factory=dict)

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
        return self.sources[source_name].get(
            key, 
            request=self.request,
            requested_version=self.versions.get(source_name),
            **self.kwargs,
        )
