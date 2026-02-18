import typing as t
import warnings
from dataclasses import dataclass, field
from .sources import ParameterSource


# Compatibility for Python < 3.11
try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass



class VersionMismatchStrategy(StrEnum):
    WARN = 'warn'
    ERROR = 'error'
    IGNORE = 'ignore'


@dataclass
class ParameterSourceProvider:
    sources: dict[str, ParameterSource]
    request: t.Any = None
    requested_versions: dict[str, t.Any] = field(default_factory=dict)
    current_versions: dict[str, t.Any] = field(default_factory=dict)
    kwargs: t.Dict[str, t.Any] = field(default_factory=dict)
    version_mismatch_strategy: VersionMismatchStrategy = VersionMismatchStrategy.ERROR
    
    def requires_refresh(self) -> bool:
        return any(
            source.requires_refresh(
                request=self.request,
                curr_version=self.current_versions.get(source_name),
                requested_version=self.requested_versions.get(source_name),
                **self.kwargs,
            ) 
            for source_name, source in self.sources.items()
        )

    def resolve(self, source_name: str, key: str) -> t.Any:
        if source_name not in self.sources:
            raise ValueError(f"Source '{source_name}' not found in provider.")
        requested_version = self.requested_versions.get(source_name)
        result_version, result = self.sources[source_name].root.get(
            key, 
            request=self.request,
            requested_version=requested_version,
            **self.kwargs,
        )
        if requested_version is None:
            # We pin the version now so the rest of the parameters in the same request will use the same version.
            # This makes use of the fact that the context will always make a copy of the provider so we are safe to 
            # mutate the versions here without affecting other requests.
            self.requested_versions[source_name] = result_version
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
