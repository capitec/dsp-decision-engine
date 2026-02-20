import typing as t
from omegaconf._utils import _DEFAULT_MARKER_
from .context import _config_source_provider
from omegaconf import Container


def resolve_parameter(
    source: str,
    key: str,
    default: t.Any = _DEFAULT_MARKER_,
    *args,
    _parent_: Container,
) -> t.Any:
    _config_source_provider_instance = _config_source_provider.get()
    if _config_source_provider_instance is not None:
        return _config_source_provider_instance.resolve(source_name=source, key=key, args=args)
    if default is _DEFAULT_MARKER_:
        ## TODO should we consider checking if we can resolve static parameters here from the global context
        # I think it might be overkill as there should be no cases we are here and there is no resolver instance
        raise ValueError(f"Parameter source '{source}' not found and no default value provided.")
    return default
