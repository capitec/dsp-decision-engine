import contextlib
from contextvars import ContextVar
from .param_source import AbstractParameterSourceProvider


_config_source_provider: ContextVar[AbstractParameterSourceProvider] = ContextVar('config_source_provider', default=None)


@contextlib.contextmanager
def config_source_provider(
    provider: AbstractParameterSourceProvider,
    # requested_versions: t.Dict[str, t.Any] | None = None,
    # request: t.Any = _UNSET_,
    **kwargs,
):
    token = _config_source_provider.set(
        # This makes a copy which is very important for isolation between requests
        provider.with_values_set(**kwargs)
    )
    try:
        yield
    finally:
        _config_source_provider.reset(token)
