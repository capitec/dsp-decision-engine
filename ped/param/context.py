import contextlib
from contextvars import ContextVar
from .param_source import ParameterSourceProvider


_config_source_provider: ContextVar[ParameterSourceProvider] = ContextVar('config_source_provider', default=None)


@contextlib.contextmanager
def config_source_provider(
    provider: ParameterSourceProvider,
):
    token = _config_source_provider.set(
        # This makes a copy which is very important for isolation between requests
        provider
    )
    try:
        yield
    finally:
        _config_source_provider.reset(token)
