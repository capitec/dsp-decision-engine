from .initialization import initialize_ped
initialize_ped()

# I think the below is old and can be deleted just keeping it here for now
# from omegaconf import OmegaConf
# from omegaconf.resolvers import oc
# import typing as t
# from omegaconf import Container
# from omegaconf._utils import _DEFAULT_MARKER_
# from contextvars import ContextVar
# import contextlib

# _active_sources: ContextVar[t.Set[str]] = ContextVar('active_sources', default=set())

# @contextlib.contextmanager
# def capture_sources():
#     sources = set()
#     token = _active_sources.set(sources)
#     try:
#         yield sources
#     finally:
#         _active_sources.reset(token)


# def resolve_parameter(
#     source: str,
#     key: str,
#     default: t.Any = _DEFAULT_MARKER_,
#     *,
#     _parent_: Container,
# ) -> t.Any:
#     from omegaconf._impl import select_value
#     active_sources = _active_sources.get()
#     if active_sources is not None:
#         active_sources.add(source)

#     return select_value(cfg=_parent_, key=key, absolute_key=True, default=default)


# OmegaConf.register_new_resolver(
#     "param",
#     resolve_parameter,
#     use_cache=False,
# )