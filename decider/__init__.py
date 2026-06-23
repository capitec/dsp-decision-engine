from .initialization import initialize_decider

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("decider")
except PackageNotFoundError:
    __version__ = "unknown"
