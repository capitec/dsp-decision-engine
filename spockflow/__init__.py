__version__ = "UNKNOWN"
try:
    from ._version import __version__ as _version__version__
    __version__ = _version__version__
except ImportError:
    try:
        from importlib.metadata import version
        __version__ = version(__name__)
    except Exception:
        pass