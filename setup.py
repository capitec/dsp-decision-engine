import os
import versioneer
from setuptools import setup

__version__ = os.environ.get("PIP_SPOCKFLOW_VERSION", None)

if __version__ is None:
    try:
        import sys

        sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
        from spockflow import __version__
    except ImportError:
        pass
if __version__ is None:
    try:
        from versioneer import get_version

        __version__ = get_version()
    except ImportError:
        pass

if __version__ is None:
    __version__ == "unknown"

setup(
    version=__version__,
    cmdclass=versioneer.get_cmdclass(),
)
