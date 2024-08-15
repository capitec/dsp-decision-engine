import versioneer
from setuptools import setup

try:
    from spockflow import __version__
except ImportError:
    from versioneer import get_version

    __version__ = get_version()

setup(
    version=__version__,
    cmdclass=versioneer.get_cmdclass(),
)
