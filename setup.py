import versioneer
from spockflow import __version__
from setuptools import setup

setup(
    version=__version__,
    cmdclass=versioneer.get_cmdclass(),
)