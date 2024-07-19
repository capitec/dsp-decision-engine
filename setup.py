import logging
from setuptools import setup

logger = logging.getLogger(__name__)

try:
    setup(
        use_scm_version={
            "write_to": "spockflow/_version.py",
            "write_to_template": '__version__ = "{version}"\n',
            "git_describe_command": 'git describe --long --tags --match "*.*"',
        }
    )
except LookupError as e:
    logger.error(e)
    exit(1)
