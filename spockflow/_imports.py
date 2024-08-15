import importlib
from .exceptions import RequiredOptionalDependencyError

# TODO see if its possible to extract this automatically
OPTIONAL_DEPS = {}


def assert_module_exists(dep: str):
    try:
        importlib.import_module(dep)
    except ImportError as e:
        package_name = OPTIONAL_DEPS.get(dep, dep)
        raise RequiredOptionalDependencyError(dep, package_name) from e
