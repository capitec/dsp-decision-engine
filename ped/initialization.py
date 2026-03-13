"""
This module handles loading any external extensions to ped
"""
import glob
import importlib
import logging
import sys
from pathlib import Path



logger = logging.getLogger(__name__)


def initialize_ped() -> None:
    """Initialise PED by loading extensions from the configured extension path
    and importing any explicitly listed extension modules.
    """
    from .settings import settings
    ext_settings = settings.ext
    ext_path = Path(ext_settings.extension_path).resolve()

    # Add the extension path to sys.path so packages inside it are importable
    ext_path_str = str(ext_path)
    if ext_path_str not in sys.path:
        sys.path.insert(0, ext_path_str)
        logger.debug("Added extension path to sys.path: %s", ext_path_str)

    # Discover and import all packages found at {ext_path}/*/__init__.py
    for init_file in glob.glob(str(ext_path / "*" / "__init__.py")):
        module_name = Path(init_file).parent.name
        logger.debug("Initialising extension module: %s", module_name)
        importlib.import_module(module_name)

    # Import any explicitly listed extension modules
    for module_name in ext_settings.extension_imports:
        logger.debug("Initialising extension import: %s", module_name)
        importlib.import_module(module_name)
