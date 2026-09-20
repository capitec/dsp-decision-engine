"""Pytest configuration for the campaign trees project.

Sets up the Python path to allow imports from the decider2 package.
This runs before test collection, so sys.path is correct.
"""
import sys
from pathlib import Path

# Add the decider2 package directory to sys.path
# Path from conftest: /home/sholto/.../decider2/evaluation/04-campaign-trees/conftest.py
# Want: /home/sholto/.../decider2/decider2/  (the actual package)
# So: parent = /home/sholto/.../decider2/evaluation/04-campaign-trees
#     parent.parent = /home/sholto/.../decider2/evaluation
#     parent.parent.parent = /home/sholto/.../decider2
#     parent.parent.parent / "decider2" = /home/sholto/.../decider2/decider2
#
# BUT we actually want to add /home/sholto/.../decider2 (the shim dir) to sys.path
# So the import "from decider2 import ..." finds /home/sholto/.../decider2/decider2/__init__.py

decider2_shim = Path(__file__).parent.parent.parent.resolve()
if str(decider2_shim) not in sys.path:
    sys.path.insert(0, str(decider2_shim))


def pytest_configure(config):
    """Re-ensure the path is correct before test collection."""
    decider2_shim = Path(__file__).parent.parent.parent.resolve()
    if str(decider2_shim) not in sys.path:
        sys.path.insert(0, str(decider2_shim))
