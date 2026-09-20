"""Campaign trees evaluation package.

This __init__.py runs on import and sets up the Python path for decider2.
"""
import sys
from pathlib import Path

# Add the decider2 package to sys.path (must use absolute path)
_decider2_pkg = Path(__file__).parent.parent.parent.resolve() / "decider2"
if str(_decider2_pkg) not in sys.path:
    sys.path.insert(0, str(_decider2_pkg))
