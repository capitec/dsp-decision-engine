"""
Root conftest – adds the project root to sys.path and initialises PED so all
module types (scorecard, decision_table, etc.) are registered before any test
file is collected.
"""
import sys
from pathlib import Path

# Project root is one level above this file (tests/../)
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ped import initialize_ped  # noqa: E402  (must come after sys.path edit)

initialize_ped()
