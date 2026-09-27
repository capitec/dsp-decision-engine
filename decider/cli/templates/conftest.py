"""Make the package importable from tests/ under plain `pytest`."""
import sys
from pathlib import Path

# The project directory is the package, so its parent goes on sys.path for `import {{name}}`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
