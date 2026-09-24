"""Adds this project's own directory to `sys.path` (the same convention
00/02/05/06/07's own `tests/conftest.py` use) -- `credit_core`, `assessment`,
`business_nested` and `limit_mgmt` still need PYTHONPATH set (see SERVE.md);
this only makes `import business_credit_e2e`, `import pipeline`, `import
inference` resolve from a `pytest` invocation run from anywhere."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
