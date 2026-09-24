from __future__ import annotations
"""Core credit library capabilities."""
import uuid
from datetime import date
from typing import Any, Optional

__version__ = "0.0.1"


def generate_decision_id() -> str:
    """Generate a stable, globally unique decision identifier (09 §5.15 item 1)."""
    return str(uuid.uuid4())
