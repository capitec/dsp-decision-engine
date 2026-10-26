"""Control and holdout groups, from a stable hash of identifiers: no randomness, no stored assignment table."""
from __future__ import annotations

import hashlib

from decikit import allow_fallback, missing_as, param


def _unit_interval(*parts: object) -> float:
    # sha256, not hash(): hash() is salted per process, so a client would move group between runs.
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16) / 16**12


@allow_fallback
def is_control(
    client_id: str, campaign_id: int, holdout_design_version: str = missing_as("v1"),
    control_percentage: float = param(0.05, ge=0.0, le=1.0),
) -> bool:
    """Whether the client is in this campaign's control group. A new design version re-draws the group."""
    return _unit_interval("control", client_id, campaign_id, holdout_design_version) < control_percentage


@allow_fallback
def is_universal_holdout(
    client_id: str, universal_holdout_design_version: str = missing_as("v1"),
    universal_holdout_percentage: float = param(0.01, ge=0.0, le=1.0),
) -> bool:
    """Whether the client is held out of every campaign, to measure the whole programme."""
    return _unit_interval("universal", client_id, universal_holdout_design_version) < universal_holdout_percentage
