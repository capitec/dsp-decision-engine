"""Client-facing wording: generic per action, because saying which rule fired tells the fraudster too."""
from __future__ import annotations

from decider import allow_fallback

SUPPRESSED = "wording.suppressed_no_consent.v1"


@allow_fallback
def client_wording_key(client_wording_key: str, client_visible: bool, channel_permitted: bool) -> str:
    """The wording key for an action the client sees, unless the client's consent forbids the channel."""
    return client_wording_key if channel_permitted or not client_visible else SUPPRESSED
