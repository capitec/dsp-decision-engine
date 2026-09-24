"""Custom request handler for campaign trees."""

from pipeline import build


def inference():
    """Return the built pipeline."""
    return build()
