"""Servable campaign targeting pipeline.

Runs the campaign targeting tree as an HTTP endpoint.
Usage:
    cd decider2/evaluation/04-campaign-trees
    /path/to/.venv/bin/python -m decider2 serve serve.py --port 8104

Or:
    /path/to/.venv/bin/python serve.py
"""
from pipeline import pipeline

if __name__ == "__main__":
    # This allows the module to be used directly with:
    # python -m decider2 serve serve.py --port 8104
    pass
