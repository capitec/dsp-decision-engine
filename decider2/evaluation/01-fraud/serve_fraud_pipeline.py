"""
HTTP serving wrapper for fraud interdiction pipeline.

Usage:
    python -m decider2 serve serve_fraud_pipeline.py --port 8103
"""

from pipelines.fraud_interdiction import pipeline, SharedParams

# Export the pipeline for the serving framework
__all__ = ["pipeline", "SharedParams"]
