"""Decision governance and replay harness (spec 09, the 09-H slice).

Operates on the evidence three of the eight live flows emit (01 transaction
fraud, 03 unsecured granting and pricing, 05 business nested entities) --
never on applicants, scorecards or rate cards of its own. See `NOTES.md` for
the exact slice and `pipeline.py` for the one capability wired through
`decider`'s serving path.
"""
