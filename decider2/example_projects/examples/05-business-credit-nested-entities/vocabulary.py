"""Project vocabulary map (doc 03 s5.2, layer 2).

`core.*` was written for *an applicant*. Here six of its capabilities are
consumed *per entity*, and the names collide with the business-level ones: the
library's `score` means the applicant's score, and this project has forty of
them plus one for the business.

Two things this file settles, both of which doc 03 s5.2 has a mechanism for and
neither of which it anticipated being *grain*-shaped:

1. A systematic prefix family per grain. Library names arrive unqualified; the
   grain they land at disambiguates them. `score` at the Entity grain and
   `score` at the Application grain are different values in different scopes,
   so they do **not** need renaming -- the grain is the namespace. This file
   only handles the cases where a library name must be pinned to a grain that
   is not its natural one.

2. The de-duplication identity. `core.bureau` keys on `client_id`; entities
   have no `client_id` until they are matched, and the identity this project
   de-duplicates on is `entity_key` (s4.2). One declaration, not 40 relabels.

Without this file the workaround is an identity-passthrough step per mismatch.
Doc 01 s5.1 counted 79 of those in one project. At a 40-entity fan-out they are
not 79, they are 79 x the number of places a grain shift happens.
"""

from decider2 import Vocabulary
from grains import Application, Entity, Event

vocabulary = Vocabulary(
    # ---- explicit pairs: library name -> project name ---------------------
    {
        "client_id": "entity_key",              # core.bureau, core.exposure
        "applicant_age_years": "entity_age_years",
        "gross_monthly_income": "entity_declared_income",
    },
    # ---- systematic families ---------------------------------------------
    prefixes={
        "bureau_": "entity_bureau_",            # core.bureau, consumed per entity
    },
    # ---- grain pinning ----------------------------------------------------
    # A library capability written for "one applicant" is pinned to the grain
    # it is consumed at. `core.adverse_events` is the interesting one: it was
    # written to classify one event for one applicant, and it is consumed here
    # at the Event grain with its thresholds broadcast down from the Entity
    # grain two levels up. See FRAMEWORK-DEMANDS D07.
    grains={
        "core.bureau": Entity,
        "core.scorecard": Entity,
        "core.calibration": Entity,
        "core.risk_grade": Entity,          # and again at Application -- used twice
        "core.adverse_events": Event,
        "core.exposure": Application,
        "core.appetite": Application,
        "core.fees": Application,
        "core.instalment": Application,     # see pricing/price_one.py: re-pinned
        "core.rate_card": Application,      # to Candidate at the call site
        "core.reason_codes": Application,
    },
)
