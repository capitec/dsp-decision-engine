"""collections_treatment: daily treatment assignment for delinquent accounts (spec 08).

Composes `credit_core` (project 00) and project 02's affordability assessment
(ARRANGEMENT mode) into the collections-specific logic spec 08 declares locally
(§4.1): account state, suspensions, the collections score and treatment matrix,
the escalation path over time, arrangement sustainability, and capacity
allocation.

Modules:
    vocab            locally declared vocabulary (§4.1): treatment codes, capacity
                     pools, non-selection reasons, suspension codes, reset events.
    state            account state derivations §4.3 requires be computed, not
                     read: `arrears_bucket_code`, `balance_band_code`.
    suspensions      §5.2: every one of the 20 suspension codes, evaluated
                     unconditionally, individually attributable.
    scoring          §5.3: the collections score, its bands, and the four
                     overlay kinds, all through `core.adjustments`.
    matrix           §5.4: the 5 376-cell treatment matrix and its overlays.
    path             §5.5: the escalation path over time -- episode, resets,
                     rolls, re-entry, as-known-on-the-day vs. as-at-now.
    arrangements     §5.6: arrangement sustainability through project 02's
                     `core.affordability`, ARRANGEMENT mode.
    capacity_alloc   §5.9: population-level ranking, rationing and
                     non-selection reasons, as one `frame_step`.
"""
