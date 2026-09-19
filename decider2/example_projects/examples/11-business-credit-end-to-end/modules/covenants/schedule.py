"""What a schedule IS, structurally. Spec 13-Q5.

    "A covenant instance fires on its own calendar for five years, against a
     definition frozen at its creation, with a step schedule, cure rights and an
     expiry. It is not a rule, not a table and not a parameter. Is it a fourth
     thing, or a rule set with a temporal scope?"

This project's answer: **a schedule is neither. It is a subject generator.**

A schedule produces the ROWS that a perfectly ordinary record-tier pipeline then
decides. It belongs in the frame tier, as a declared expansion with a known
schema transform (doc 02 s5), and the decision logic downstream of it needs no
new concept at all. 2.4 M covenant tests a year are 2.4 M rows, and L2
(modules/covenants/test.py) is a normal module over them.

That answer is worth defending because the alternative -- a `Schedule` node kind
in the graph, holding time and firing -- puts a clock inside the decision engine.
A clock inside a decision engine is unreplayable by construction: you cannot
replay 2029-06-30 by waiting for it. Keeping the calendar in the frame tier
keeps `decision_date` the only time the record tier knows about, which is doc 00
7.3 and 09 5.15 item 4.

What it costs, stated honestly: the expansion is a frame operation with a
non-trivial schema transform (one instance row in, 0..n test rows out, carrying
the binding), and doc 02 s5's declarative-wrapper requirement means `Expand` has
to be a first-class frame op rather than an `@breaks_lineage` escape. It is not
in doc 02's stage-one set (join, aggregate, filter). FRAMEWORK-DEMANDS D12.
"""

from decider2.frame import Expand
from decider2 import param

# --------------------------------------------------------------------------
# One covenant instance -> its due test dates. The calendar rule comes from the
# BOUND definition's `test_date_rule` and the client's financial year end, so
# two instances of the same template on two clients fire on different dates and
# the difference is contractual rather than configured.
#
# Frequencies (spec 5.5): annual 68%, quarterly 26%, monthly 6%.
# Seasonality: 62% of annual tests attach to the two commonest year ends, so
# the peak month carries ~310 000 tests against a mean of 195 000. That 13x is
# spec 8.3 tension 3 and it is a property of the EXPANSION, which means it is
# visible before the batch runs and can be smoothed by the scheduler rather
# than discovered by it.
# --------------------------------------------------------------------------
DueTests = Expand(
    "covenant_instances",
    into="covenant_tests",
    rule="bound.test_date_rule",            # from the definition, not from config
    anchor="client.financial_year_end",
    horizon_months=param(15, ge=1, le=24,
        description="How far ahead due tests are materialised"),
    carries=[
        "covenant_instance_id",
        "covenant_definition_version",       # the binding travels on every row
        "facility_id", "client_id",
        "step_schedule",                     # a covenant is not one number
        "delivery_deadline_days",
        "grace_runs_from",
    ],
    schema_transform="1 instance row -> 0..n test rows",
)


def threshold_on_step_schedule(step_schedule: list, test_date) -> float:
    """A covenant is not one number. Spec 5.3 O15.

        gearing <= 3.00 falling to 2.75 at month 18 and 2.50 at month 36

    The step schedule is a CONTRACTUAL parameter: fixed per instance at
    documentation, changeable by nobody, including the owner of the standard it
    was drawn from (spec 6.4). It therefore cannot live in a params document at
    all -- a policy owner opening `config/covenants/production.json` must find
    nothing there to change, because anything they could change would be a term
    of 960 000 contracts.
    """
    pass  # the last step whose effective month <= elapsed months at test_date


def three_dates(test_date, delivery_date, determination_date) -> tuple:
    """Test / delivery / determination. Spec 5.5.2. Collapsing any two is a
    defect class, so they are three columns and no step takes fewer than three.

      - the ratio is computed on data as at the TEST DATE;
      - the breach OCCURS on the test date, so a cure period measured from it
        can be part-spent before the client knows. Which convention applies is
        in the bound definition's `grace_runs_from`, and older facilities are on
        the older convention -- which is why it is bound and not policy;
      - the DETERMINATION DATE is what project 09 replays against. A test run
        three months late must reproduce as the test it was, not as a test of
        today.
    """
    pass  # validate ordering; emit all three; refuse if any is the run date
