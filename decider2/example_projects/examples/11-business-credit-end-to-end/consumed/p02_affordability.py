"""Project 02's regulated affordability assessment, and the missing fifth mode.

This is the cleanest example in the project of what a consumer does when a
component gives it 90% of what it needs, so it gets its own file despite being
called 0..1 times per assessment and about 620 times a day.

---------------------------------------------------------------------------
The gap
---------------------------------------------------------------------------
Project 02 publishes four assessment modes as a CLOSED SET (its 5.8): new
application, limit increase, arrangement, scenario. This project needs a fifth:
a periodic re-assessment of a sole proprietor whose evidence is a year old and
whose business income is the same money as their personal income, counted once.

The four options (spec 5.17.2) and what each would cost here:

  EXTEND        project 02 adds mode 5 on project 02's cadence, which is
                statutory-change-driven and therefore not ours to schedule.
  COMPOSE       we compute the fifth mode locally around the published four.
                Costs: the local part is unowned by Compliance, which is the
                team that must sign the statutory verdict. Disqualifying.
  PARAMETERISE  the modes become open. Project 02's own warning is that "modes
                must not become copies", and a fifth mode opened by a consumer
                is how the fifth becomes the ninth.
  FORK          never, by policy.

---------------------------------------------------------------------------
The choice, and the honest cost
---------------------------------------------------------------------------
EXTEND. We wait. Fork pressure #4 says plainly that the regulated assessment is
heavy, the fifth mode does not exist, and a local approximation would pass most
cases -- and that passing most cases is exactly the shape of a defect that
surfaces at an ombud hearing.

Until the mode lands, EP-3 on a sole-proprietor regulated facility routes to
referral with reason 5872 `periodic_affordability_mode_unavailable`. That is a
worse business outcome than an approximation and a better governance one, and
the referral volume (about 90/month) is the number that gets the mode onto
project 02's roadmap. An approximation would remove exactly that pressure.
"""

from decider2 import consume, gap, EXTEND, Branch
from consumed.manifest import MANIFEST, p02_major
from roles import SOLE_PROPRIETOR, SURETY

p02 = consume("credit.affordability", manifest=MANIFEST, major=p02_major)

# Spec 5.17.4 scenario 1 / 11 change scenario 16: project 02 ships a major on a
# statutory date nobody chose. EP-1 must move on the date; the in-flight EP-3
# cohort must not; every replay must resolve the version it ran on. The pin
# branch is compiled with both arms and selected per row from the manifest.
Affordability = Branch(
    p02_major,
    [p02.at_major(2), p02.at_major(3)],
    modifies=["affordability_verdict_code", "max_affordable_instalment_cents",
              "discretionary_income_cents"],
    fuse=False,        # see consumed/manifest.py: arms must not fuse
)

PERIODIC_SOLE_PROP = gap(
    "affordability_mode_5",
    component="credit.affordability@02",
    needs="periodic re-assessment of a sole proprietor on year-old evidence",
    resolution=EXTEND,
    owner="compliance",            # NOT this project
    raised="2026-02-03",
    review="2026-11-30",
    interim="refer, reason 5872",
    because=(
        "the statutory verdict is Compliance's to sign and a locally computed "
        "approximation of it is unsigned. Spec 5.17.3: one component, one "
        "answer, two consequences -- hard fail under the regulated regime, an "
        "input under the unregulated one. A local copy would have to reproduce "
        "that split, and would get it wrong in the direction that approves."
    ),
)

# The sole-proprietor double-count. Spec 5.17.3(1): drawings deducted from
# EBITDA at O14 and counted as income at project 02 is the same rand on both
# sides, and the resulting affordability is wrong in the client's favour by
# exactly the drawings.
#
# `SOLE_PROPRIETOR.once` (roles.py) declares the money is counted once and
# names both consumers. The framework then refuses to compile a graph where
# both `O14.debt_service` and `p02.income` read `business.drawings_cents` on a
# path that reaches one verdict. It is a static check: lineage already knows.
Affordability = Affordability.under(SOLE_PROPRIETOR).asserting("counted_once")

# Spec 5.17.3(2): the proposed instalment is a NEW OBLIGATION in project 02's
# terms and a DEBT SERVICE ITEM in O14's terms, and it must not appear in both.
# Same mechanism, different name, declared at the role rather than per call.
SuretyAffordability = Affordability.under(SURETY).asserting("instalment_once")
