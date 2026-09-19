"""Business-day arithmetic. The single most-used primitive in this project.

Every notice period, every grace end, every "within N business days" predicate
in suspensions/ and path/ resolves through here.

The representation is the point: the ZA public-holiday calendar compiles to a
DENSE int32 array indexed by date ordinal, holding the running count of
business days. Then

    bdays_between(a, b)  ==  ORD[b] - ORD[a]          # one subtract
    add_bdays(a, n)      ==  INV[ORD[a] + n]          # one add, one index

Both are O(1) inside a numba kernel with no loop and no branch. A naive
"iterate forward skipping weekends and holidays" implementation is a loop whose
trip count depends on data, which is the shape that makes a 2.3M-row kernel
unpredictable and a <400ms budget unmeetable.

DEVIATION FROM DOC 03: doc 03 §4 "Tables (keyed lookups) — provisional" sketches
`tables.term.max_loan[term]` and says nothing about how a table is BUILT from a
source artefact. A calendar is not a lookup of a stored value; it is a stored
value plus a derived index. See FRAMEWORK-DEMANDS #7.
"""

from decider2 import Table, table_index, step
from decider2.types import Date, i4


class BusinessDayCalendar(Table):
    """Gazetted ZA public holidays. Compliance-owned, annual, effective-dated."""

    holiday_date: Date
    gazette_reference: str
    applies_from: Date
    applies_to: Date | None


@table_index(BusinessDayCalendar, name="bday")
def build_bday_index(cal: BusinessDayCalendar, first: Date, last: Date) -> tuple:
    pass  # returns (ORD: i4[N], INV: i4[M]) — prefix count of business days, and its inverse


def bdays_between(start: Date, end: Date, bday) -> i4:
    pass  # ORD[end] - ORD[start]; negative if end precedes start


def add_bdays(start: Date, n: i4, bday) -> Date:
    pass  # INV[ORD[start] + n]; the next business day at or after, then n more


def days_between(start: Date, end: Date) -> i4:
    pass  # plain ordinal subtraction; Date is int32 days since epoch


def age_days(when: Date, decision_date: Date) -> i4:
    pass  # days_between(when, decision_date); the single most-written predicate


def is_business_day(d: Date, bday) -> bool:
    pass  # ORD[d] != ORD[d - 1]
