"""Reason-code and absence-reason constants used by the skeleton.

The reason registry itself is `core.reason_codes` -- 380 codes, Compliance-
owned, monthly cadence, of which this project references 96.  This file holds
only the handful the *skeleton* needs to name in a `Halt`.  Every other reason
code in the project lives in a rule row, a gate row or a suppression rule,
where its owner can see it.

The `ABSENT_*` codes are a separate axis and this is the point of the file.
When `Halt` stops the flow, every downstream value becomes absent -- and
"absent because we declined at a gate" is a different fact from "absent because
the bureau returned nothing" and from "absent because the value is genuinely
null".  §5.1, §5.3 and core-library §7.4 each independently demand that
distinction and none of them can be served by a plain `Optional`.
See FRAMEWORK-DEMANDS #6.
"""

from decider2.missing import AbsenceReason

# --- outcomes that stop the flow -------------------------------------------
FRAUD_DECLINE = 3
AFFORDABILITY_FAIL = 3

# --- why a downstream value is absent --------------------------------------
ABSENT_GATE_DECLINE = AbsenceReason(10, "not evaluated: declined at an eligibility gate")
ABSENT_FRAUD_DECLINE = AbsenceReason(11, "not evaluated: fraud decline")
ABSENT_AFFORDABILITY_FAIL = AbsenceReason(12, "not evaluated: affordability fail")
ABSENT_NO_BUREAU_RECORD = AbsenceReason(20, "bureau returned no subject")
ABSENT_BUREAU_ENQUIRY_FAILED = AbsenceReason(21, "bureau enquiry failed")
ABSENT_BUREAU_NO_ACCOUNTS = AbsenceReason(22, "bureau subject with zero accounts")
ABSENT_NOT_YET_RETRIEVED = AbsenceReason(30, "gate input not retrieved at this point")
ABSENT_NEW_TO_BANK = AbsenceReason(31, "no internal behaviour exists")
