"""One-off script: builds sample_request.json. Not part of the delivered package
surface -- run once, output committed."""
import json

def account(ref, type_code, provider, internal, balance, limit, instalment, rate, term, arrears,
            opened, secured=False, security_type=0, status=0, disputed=False,
            quotation_amount=0.0, quotation_reference="", quotation_expiry_date="2000-01-01"):
    # `existing_quotation` (spec 06 §4.3) is flattened into three scalar fields, not a
    # nested struct -- a struct-within-a-struct with mixed types crashes single-record
    # scoring at `decider`'s numpy conversion boundary (see NOTES.md "Framework
    # friction": `DTypePromotionError`, DateTime64 vs. Float64, inside `Series.to_numpy()`).
    # A "no quotation held" account carries the same sentinel-not-null convention as
    # `last_consolidation_date` below: `quotation_amount=0.0` (settlement.py treats 0 as
    # "no quotation on file" via `existing_quotation`-style checks adapted to this shape).
    return {
        "account_ref": ref, "account_type_code": type_code, "provider_code": provider,
        "is_internal": internal, "balance": balance, "credit_limit": limit, "instalment": instalment,
        "nominal_annual_rate": rate, "remaining_term_months": term, "months_in_arrears": arrears,
        "opened_date": opened, "is_secured": secured, "security_type_code": security_type,
        "account_status_code": status, "is_disputed": disputed,
        "quotation_amount": quotation_amount, "quotation_reference": quotation_reference,
        "quotation_expiry_date": quotation_expiry_date,
    }

accounts = [
    account(1, 1, 0, True, 45000.0, None, 1850.0, 0.235, 28, 0, "2022-03-10"),          # internal Flex Loan
    account(2, 10, 0, True, 12000.0, 15000.0, 450.0, 0.27, None, 0, "2021-06-01"),       # internal credit card
    account(3, 20, 105, False, 38000.0, None, 1420.0, 0.245, 22, 1, "2020-11-15",
            quotation_amount=38900.0, quotation_reference="Q-105-8891", quotation_expiry_date="2026-11-01"),
    account(4, 11, 106, False, 9500.0, 10000.0, 610.0, 0.289, None, 0, "2019-02-20"),    # store card
    account(5, 10, 101, False, 21000.0, 25000.0, 780.0, 0.262, None, 0, "2023-01-05"),   # credit card
    account(6, 20, 102, False, 55000.0, None, 1980.0, 0.219, 30, 0, "2018-07-12"),       # personal loan (other bank)
    account(7, 10, 201, False, 6800.0, 8000.0, 320.0, 0.299, None, 0, "2022-09-30"),     # credit card, slow provider
    account(8, 20, 901, False, 17000.0, None, 690.0, 0.235, 18, 0, "2021-04-18"),        # non-quoting provider
    account(9, 2, 103, False, 185000.0, None, 4650.0, 0.115, 42, 0, "2019-05-01",
            secured=True, security_type=1),
]

req = {
    "decision_id": "06-sample-0001",
    "application_id": 555001,
    "decision_date": "2026-09-24",
    "channel_code": 1,
    "requested_amount": 60000.0,
    "assessment_mode_code": 1,
    "client_nominated_settle": [0],
    "client_excluded_settle": [0],
    "hardship_declared": False,
    "accounts": accounts,
    "risk_grade": 6,
    "applicant_age_years": 41.0,
    "dependants_count": 2,
    "employment_type_code": 1,
    "payslip_income": 46000.0,
    "variable_pay_history": [0.0],
    "declared_expenses": {"groceries": 3200.0, "transport": 1400.0, "utilities": 1100.0},
    "statement_expenses": {"groceries": 3050.0, "transport": 1350.0, "utilities": 1050.0},
    "bureau_as_of_date": "2026-09-20",
    "court_ordered_deductions": 0.0,
    "segment_code": 1,
    "objective_id": "OBJ-02",
    "consolidations_last_24_months": 0,
    "client_under_debt_review": False,
    "client_under_administration": False,
    # A sentinel far in the past, not `null` -- a lone `None` scalar in a single-record
    # request is exactly as ambiguous as an empty ragged list at the arrow-import
    # boundary (02's NOTES.md "Framework friction" #4.1: `Null`-dtype column vs. a
    # declared nullable type). "No prior consolidation on file" is expressed as "long
    # enough ago that CON-ELIG-03's 6-month window never binds", not as a null.
    "last_consolidation_date": "2000-01-01",
    "income_verified": True,
    "bureau_unobtainable": False,
    "active_reckless_lending_allegation": False,
}

with open("sample_request.json", "w") as f:
    json.dump(req, f, indent=2)
print("wrote sample_request.json with", len(accounts), "accounts")
