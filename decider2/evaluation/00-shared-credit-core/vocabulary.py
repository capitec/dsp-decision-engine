"""Project vocabulary — mapping from local names to library names.

This demonstrates how a consuming project with different naming conventions
can adapt the shared library without creating passthrough steps.

Example: A product team's internal names don't match the library's canonical
vocabulary. Instead of renaming at every call site, they declare the mapping once.

Implements: Credit Policy §7.4.2 on reusability
"""

from decider2 import Vocabulary

# Example 1: systematic prefix mapping for Access Facility product
access_facility_vocabulary = Vocabulary(
    {
        # explicit pairs for values with no systematic relationship
    },
    prefixes={
        # bureau_ → cb_ (bureau data comes with a different prefix)
        "bureau_": "cb_",
    },
)

# Example 2: explicit pair-by-pair mapping for Flex Loan
flex_loan_vocabulary = Vocabulary(
    {
        # The library uses 'applicant_age_years' but this project uses 'age'
        "applicant_age_years": "age",
        # The library uses 'dependants_count' but this project uses 'num_dependents'
        "dependants_count": "num_dependents",
        # The library uses 'declared_living_expenses' but this project tracks 'monthly_expenses'
        "declared_living_expenses": "monthly_expenses",
        # Output mapping: the library produces 'net_monthly_income'
        # but this project stores it as 'take_home_pay'
        # This is a write relabel
    },
)

# Example 3: A consumer that uses the library unchanged (no vocabulary)
no_mapping_vocabulary = Vocabulary({})

__all__ = [
    "access_facility_vocabulary",
    "flex_loan_vocabulary",
    "no_mapping_vocabulary",
]
