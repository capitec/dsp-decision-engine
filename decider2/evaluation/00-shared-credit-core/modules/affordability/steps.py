"""Affordability capability — discretionary income and capacity.

Implements: Credit Policy §6.5
"""


def living_expenses(
    declared_living_expenses: float,
    gross_monthly_income: float,
    dependants_count: int,
) -> float:
    """Applied living expenses after applying statutory minimum floor.

    The floor is a function of income band and dependant count.
    Simplified: floor = R3000 + (R500 * dependants_count).

    Implements: Credit Policy §6.3
    """
    min_floor = 3000.0 + (500.0 * dependants_count)
    return max(declared_living_expenses, min_floor)


def existing_obligations(
    num_accounts: int,
    average_account_instalment: float,
) -> float:
    """Monthly cost of debt already held.

    Simplified: sum of imputed instalments on existing accounts.
    Implements: Credit Policy §6.4
    """
    return num_accounts * average_account_instalment


def discretionary_income(
    net_monthly_income: float,
    living_expenses: float,
    existing_obligations: float,
) -> float:
    """Income remaining after living expenses and debt obligations.

    Implements: Credit Policy §6.5.1
    """
    return net_monthly_income - living_expenses - existing_obligations


def affordability_verdict_code(
    discretionary_income: float,
    instalment: float,
    params,
) -> int:
    """Pass/fail on affordability.

    1 = pass, 2 = marginal, 3 = fail
    """
    if discretionary_income < 0:
        return 3  # fail
    if discretionary_income < instalment * params.marginal_threshold:
        return 2  # marginal
    return 1  # pass


def max_affordable_instalment(
    discretionary_income: float,
    params,
) -> float:
    """Maximum monthly commitment after affordability buffer.

    Implements: Credit Policy §6.5.2
    """
    return discretionary_income * params.affordability_buffer
