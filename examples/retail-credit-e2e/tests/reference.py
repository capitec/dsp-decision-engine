"""The price of an offer worked out by hand from the params document, to check the pipeline against."""


def find(node, key):
    """The first dict stored under `key` anywhere in a params document."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            if (found := find(value, key)) is not None:
                return found
    return None


def card_rate(params, amount, term, grade) -> tuple[float, int, int]:
    """(rate as a fraction, amount band, term band) of the card cell that prices this request."""
    card = find(params, "card_rate")
    floors = [row["floor"] for row in card["amount_bands"]]
    band = max(i for i, floor in enumerate(floors) if floor <= amount)
    term_band = min(max(int(term), 6), 45) - 6
    return card["cells"][(band * 40 + term_band) * 12 + grade - 1]["rate"] / 100.0, band, term_band


def instalment(params, amount, term, grade, credit_life_rate, loading=0.0, addon=0.0) -> float:
    """Initiation fee capitalised, credit life on the amount financed, then the annuity plus the service fee."""
    fee = round(min(210.0 + 0.095 * max(0.0, amount - 1200.0), 1480.0) * 1.15, 2)
    financed = amount + fee
    premium = round(min(financed / 1000.0 * credit_life_rate, 1000.0), 2)
    r = (card_rate(params, amount, term, grade)[0] + loading + addon) / 12.0
    return round(financed * r / (1.0 - (1.0 + r) ** -term) + 87.98 + premium, 2)
