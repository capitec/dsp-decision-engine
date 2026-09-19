"""5.3 — merchant and acquirer enrichment (card authorisations).

Three tables of wildly different size behind one interface: MCC risk (1 024
rows, analyst-edited weekly), BIN/issuer (~50 000 rows, monthly scheme file),
merchant reputation (3.2 M identifiers, daily). Same declaration, different
backing store, chosen by the framework from the declared row count and cadence
— not by the author. See FRAMEWORK-DEMANDS.md #18.
"""

from __future__ import annotations

from decider2 import Observed, module, observed, param

from fraud_interdiction.tables.definitions import BIN_ISSUER, MCC_RISK, MERCHANT_REPUTATION

UNKNOWN_BAND = 0


def mcc_risk_band(mcc: Observed[int], event_timestamp) -> Observed[int]:
    """Merchant category risk band from the version in force."""
    pass  # MCC_RISK.as_at(event_timestamp)[mcc].risk_band


def merchant_reputation_band(merchant_id: Observed[str], event_timestamp) -> Observed[int]:
    """Reputation band, or UNKNOWN_BAND for the 2.1% of card-not-present events
    whose merchant is unrecognised.

    UNKNOWN_BAND is a *band*, not an absence: an unrecognised merchant is a real
    and meaningful state that several CF rules key on directly, and modelling it
    as ABSENT would push those rules into their on_absent behaviour and silently
    switch them off on exactly the population they exist for.
    """
    pass


def merchant_chargeback_band(merchant_id: Observed[str], event_timestamp) -> Observed[int]:
    pass


def issuer_attributes(bin_prefix: Observed[int], event_timestamp) -> Observed[int]:
    """BIN-derived issuer attributes where the card is not the Bank's."""
    pass  # BIN_ISSUER.as_at(event_timestamp)


def merchant_on_client_prior_use(client_id: str, merchant_id: Observed[str]) -> Observed[bool]:
    pass


def acquirer_country_risk_band(acquirer_country: Observed[int], event_timestamp) -> Observed[int]:
    """Country risk from the Compliance-owned table.

    Spec §11.16: a country is sanctioned overnight, the table changes outside its
    monthly cadence with same-day effect, and four rules' behaviour changes
    without any rule being edited. That works because the *band* is the rule's
    input and the country-to-band mapping is a versioned table — the rules read
    `>= 4`, and which countries are band 4 is Compliance's to change.
    """
    pass


Merchant = module(
    mcc_risk_band,
    merchant_reputation_band,
    merchant_chargeback_band,
    issuer_attributes,
    merchant_on_client_prior_use,
    acquirer_country_risk_band,
    name="merchant",
    contract="contracts/feature_vector.json#/merchant",
)
