"""The interface four product arms share, and the not-applicable mechanism.

Spec question 4: "What is product routing, structurally? One scenario reaches
four heterogeneous product subflows with different inputs, different tables,
different policy sets, different owners and different outputs, and several can
answer the same question. Is that one construct or four, and where does the
shared part end?"

The answer this sketch gives:

    Routing is TWO constructs and the shared part ends at a frozen contract.

    - Which products MAY carry a set: a frame-tier Fanout against a routing
      table (products/routing.py). One-to-many. Set-shaped. A row per (set,
      product).
    - Which product logic runs for a given ROW: a record-tier Branch over
      product_code (search/evaluate.py). One-to-one. Record-shaped. Only the
      taken arm executes.
    - What the four arms must all produce: contracts/product_offer.json, frozen,
      checked at composition.

The shared part ends at the contract and NOT one line further. There is no base
product module, no template method, no "common pricing" the four arms
specialise. Four product teams on four release cadences editing one shared
superclass is the failure doc 01 5 records in a different costume. What they
share is a schema, and a schema does not have to be edited to be extended.
"""

from decider2 import module, step
from decider2.values import NotApplicable, na


# --- the commensurable surface ------------------------------------------------
#
# Five values, and everything downstream of pricing reads only these:
#
#   committed_monthly      what the client is contractually committed to pay
#   total_cost_of_credit   every rand paid over the horizon
#   horizon_months         over how long total_cost_of_credit is measured
#   advance_or_limit       the amount advanced, or for product 20 the limit
#   nominal_annual_rate    normalised to an absolute annual rate
#
# Getting these five right is what lets a 60-month unsecured loan, a revolving
# limit with a promotional rate, a balloon-structured vehicle refinance and an
# 84-month sub-term inside a 240-month bond be RANKED AGAINST EACH OTHER. Get
# one wrong and the objective silently prefers a product.
#
# The three that are usually got wrong, and this file exists to name them:
#
#   1. Product 20's committed_monthly is its STRESSED payment, not its
#      promotional minimum payment. A client who can afford the R310 promotional
#      minimum but not the R1 240 stressed payment has been sold a cliff, and an
#      objective minimising instalment will choose product 20 every time if it
#      is handed R310.
#
#   2. Product 30's total_cost_of_credit INCLUDES THE BALLOON. A balloon lowers
#      the instalment and is therefore attractive to the objective, and it
#      leaves a lump sum at the end. If the balloon is outside total cost, the
#      anti-harm rule can be defeated by structure rather than by argument -
#      which is a sentence that should be read twice, because it describes an
#      exploit available to anyone tuning a parameter.
#
#   3. Product 40's horizon_months is 84 - the mandatory sub-term - NOT the
#      bond's remaining 240. Against 240 months the anti-harm threshold is
#      unreachable by construction and product 40 wins everything.


@step(output="committed_monthly")
def committed_monthly_contract() -> float:
    """CONTRACT STUB. Each arm implements this. Documented here so the meaning is one place.

    What the client is contractually committed to pay per month for this
    facility. Not the minimum. Not the promotional figure. Not the figure before
    a required insurance premium. The figure whose absence from the client's
    bank account on the 25th is a missed payment.

    The composition-time assertion in the contract file reads:

        "committed_monthly is the figure the affordability test uses - an arm
        that tests affordability against a different figure from the one it
        publishes here is a defect, and this is the assertion that catches it."

    That assertion is checkable because AffordScenario (search/evaluate.py) binds
    `instalment` to this exact column by name. An arm that wanted to be tested
    on a different figure would have to rebind the library module, which is a
    visible `.at()` in a product file and a code review conversation.
    """
    pass  # implemented per arm


# --- na(): the third state ----------------------------------------------------
#
# Doc 03 8.2: "Every arm must produce every declared `modifies` value, with
# agreeing types - validated at build time."
#
# That is right for two arms of a sector cap. It is wrong for four products where
# one has no term, no instalment and no balloon. The available workarounds are
# all bad:
#
#   term_months = 0        a sentinel that arithmetic will happily use. CON-INT-
#                          05 computes "term extension over the longest settled
#                          remaining term" and gets a large negative number,
#                          which passes.
#   term_months = None     doc 03 1 tier 3. Correct, and it makes every
#                          downstream step Optional and forces `if x is None`
#                          into fourteen interventions. It also CONFLATES
#                          not-applicable with not-collected, which library 7.4
#                          says must never be conflated.
#   split the Branch       four products, two Branches, two contracts. The
#                          objective then cannot rank across them.
#
# So: na() is a DECLARED NOT-APPLICABLE, a third state distinct from a value and
# from a null, and the framework enforces that a step reading a may_be_na column
# declares `Maybe[T]` and handles it. See FRAMEWORK-DEMANDS D4.
#
# The same mechanism serves spec 5.7.6 - "not-applicable is distinct from
# passed" - for intervention outcomes. CON-INT-10 (minimum external proportion)
# does not apply to product 20, and a record showing it as PASSED is misleading.
# One concept, two uses, which is the test for whether a concept earns its place.


ProductOffer = module(
    committed_monthly_contract,
    name="product_offer_contract",
    contract="contracts/product_offer.json",
    abstract=True,  # declares the interface; has no implementation of its own
)


# --- what `abstract=True` is NOT ----------------------------------------------
#
# It is not a base class and the four arms do not inherit from it. It is a module
# that declares reads/writes and has no steps, existing so the contract has a
# Python object to hang off and so `pipeline.schema()` can name it. The arms
# reference the same contract FILE, not this object. Nothing resolves through it
# at runtime and removing it would change no behaviour - which is the property
# that distinguishes a schema from a superclass.
