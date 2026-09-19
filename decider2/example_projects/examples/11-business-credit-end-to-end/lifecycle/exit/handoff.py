"""L6 -- exit and handoff. Spec 5.9. Thirty-two decision points and one contract.

The interesting part is not the exit decision; it is that the handoff contract
is a CONSUMER-DRIVEN INTERFACE on a flow the consumer never runs.

Spec 5.15.1 collision 5: "Recoveries needs a field in the handoff that
Origination Engineering owns -- a change to a phase for the benefit of a team
that never runs it." And spec 5.15.2 requirement 6: Provisioning, Recoveries and
Audit change nothing and can BLOCK ANYTHING, "and their requirements must be
expressible as CONSTRAINTS ON THE ARTEFACT, not as review comments."

That last sentence is a framework demand and this file is where it lands.
FRAMEWORK-DEMANDS D13.
"""

from decider2 import module, consumer_contract, required_output

HANDOFF = consumer_contract(
    "recoveries_handoff",
    consumer="business_recoveries",
    # A consumer contract is the mirror of doc 03 5.1's `contract=`: that
    # freezes what a PUBLISHER promises, this freezes what a CONSUMER requires.
    # A build that would stop producing any of these fails, naming the consumer
    # and the field, in the team that broke it -- not in Recoveries' sprint
    # three months later when they open a file and it is not there.
    requires=[
        "decision_history[*].{kind,dates,outcome,authority,approver,evidence_ref}",
        "security_position.allocation@exit_date",
        "security_position.valuations[*].{date,basis}",
        "security_position.prior_encumbrances",
        "security_position.registrations[*].status",
        "covenant_history[*].{test,breach,cure,waiver.conditions,waiver.expiry}",
        "cross_default_state",
        "entity_structure@exit_date",
        "entity_structure@origination",
        "entity_structure_delta",
        "group_position.{members,exposures,cross_guarantees,composite_security}",
        "forbearance_history.probation_clocks",
        "staging_history.triggers",
    ],
    # The delta in item 4 is the one Recoveries actually uses and the one an
    # origination team would never think to produce: "the sureties Recoveries
    # will pursue were given by people whose roles may have changed."
    rationale="spec 5.9's six-item handoff contract",
)


@required_output
def handoff_package(facility_id: int, exit_kind: int) -> dict:
    """Recovery / provisioning / voluntary closure. Three exits, one package.

    Two requirements that OUTLAST the handoff:

      - The history does not stop. A workout decision in 2032 must be able to
        reach the 2026 origination record and read it, INCLUDING THE COMMITTEE
        PACK THAT APPROVED IT. Handoff is a change of owner, not an archive. So
        the package contains addresses into the decision store, not copies, and
        the store's retention is seven years after closure.
      - Return is possible and IS NOT AMNESIA. A facility that comes back from
        W4 to W1 carries its history. "A 'clean' facility with a two-year-old
        forbearance and its probation still running is not a clean facility, and
        the next review, the next amendment and the next appetite check must all
        know that."

    The second is enforced at O1: `existing_state` loads all four flags and
    their clocks for every subject, on every entry point, with no exception for
    a facility that has returned. There is no `is_clean` shortcut anywhere in
    the project, because that shortcut is the amnesia.
    """
    pass  # assemble addresses per HANDOFF; assert the contract is satisfied


def exit_recommendation(triggers_met: list) -> int:
    """18 declared exit triggers (spec 5.4.1 output 4). A RECOMMENDATION is not
    a decision -- L6 is -- but it must be produced, and "a review that produces
    no recommendation on a facility meeting four triggers is a finding."

    So the finding is computed, not hoped for: the review's own coherence check
    (spec 9.2 property 3) counts triggers met against recommendations produced,
    monthly, over the whole book, and reports exceptions. "A book with zero
    exceptions is a book where the check is not working."
    """
    pass  # count triggers; recommend; emit a finding where the count disagrees


Exit = module(handoff_package, exit_recommendation, name="exit",
              owner="business_recoveries", co_owners=["portfolio_management"],
              taps=["exit_kind", "triggers_met", "branch_path"])
