"""L4 -- the amendment re-open matrix. Spec 5.7, 13-Q22.

    "Which phases an amendment re-opens must be DECLARED DATA, reviewed and
     approved as policy, and not left to the judgement of whoever is handling it."

14 amendment kinds x 28 phase-parts = 392 cells, owned by Credit Governance,
approved by Credit Committee, versioned and effective-dated like any other table.

---------------------------------------------------------------------------
13-Q22: how does the matrix stay honest?
---------------------------------------------------------------------------
    "It is a governance artefact that is also a control-flow artefact, and the
     failure mode is that the code and the matrix drift apart silently."

The answer this project gives is the only one that actually works: **there is no
code for the matrix to drift from.** The matrix IS the control flow. `scoped_by`
takes the governance artefact and produces the pipeline; there is no second
expression of which phases run.

Two checks make that safe rather than merely true:

  M1  `assert_matrix_total(matrix, pipeline)` -- every phase-part named in the
      matrix exists in the pipeline, every phase-part in the pipeline is named
      in the matrix, no blanks. 392 cells, zero orphans, machine-checked on
      every build. A phase renamed in Python without a matrix edit fails the
      build naming both.
  M2  The matrix's effective dating is the POLICY rule, so a 2027 amendment
      replays against the 2027 matrix. Spec 5.7 requirement 4: the re-opened
      phases run on TODAY'S artefacts against a record produced on older ones,
      so the amendment's decision of record carries a `comparison_basis_code`
      against the origination like any other re-decision.
"""

from decider2 import governance_matrix, assert_matrix_total, module, RunLevel

amendment_reopen = governance_matrix(
    "tables/governance/amendment_reopen_matrix.csv",
    rows="amendment_kind",            # AM-01..AM-14
    cols="phase_part",                # 28 parts across O1-O17
    values=RunLevel,                  # FULL | PARTIAL(part) | NOT_RUN
    owner="credit_governance",
    approved_by="credit_committee",
    effective_dated=True,
    total=True,
)


def amendment_kind(request: dict) -> int:
    """AM-01..AM-14. Spec 5.7's table.

    The two that matter most:
      AM-02  limit increase beyond AM-01's bounds -> re-opens O1-O17 IN FULL.
             "It is a new credit decision wearing an amendment's clothes."
      AM-11  ownership change consent -> O2, O3, O5, O6, O9, O10 AND A CASCADE.
             The single most expensive amendment in the matrix.
    """
    pass  # classify from the request; refuse ambiguity rather than defaulting


def not_an_amendment(reopened: list) -> bool:
    """Spec 5.7 requirement 1: an amendment that re-opens NOTHING is not an
    amendment; it is a record change, handled elsewhere, with no decision of
    record.

    Checkable from the matrix row alone, statically, before anything runs -- so
    a proposed 15th amendment kind whose row is all NOT_RUN fails review with a
    reason rather than shipping as a decision type that writes empty records.
    """
    pass  # all(cell is NOT_RUN for cell in row)


def inherits_definitions(facility_id: int, amendment_kind: int) -> list:
    """Requirement 2: an amendment INHERITS the facility's covenant definitions
    unless it explicitly replaces them (AM-06, AM-07, AM-12).

    A replacement creates a NEW `covenant_definition_version` dated at the
    amendment, leaving every prior test bound to the prior version. Expressed as
    a call to `modules.covenants.definition.bind` on NEW instances only -- there
    is no rebind anywhere in the project, which is what makes requirement 2 true
    by absence rather than by discipline.
    """
    pass  # carry forward bindings; bind new instances for replaced covenants


def security_amendments_reopen_the_pool(amendment_kind: int, items: list) -> list:
    """AM-04 (substitution) and AM-05 (release) re-open O12 FOR EVERY FACILITY
    SHARING EITHER ITEM, not just the subject.

    This is not read from the matrix -- the matrix cannot know which facilities
    share an item, because that is data. It comes from
    `modules.security.allocation.Allocation.reopens`, which the framework
    evaluates to produce the affected set. So the matrix says WHICH PHASES and
    the cross-record declaration says WHICH SUBJECTS, and the two compose.

    AM-05's note is the sharp one: "the authority is that of the RESULTING
    position, not of the release."
    """
    pass  # Allocation.reopens(items) -> facility set -> re-run O12 for each


MATRIX_CHECK = assert_matrix_total(amendment_reopen, "pipelines.origination")

Amendment = module(amendment_kind, not_an_amendment, inherits_definitions,
                   security_amendments_reopen_the_pool,
                   name="amendment", owner="credit_governance",
                   taps=["amendment_kind", "phases_reopened", "matrix_version"])
