"""Three renderings from one record. See artefacts/explanations-FLX-2027-03-0418822.md
for what each one actually looks like for a real decision.

The renderings are not the same document at three lengths. They differ in what
they MAY contain, and the projection (explain/disclosure.py) is what enforces
that - the templates below cannot leak, because they never see a field above
their audience's level.

Three properties worth stating because each cost something:

1. THE CONSULTANT RENDERING IS NOT A SUMMARY. It is a different question.
   The analyst asks "what happened"; the consultant asks "what do I say, and
   what can the client do". So its content is: the dominant reason in the
   registry's wording at the decision date, in the client's language, plus the
   ONE thing that would most have changed the answer - which is computed, not
   written, from the chain: the binding link with the largest delta.

2. THE ADJUDICATOR RENDERING GLOSSES EVERY IDENTIFIER. Spec §5.2 forbids
   unexplained internal identifiers. So `CAP-0210` never appears alone; it
   appears as 'rule CAP-0210 ("where an account has been three or more months
   in arrears in the last twelve, the amount is limited to R40 000"), owned by
   Credit Risk Policy, approved CC-2026-08, in force from 2026-04-01'. The
   sentence is the `holds` sentence - the SAME string the reviewable artefact
   prints and the same string CI verifies. One sentence, three consumers.

3. THE OVERLAY IS ALWAYS VISIBLE, FOR ALL THREE AUDIENCES (§5.14.5). Whether a
   decline came from the applicant's circumstances or from the Bank's own
   deliberate conservatism is a different question with a different answer, and
   the adjudicator is entitled to know which they are looking at. The
   consultant gets it too, in one sentence, because a consultant who does not
   know the Bank tightened will tell the client something false.
"""

from __future__ import annotations

from explain.disclosure import Disclose
from explain.record import DecisionRecord


def consultant(r: DecisionRecord, *, language: str) -> str:
    """<= 5 sentences, one screen, < 3 s. No thresholds, no scores, no internal codes.

    Three languages from the reason registry at r.reason_registry_version.
    """
    pass  # project(r, CONSULTANT) -> dominant reason wording + largest-delta chain link, in words


def analyst(r: DecisionRecord) -> str:
    """Unbounded, 4-10 pages, < 15 s. Everything, plus the intervention affordances.

    Every gate evaluated in order; every rule evaluated with fired/bound/not-
    applicable distinguished; every tree node visited by stable node_key with
    the client's actual values; every chain in full; score contributions signed
    and ranked with bins; every cell read with coordinates and table version;
    adjusted and unadjusted side by side; and a one-click handle into
    whatif/intervene for each input and each param.
    """
    pass


def adjudicator(r: DecisionRecord, *, complaint_ref: str) -> str:
    """6-12 pages plus appendices. Not time-critical; correctness absolute.

    Reviewed by Compliance before issue - and the review is recorded, because
    spec §5.2 requires the Bank to be able to show what it actually told
    someone.
    """
    pass


def gloss(identifier: str, r: DecisionRecord) -> str:
    """Every internal identifier -> a sentence. Fails loudly if the identifier
    has no clause sentence, which is how a rule without a `holds` becomes
    visible to the person who has to defend it rather than to nobody."""
    pass


def issued(rendering: str, *, decision_id: str, audience: Disclose, by: str) -> str:
    """Spec §5.2: every explanation issued to a client is itself retained, in
    full, for 7 years - which decision, which rendering, which wording version,
    who issued it, when. When a client disputes what they were told, the Bank
    must be able to show what it actually said."""
    pass
