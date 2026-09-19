"""The asymmetry register, and verifying it INDEPENDENTLY of the flow that
claims to enforce it.

Spec §5.14.6: most overlays in this estate may only tighten. Flow 02's may not
increase affordability capacity; flow 03's may not raise a ceiling or price
below the card; flow 08's may not weaken a regulatory suspension, a notice
period or a contact frequency cap.

"An overlay that can be made to loosen by supplying a negative magnitude has no
asymmetry at all, and finding that out belongs here rather than in an
incident."

TWO CHECKS, AND THE SECOND IS THE POINT
  1. DEFINITION TIME. The register holds which asymmetry applies to which
     overlay kind in which flow, and refuses an entry whose magnitude sign
     contradicts it. Necessary, trivially bypassed by a flow that does its own
     arithmetic.

  2. INDEPENDENT VERIFICATION, WHICH IS WHY THIS FILE EXISTS. The harness does
     not trust the flow's enforcement. It proves the property from the flow's
     MANIFEST plus a bounded search:

     (a) STRUCTURAL. For each overlay consumption point, the manifest's graph
         shows the composition: `adjusted = min(base, base * (1 - m))` composes
         monotonically downward for m in [0, 1]; `adjusted = base * (1 - m)`
         does not, because m < 0 loosens. This is a property of the STEP, and
         because steps are small pure scalar functions over declared types
         (doc 03 §1), it is readable. Where the composition is a registered
         capability (`core.adjustments`), the check is done once for the
         capability and inherited by every consumer - which is 21 checks
         instead of 8 x N.

     (b) EMPIRICAL, AS A BACKSTOP. Sweep the magnitude across its declared
         bounds AND across its bounds' negation, over 10 000 golden records,
         and assert monotonicity of the affected output. This is a `holds`
         assertion the harness writes rather than the flow team:

             @holds("no magnitude of ADJ kind cap_reduction can raise amount_cap",
                    given=sweep("magnitude", -1.0, 1.0, 41),
                    then=lambda r: r.amount_cap <= r.amount_cap_unadjusted)

         It runs in the flow's own certification suite, as a record the flow
         team can see fail, but it is AUTHORED HERE from the register - so a
         flow cannot delete it, and a new overlay kind cannot ship without one.
"""

from __future__ import annotations

from typing import Literal

from manifest.model import GovernanceManifest


class AsymmetryRule:
    flow: str
    kind: str
    direction: Literal["tighten_only", "loosen_only", "either"]
    affected_output: str
    statutory: bool          # flow 08's notice periods and contact caps: yes. Cannot be waived.
    approval_to_waive: str | None


def verify_structural(m: GovernanceManifest, rules: tuple[AsymmetryRule, ...]) -> tuple["Finding", ...]:
    """Read the composition expression at each overlay consumption point."""
    pass


def generated_claims(rules: tuple[AsymmetryRule, ...]) -> tuple["Claim", ...]:
    """The `holds` assertions the harness injects into each flow's golden suite."""
    pass


def waivers() -> tuple["Waiver", ...]:
    """The one legitimate loosening in the estate: flow 03's CAP-0420 campaign
    uplift, which may RAISE amount_cap by up to 25% subject to an absolute
    R250 000 authority ceiling and subject to never exceeding a cap set by a
    rule of class `regulatory`.

    It is not an overlay - it is a rule with an authority reference, which is
    the right modelling and is the reason it is auditable. Recorded here anyway,
    because "the only thing in this estate that may loosen" is a sentence
    somebody should be able to grep for.
    """
    pass
