"""One decision explained three ways, from its evidence alone."""
from __future__ import annotations

import pkgutil
from dataclasses import dataclass

from governance.evidence import Evidence
from governance.flows import FLOWS


@dataclass(frozen=True)
class Explanation:
    audience: str
    text: str
    detail: dict


def _describe(flow, code) -> str:
    # workaround: ReasonCodeRegistry has no public lookup from a code to its wording.
    entry = pkgutil.resolve_name(flow.registry)._by_code.get(code)
    return entry.description if entry else str(code)


def explain(evidence: Evidence, audience: str) -> Explanation:
    """What a decision was and why, for a `consultant`, an `analyst` or an `adjudicator`.

    The consultant gets the deciding reason and nothing internal; the analyst gets every reason, the fields that
    show how the flow decided, and each adjusted value beside its unadjusted one; the adjudicator gets the
    reasons in plain sentences with the policy version.

        explain(evidence, "consultant").text   # "Maximum affordable instalment exceeded"
    """
    flow, r = FLOWS[evidence.flow], evidence.response
    codes = r.get(flow.reasons) or []
    primary = _describe(flow, r.get(flow.primary)) if r.get(flow.primary) else None
    if audience == "consultant":
        return Explanation(audience, primary or f"No reason applies (outcome {r[flow.outcome]}).", {})
    reasons = [{"code": c, "description": _describe(flow, c)} for c in codes]
    if audience == "analyst":
        pairs = {k[:-11]: {"adjusted": r[k[:-11]], "unadjusted": r[k]} for k in r
                 if k.endswith("_unadjusted") and k[:-11] in r}
        detail = {"reasons": reasons, "how": {k: r.get(k) for k in flow.detail}, "adjusted_vs_unadjusted": pairs,
                  "config_version": evidence.config_version, "documents": evidence.documents}
        return Explanation(audience, f"Outcome {r[flow.outcome]}. Deciding reason: {primary}.", detail)
    if audience == "adjudicator":
        text = (f"Decision {evidence.decision_id} was made on {evidence.decision_date} under version "
                f"{evidence.config_version} of the {flow.code} policy. Outcome {r[flow.outcome]}. "
                f"Reasons, most severe first: {'; '.join(x['description'] for x in reasons) or 'none'}.")
        return Explanation(audience, text, {"reasons": reasons, "registry": r.get("reason_registry_version")})
    raise ValueError(f"unknown audience {audience!r}")
