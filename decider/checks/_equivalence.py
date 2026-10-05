from __future__ import annotations

from typing import Any

from decider.checks._model import Finding, Severity
from decider.checks._run import Check
from decider.testing import MODES, assert_equivalent, corpus


def equivalence(modes: tuple[str, ...] = MODES) -> Check:
    """Whole-path check: every execution mode must agree on the generated corpus.

    Reuses `testing.corpus` (the generated boundary/regime cases) and
    `testing.assert_equivalent` (cross-mode comparison), so this is the same
    evidence a test suite collects, surfaced as findings. A mode divergence —
    or a crash on any generated case — is an `error`-severity finding.

    Example::

        checks.run(pipeline, suites=[checks.default_suite, (equivalence(),)])
    """

    def check(step: Any) -> list[Finding]:
        try:
            frames = corpus(step)
        except ValueError:
            return []  # the step reads no input; there are no cases to generate
        findings: list[Finding] = []
        for name, frame in frames.items():
            try:
                assert_equivalent(step, frame, modes=modes)
            except Exception as e:
                findings.append(Finding(check="equivalence", severity=Severity.ERROR,
                                        message=f"corpus {name!r}: {type(e).__name__}: {e}",
                                        detail={"frame": name}))
        return findings

    check.__name__ = "equivalence"
    return check
