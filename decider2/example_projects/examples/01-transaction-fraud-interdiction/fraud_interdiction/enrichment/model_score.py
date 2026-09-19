"""5.5 — the external fraud model score.

The design constraint is one sentence in the spec and it is the sharpest
determinism requirement in the document:

    "Two events with identical inputs, one scored and one not, may receive
     different actions; what is not allowed is the same event replayed producing
     a different action because the replay happened to get a score."

So the score is an `Observed`, its absence carries a *reason*, and the replay
path reads the recorded value — including the recorded absence. The network call
does not exist in the decision tier at all; it happens in enrichment, and the
decision tier cannot tell a live score from a replayed one because it only ever
sees a feature-vector column.

Spec §11.7 (the model becomes two models) is why this module is parameterised
over a model slot rather than hard-coding one score. Two slots means four
absence combinations, and the degraded-mode table in `config/degraded_modes.json`
enumerates all four rather than deriving them.
"""

from __future__ import annotations

from decider2 import Observed, module, observed, param
from decider2.types import Instant

ABSENCE_TIMEOUT = 1
ABSENCE_ERROR = 2
ABSENCE_CIRCUIT_OPEN = 3
ABSENCE_NOT_REQUESTED = 4


def card_model_score(
    value: Observed[float] = observed(
        source="model:card",
        deadline_ms=param(8.0, ge=1, le=50, unit="ms"),
        absence_reasons=(ABSENCE_TIMEOUT, ABSENCE_ERROR, ABSENCE_CIRCUIT_OPEN, ABSENCE_NOT_REQUESTED),
    ),
) -> Observed[float]:
    """0–1000, or an explicit absence with a reason code."""
    pass


def scam_model_score(
    value: Observed[float] = observed(source="model:scam", deadline_ms=8.0),
) -> Observed[float]:
    """The second model, spec §11.7. Absent independently of the card model."""
    pass


def model_presence_code(
    card_model_score: Observed[float],
    scam_model_score: Observed[float],
) -> int:
    """0 both, 1 card only, 2 scam only, 3 neither.

    Four states, named, so the degraded-mode table can key on them and the
    post-mortem question "which combination were we in" has an answer that is a
    column rather than an inference.
    """
    pass


def model_factor_codes(card_model_score: Observed[float]) -> tuple[int, int, int]:
    """Up to three contributing-factor codes, recorded but not read by rules."""
    pass


ModelScore = module(
    card_model_score,
    scam_model_score,
    model_presence_code,
    model_factor_codes,
    name="model_score",
    contract="contracts/feature_vector.json#/model",
    taps=["model_presence_code"],
)
