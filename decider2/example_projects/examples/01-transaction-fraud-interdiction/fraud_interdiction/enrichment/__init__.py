"""The enrichment pipeline — spec §5.2 to §5.6.

Its single job is to produce **the feature vector**: the ~380-column artefact
that is (a) what the rule kernels read, (b) what the decision record stores, and
(c) what the backtest replays. One artefact, three consumers, frozen by
`contracts/feature_vector.json`.

That identity is the whole equivalence story. The backtest does not "use
recorded values where possible" — it cannot do anything else, because the
decision pipeline's only leaf inputs *are* the feature vector's columns, and the
decision record *is* a feature vector. Spec §5.17's equivalence requirement
stops being a promise and becomes a type.

    Interdiction = Enrichment | Decision        # real-time
    Backtest     = ReadRecords | Decision       # 90 days, identical Decision

Deviation from doc 03: `fan(...)` is not in doc 03. Enrichment sources are
mutually independent and must be *issued concurrently* — the model score has an
8 ms deadline that must overlap the lookups (spec §5.5). `|` means sequence
(doc 03 §8.1), so writing them with `|` would forbid the concurrency the latency
budget requires. `fan()` declares independence; the runtime may issue members
concurrently and must join before the next `|`. See FRAMEWORK-DEMANDS.md #12.
"""

from decider2 import fan

from fraud_interdiction.enrichment.client_context import ClientContext
from fraud_interdiction.enrichment.completeness import Completeness
from fraud_interdiction.enrichment.counterparty import Counterparty
from fraud_interdiction.enrichment.device_session import DeviceSession
from fraud_interdiction.enrichment.merchant import Merchant
from fraud_interdiction.enrichment.model_score import ModelScore
from fraud_interdiction.enrichment.velocity import Velocity

Enrichment = (
    ClientContext
    | fan(Counterparty, DeviceSession, Merchant, Velocity, ModelScore)
    | Completeness
)

__all__ = ["Enrichment"]
