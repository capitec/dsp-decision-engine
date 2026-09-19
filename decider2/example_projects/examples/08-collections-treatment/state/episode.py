"""Episodes — the frame within which `path_position` means anything.

An episode opens when days_past_due moves 0 -> positive and closes after 5
consecutive days at 0. `episode_id` is the join key for "promises this episode",
"max intensity this episode", "attempts at this position".

WHY THIS IS NOT A STORED FIELD. Spec §13.2 asks where temporal state lives
between runs. If `episode_id` is a mutable column that yesterday's run wrote,
then a replay of 2026-03-11 in 2029 reads a column written in 2026 by rules
nobody can now name, and acceptance criterion 3 fails. So the episode is a FOLD
over BUCKET_ROLLS, and `episode_id` is content-derived:

    episode_id = stable_hash64(account_id, episode_opened_on)

which means it is the same integer whoever computes it, whenever, from the same
events — including the live-call path, which computes it from one account's roll
history in about 40 microseconds and never consults a database for it.

The cost of deriving rather than storing is a fold over up to 24 months of roll
observations per account, 55.2M rows, every day. That is one frame-tier pass and
it is affordable. The fold for PATH POSITION is not — see path/sequence.py for
the checkpoint mechanism, and note that episode identity is deliberately kept
cheap enough not to need one, because a checkpoint you can avoid is a
reproducibility hazard you can avoid.
"""

from decider2 import fold, module, param, step, stable_hash64
from decider2.types import Date, i1, i2, i4, i8

from ..timelines.streams import BUCKET_ROLLS
from ..timelines.calendar import age_days


class EpisodeFields:
    episode_id: i8
    episode_opened_on: Date
    episode_age_days: i4
    entry_bucket_code: i1
    is_chronic_reager: bool
    rolled_since_yesterday: bool
    rolled_to_worse: bool
    re_entry_class: i1        # 0 normal, 1 re-default<90d, 2 cured-twice, 3 chronic


@fold(over=BUCKET_ROLLS, window="months(24)", emits=EpisodeFields)
def episode_fold(state, obs, cure_close_days: i2 = param(5, ge=1, le=30)):
    """Open on 0 -> positive; close after `cure_close_days` consecutive days at 0."""
    pass  # pure (state, event) -> state; the same function runs batch and live


@step(output="re_entry_class")
def classify_re_entry(
    times_cured_12m: i2,
    days_since_last_cure: i4 | None,
    episodes_12m: i2,
    re_default_window_days: i4 = param(90, ge=0, le=365),
    chronic_episode_count: i2 = param(4, ge=2, le=12),
) -> i1:
    """Spec §5.5 re-entry table. An account that cures and re-defaults is not a new account."""
    pass


@step(output="rolled_to_worse")
def detect_roll(arrears_bucket_code: i1, bucket_yesterday: i1 | None) -> bool:
    """A roll mid-sequence is asymmetric (spec §5.5) and both halves are needed."""
    pass


Episodes = module(
    episode_fold,
    classify_re_entry,
    detect_roll,
    name="episodes",
    taps=["episode_id", "re_entry_class", "rolled_to_worse"],
)
