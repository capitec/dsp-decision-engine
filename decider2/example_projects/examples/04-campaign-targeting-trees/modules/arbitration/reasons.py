"""Arbitration refusal codes — spec §5.6 req 3.

*"Not selected" without a reason is not acceptable output, because the campaign
owner's first question every month is "how much of my population did I lose and
to whom".*

`REF_RANK` carries a detail column naming the campaign that won, which is the
"and to whom" half.  ~52 M rows per monthly cycle land in the arbitration ledger,
one per qualifying pair that did not result in a contact.
"""

REF_RANK = 1        # lost on rank; detail = winning_campaign_id
REF_CAPACITY = 2    # channel capacity exhausted; detail = channel_code
REF_FATIGUE = 3     # fatigue cap reached; detail = which cap
REF_CONTROL = 4     # control or universal holdout
REF_SUSPENDED = 5   # campaign suspended
REF_FAIRNESS = 6    # share cap over the rolling window; detail = campaign_id

CODES = {
    REF_RANK: "Lost on rank to a competing campaign",
    REF_CAPACITY: "Channel capacity exhausted",
    REF_FATIGUE: "Contact fatigue cap reached",
    REF_CONTROL: "Control group — evaluated, deliberately not contacted",
    REF_SUSPENDED: "Campaign suspended",
    REF_FAIRNESS: "Fairness share cap over rolling 6 cycles",
}
