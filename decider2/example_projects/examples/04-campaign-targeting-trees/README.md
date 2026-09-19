# 04 — Campaign targeting trees: an ideal-world sketch

An answer to
[`04-campaign-targeting-trees.md`](../../04-campaign-targeting-trees.md),
written as the shape it would take in a codebase if the authoring surface
could be anything. Nothing here runs. Function bodies are `pass` with a
one-line comment; the value is in the signatures, the composition
expressions, the table files and the artefacts a person actually signs.

Doc 03 is treated as a proposal and departed from repeatedly — thirty-two
numbered departures, each traced to the spec section that forced it and each
marked satisfied / needs-extension / ugly-departed-from / genuinely-awkward,
in [`FRAMEWORK-DEMANDS.md`](FRAMEWORK-DEMANDS.md). Read that file second and
this one first; every invented construct named below (`data_shaped_kind`,
`allocate`, `shadow`, `carried`, `identity_bearing`) is a demand there, not a
settled feature.

---

## 1. The shape, in one page

```
04-campaign-targeting-trees/
  campaigns/
    023-flex-topup/              One directory per campaign. 60 of these.
      campaign.yaml               Owner-authored: population, channels,
                                  measurement design, sign-off requirement.
      trees/v11.json               Published tree versions. Immutable once
      trees/v12.json               written; a new version is a new file.
      identity_maps/v11_to_v12.json  Generated at publication. What a
                                     non-engineer signs before v12 goes live.
      routes/v11.routes.csv        The route dictionary for this version.
      node_meta/v11.node_meta.csv  Node metadata, warehouse-joinable.
      validation/v12.report.json   The eight-plus-two checks, and both
                                   population-impact readings.
      signoff/v12.yaml             ONE bundle, two signatures.
    061-drive-settlement/          Added mid-cycle (scenario 3). Costs one
      campaign.yaml                directory and one registry row.
    041-card-upgrade/
      RETIRED.yaml                 Retired (scenario 4). Nothing deleted.
  modules/                         Shared mechanism, never per-campaign.
    tree/          canonical.py    Node/leaf/edge/route identity — the
                    document.py    single most opinionated file in the sketch.
                    emit.py        Codegen; the portable interpreter.
                    routes.py      Route enumeration and the two-grain trick.
                    identity_map.py  carried_forward/changed/added/removed.
                    validate.py    All ten publication checks.
                    __init__.py    The `CampaignTree` kind declaration.
    suppression/    __init__.py    Bitmask ruleset; the absolute/measurement
                    steps.py       split as a params value.
    holdout/        __init__.py    stable_hash64; control, universal holdout,
                                   variant, the 5% policy arm.
    overlays/       sites.py       The three declared application sites.
                    resolve.py     Stack resolution, four failure modes.
    arbitration/    allocate.py    Population-level constrained assignment.
                    reasons.py     REF_RANK, REF_CAPACITY, REF_FATIGUE, ...
    fatigue/        __init__.py    The one carried (cross-cycle) input.
    pathcapture/    render.py      Rendering a stored path, offline.
  pipelines/
    monthly_cycle.py               14.2 M x up to 60 trees, six hours.
    daily_delta.py                 150k-400k clients, inherits the monthly
                                   manifest, thirty minutes.
    publication.py                 How a tree reaches production. Does not
                                   run in the cycle.
  registries/                      Shared, owned, dated tables.
    campaigns.csv                  An INDEX, regenerated from campaign.yaml.
    channel_capacity.csv           Channel Operations. Monthly.
    feature_bands.json             Campaign Analytics. Quarterly. The one
                                   table flagged identity_bearing.
    suppressions.json              Compliance / Campaign Analytics. Monthly.
  overlays/
    register/2026-10.json          Every overlay, its scope, its expiry.
    stacking_order.json            The declared composition order — its own
                                   approved artefact, not an accident of code.
  warehouse/ddl/paths.sql          The fact table and its three dictionaries,
                                   plus the four queries that justify them.
```

### Why this layout

**`campaigns/<id>-<name>/` is one directory per instance, not per mechanism.**
Doc 07 §1's `modules/` is a reuse-and-audit unit for *code shared across
instances*; nothing in it anticipates sixty independently-owned, weekly-changing
business objects that must individually survive being paused, retired, and
never having their id reused. `campaigns/061-drive-settlement/campaign.yaml`
states the cost of adding one directly: "a `trees/` directory with one
published version, and a row in `registries/campaigns.csv`. Nothing is edited:
no pipeline file, no shared module, no other campaign's anything." Retiring one
is the same story backwards —
`campaigns/041-card-upgrade/RETIRED.yaml` lists what survives (every tree
version, every identity map, "190 node keys remain valid join keys in the
warehouse forever") and states the one absolute rule: `campaign_id 41 is NEVER
reused`. See demand #31.

**Every published tree version is its own file, and identity maps sit beside
them, not inside them.** `trees/v11.json` and `trees/v12.json` are two files,
not one file with a `versions` array, for the same reason project 02's dated
tables are separate files per version: a column is rewritable by a single
edit; a file is immutable and a mutation is detectable. `identity_maps/
v11_to_v12.json` is generated *at* publication and lives one level away from
the trees it compares — it is a review artefact in its own right, seen by the
campaign owner and Credit Risk *before* they read the tree.

**`modules/` is split by mechanism, and no directory is named after a
campaign.** `modules/tree/`, `modules/suppression/`, `modules/holdout/`,
`modules/overlays/`, `modules/arbitration/`, `modules/fatigue/` and
`modules/pathcapture/` each exist because they are used by all sixty trees
identically. The temptation this project resists is a `modules/campaign_23/`
that quietly forks logic per campaign; there is exactly one `CampaignTree`
kind, and campaign 23 differs from campaign 31 only in which document it
loads.

**`overlays/` sits beside `campaigns/`, not inside any one campaign's
directory**, because an overlay's approval, expiry and scope are owned by
Credit Risk Policy or a campaign forum on a schedule that has nothing to do
with when the campaign's tree was last touched (spec §6.2: "an overlay is not
a campaign-local parameter"). `overlays/stacking_order.json` is its own file
for the same reason: the order two overlay *kinds* compose in is a decision
independent of any one campaign, reviewed once and referenced by every
resolution.

**`warehouse/ddl/paths.sql` is checked in and readable without the engine.**
Spec §5.4.1(b) requires the stored path to join "using ordinary database
operations, with no decoding logic that exists only inside the decision
system," so the schema that makes that true is itself part of the authoring
surface, not an implementation detail left to a data engineer downstream.

---

## 2. Walkthrough: a monthly cycle

Read [`pipelines/monthly_cycle.py`](pipelines/monthly_cycle.py) alongside
this. The docstring states the shape before the code does, because the shape
is the whole design decision:

```
for chunk in chunks(64):            # ~220 000 clients, by stable hash bucket
    extract 287 columns once        # the union of every live tree's reads
    for campaign in live:           # 60 kernels over the same resident columns
        evaluate, capture the route, write per-campaign outputs
arbitrate once over everything      # partition="cycle"
```

**Stage 1, population assembly.** `Filter(pl.col("mart_as_at") >=
pl.col("mart_freshness_floor"), on_fail="raise")` — a stale mart does not
warn, it stops the cycle before anything runs. `Join("feature_mart", ...,
snapshot="frozen")` freezes the stage-1 read for the cycle's whole life
(spec §5.1 req 1). `AttachFatigueState` joins the one input this cycle cannot
reconstruct from its own snapshot — see §3.6 below and demand #30.

**Stages 2–4 and 7, per campaign, inside one chunk.** `campaign_stage(23,
cycle_date)` composes one campaign's record-tier work as a single pipeline
expression:

```python
HoldoutAssignment                    # partition="record", before the tree
| GlobalSuppressions                 # partition="record"
| ChannelSuppressions
| ScoreShift                         # overlay site 1
| EvaluationGate(TreeStage)          # absolute suppressions skip; others don't
| AmountCapOverlay                   # overlay site 3
```

`parallel(fuse(campaign_stage(23, cycle.cycle_date)), ...)` runs this sixty
times over the columns already resident in the chunk. The docstring is
explicit about why the loop is nested this way round and not the obvious one:
"400 M evaluations at ~100 ns is 40 seconds of kernel time against a six-hour
window; 14.2 M x 287 columns is 32 GB of extraction. Extract once per chunk,
run 60 kernels while the columns are resident." The per-tree column set is
narrower still — 9 to 74 features per `reads`, computed statically from each
tree document, which is static lineage doing real extraction work rather than
a governance nicety: it is what lets the boundary pull 20 columns for campaign
23 instead of 287.

**The overlays-off twin**, immediately after: `shadow("campaign_stage:*",
thresholds="unadjusted", params={"score_shift": 0.0, "odds_multiplier": 1.0,
"cap_reduction_pct": 0.0}, prefix="unadjusted_")` runs the identical sub-graph
a second time with one input neutralised, for **every** evaluation, producing
`unadjusted_route_digest` and `unadjusted_leaf_key` beside the adjusted
columns. See demand #26 for why this is a combinator and not a second
pipeline.

**Stage 5, bulk pre-assessment.** `Join("project03_preassessment", on=
["client_id", "product_code"], how="left")` — 7.9 M clients through another
project's batch output, joined rather than recomputed, because nothing here
is allowed to advertise more than project 03 would grant (spec §5.5 req 1).

**Stage 6, arbitration.** `| Arbitration` — the one frame-tier module in the
whole per-campaign composition, `partition="cycle"`, over the concatenated
output of all sixty campaign stages. See §3.5.

**Stage 8, output.** `Aggregate(by=["campaign_id", "tree_version",
"overlay_stack_id", "variant", "is_control", "route_digest"], metrics=
{"evaluations": pl.len()}, emit="paths_rollup")` — the frame tier's one job
here is counting, not deciding.

The file closes with two assertions rather than a claim:

```python
assert pipeline.rerun_unit("leaf")             == "campaign"
assert pipeline.rerun_unit("contact_sequence") == "cycle"
```

Through stage 5 a campaign is exactly re-runnable alone; from stage 6 it is
not, because `Arbitration` made the dependency real. Spec §13 Q10 asks for the
honest answer stated rather than discovered at 03:00; these two lines are it
(demand #32).

---

## 3. The hard parts, and how each is expressed

### 3.1 Sixty analyst-authored trees, reaching production without an engineer

[`pipelines/publication.py`](pipelines/publication.py) is the whole answer,
and its docstring states the design's real achievement as a cost table:

```
THRESHOLD MOVE (30/month)          STRUCTURAL CHANGE (5/month)
new tree_version document          new tree_version document
shape_fingerprint UNCHANGED        shape_fingerprint CHANGED
artefact cache HIT                 artefact cache MISS -> compile, ~24 s
zero compilation
node keys for touched nodes move   node keys for touched nodes move
published in ~90 seconds           published in ~3 minutes
```

Both paths run the same ten steps — canonicalise, `validate_tree`, build the
identity map, check every overlay in force still applies, enumerate routes,
emit and compile *only if the shape fingerprint is new*, publish node
metadata, emit the portable interpreter, collect two signatures, write a
manifest amendment — and neither path lets an engineer touch the tree. What
makes "without an engineer" true rather than aspirational is that
[`modules/tree/validate.py`](modules/tree/validate.py)'s ten checks
(`check_total`, `check_reachability`, `check_no_contradiction`,
`check_features`, `check_leaf_outcomes`, `check_prohibited`,
`check_population_impact`, `check_overlay_disposition`,
`check_identity_collisions`, `check_route_budget`) are total over the closed
condition algebra: "every operator has a rule; an operator without one cannot
be added" (demand #14). A rejected submission names the node and the problem
— `validation/v12.report.json`'s `overlay_disposition` check is a live
example, blocking publication because `OV-2026-114` targeted a slot v12 no
longer contains, and naming exactly that.

### 3.2 Node identity, stable across tree versions

[`modules/tree/canonical.py`](modules/tree/canonical.py) states the rule in
one line and means it literally:

```
node_key = "n_" + blake2b64(canonical_condition_text ‖ discriminator)
```

"and *nothing else* goes in. Not ancestry, not depth, not export order, not
tree size, not sibling anything, not the tree id, not the version." The
consequence is demonstrated, not asserted, by the sketch's own worked
publication: `campaigns/023-flex-topup/trees/v11.json` and `trees/v12.json`
differ in exactly one number — node 6's `discretionary_income` threshold, R2
200 to R2 600 — and `identity_maps/v11_to_v12.json` records the result as
**one** `changed` entry (`n_a1f45e9c2b70d863` → `n_b7302ce8149fa65d`) against
**nine** `carried_forward` entries with `match: "exact"`, every sibling node
key byte-identical across the publication. `identity_maps/v11_to_v12.json`'s
own closing note contrasts this against the harder case: "the re-fit in which
the analyst inserted two levels and split one node into three. The modelling
tool renumbered all fourteen surviving nodes in the export; the identity map
shows fourteen `carried_forward` with `match: exact`, because the export's
numbering is not an input to identity."

Where two authored nodes genuinely collide — the same canonical condition
used twice on purpose — the analyst says so explicitly with an `as` label,
hashed into the key and enforced at publication by `check_identity_collisions`
(demand #10). A node's `slot_id` (`s:<node_key>:<ordinal>`) carries the node
key inside its own name, which is what makes a volume dial pointed at a
now-renamed node fail *by construction* rather than by a cross-reference
someone remembered to add (demand #11) — exactly the mechanism that blocked
`OV-2026-114` above. And a leaf's identity is declared narrower than its
payload: `leaf_key` covers outcome, tier and reason label, and deliberately
excludes the amount rule, the channel list and the priority weight, so a
campaign owner tuning `priority_weight` from 0.58 to 0.61 does not break the
leaf's response history (demand #13).

### 3.3 Path capture at ~400 M evaluations a cycle

Three properties, and [`modules/tree/routes.py`](modules/tree/routes.py)
answers all three with one recording, not three mechanisms.

**Compact.** [`warehouse/ddl/paths.sql`](warehouse/ddl/paths.sql)'s `paths`
table carries one `route_digest BIGINT` per evaluation — 8 bytes — computed
by [`modules/tree/emit.py`](modules/tree/emit.py)'s in-kernel fold as the
traversal happens: "two integer ops per node, an immediate XOR of the edge id
and a multiply... at 400 M evaluations x 7.4 nodes that is ~6 billion integer
ops per cycle — call it three seconds." 400 M rows at ~38 bytes compressed is
15 GB against the spec's 120 GB budget, with headroom stated as deliberate:
"the margin is deliberate: the budget must hold when 60 campaigns becomes 90."

**Joinable.** The explosion to "one row per node visited" — the query the
analytics team actually writes — happens against a **150k-row dictionary**,
never against the 400 M-row fact table: `route_node` (the route dictionary,
published once per tree version) and `node_meta` (published once per tree
version, human-readable conditions, no engine required). One campaign-cycle's
7 M-row partition joined to the 150k dictionary is "a broadcast join,
comfortably inside the 90-second budget." One recording, two grains, exactly
answering spec §13 Q3.

**Stable across versions, and human-readable on demand.** Both properties are
inherited from §3.2's identity mechanism and §3.6's rendering below — path
capture does not solve them independently, it depends on them.

The fourth property doc 02 §3.1 does not anticipate is that the equivalence
ladder needs a rung for a *portable* rendering: `emit_portable_interpreter()`
ships a dependency-free evaluator inside the artefact itself, and
`portable == interpreted == stepped == fused` is the assertion that makes
`modules/pathcapture/render.py`'s "THE IMPORT LIST IS THE POINT" comment true
— `import csv, json, datetime`, nothing else, still agreeing with the compiled
kernel two years later (demands #9, #12).

### 3.4 The 14.2 M × 60 batch shape, and the record/frame boundary

Everything in `campaign_stage()` — the tree, both suppressions modules,
`ScoreShift`, `AmountCapOverlay`, `HoldoutAssignment` — declares
`partition="record"` or `partition="campaign"`: scalar logic over one client,
compiled once, evaluated 400 million times with no dependency on any other
row. The single frame-tier module in the whole cycle is `Arbitration`,
`partition="cycle"` (§3.5). That is the entire record/frame split spec §5.6
asks for stated as a property of the composition: `pipeline.rerun_unit(...)`
reads it off directly rather than requiring anyone to reason about which
stage depends on which.

The 60-kernels-per-chunk shape (§2 above) is where this boundary earns its
keep at volume: extraction is the frame tier's one real cost here (32 GB per
chunk pass), and it is paid **once**, not once per campaign, because sixty
record-tier kernels can share one resident column set. `decider2`'s calling
convention (doc 02 §1, records not positional args above ~64 columns) is what
makes a 74-column tree kernel and a 20-column tree kernel both cheap to
dispatch side by side.

### 3.5 Population-level capacity arbitration over per-client decisions

[`modules/arbitration/allocate.py`](modules/arbitration/allocate.py) opens
with the question spec §13 Q9 asks directly — "where does a population-level
constraint sit relative to per-client decisions?" — and answers it with an
invented frame-tier kind rather than an escape hatch (demand #24):

```python
Arbitration = allocate(
    name="arbitration", partition="cycle",
    demand="qualifications", unit=["client_id", "campaign_id"],
    score=pl.col("priority_weight") * pl.col("forum_weight")
          + pl.col("expected_value_cents") * pl.col("ev_weight"),
    order=tie_break("score", descending=True,
                    then="stable_hash64(client_id, campaign_id, cycle_id)"),
    constraints=[
        refusal.when(pl.col("campaign_suspended"), REF_SUSPENDED),
        refusal.when(pl.col("is_control") | pl.col("is_universal_holdout"), REF_CONTROL),
        allocate.cap(by="client_id", limit=param(2, ge=1, le=5), refusal=REF_RANK),
        allocate.capacity(by="channel_code",
            limit=Table("registries/channel_capacity.csv", column="monthly_capacity"),
            refusal=REF_CAPACITY),
        allocate.fairness(share_cap=param(0.35, ...), floor_of_rank_one=param(0.60, ...),
                          refusal=REF_FAIRNESS),
    ],
    method="rank_cut_repair",
    reports=["channel_utilisation", "refused_demand_value", "rank_one_fill_rate"],
)
```

`~61 M qualifications` collapse to `~9.1 M contacted` here, and every one of
the other ~52 M rows carries a named reason from
[`modules/arbitration/reasons.py`](modules/arbitration/reasons.py) — `REF_RANK`
even names the campaign that won. Three guarantees are properties of the kind,
not of this project's discipline: **totality** (`refusal=` is required, so
"not selected, no reason" cannot be expressed), **determinism** (the tie-break
is declared, never row order), and **accounting** (channel utilisation and
refused-demand value are outputs of `allocate`, not a report computed
afterwards). Scenario 7 — outbound call capacity cut from 260 000 to 150 000
for three months — is then a single row edit in
`registries/channel_capacity.csv`; no campaign, no tree, no pipeline file
changes, and the value of the demand `Arbitration` refuses because of it is
reported to the campaign forum by the same mechanism that reports every other
refusal.

### 3.6 Deterministic holdout, with control clients fully evaluated

[`modules/holdout/__init__.py`](modules/holdout/__init__.py) states the
requirement as an absence: "no `if control: skip` anywhere in the skeleton, so
there is nothing for a performance optimisation to find." Control, universal
holdout and variant membership are each `stable_hash64(client_id, campaign_id,
design_version)` and nothing else — no stored assignment table, no dependence
on processing order (demand #19). The control flag enters the pipeline for
the first time inside `Arbitration`, as `REF_CONTROL` — a refusal reason, not
a gate — so a control client walks the entire tree, reaches a leaf, and gets
a path recorded exactly like a treated client. `warehouse/ddl/paths.sql`'s
`paths` table says so in its own comment: `is_control BOOLEAN NOT NULL --
CONTROLS HAVE PATHS. §5.7 req 4`.

Variant selection — champion versus challenger — is modelled as an n-way
`Branch` over the *tree versions themselves* rather than as a parameter of one
tree, which is doc 03 §8.2 applied exactly as designed:
`pipelines/monthly_cycle.py`'s `TreeStage = Branch("variant_index",
[CampaignTree.of(champion), CampaignTree.of(challenger_a), ...])` keeps "which
tree ran" visible in the composition expression rather than hidden in a
lookup. And because assignment is a pure function of identifiers,
`enumerate_movers(old_version, new_version, client_ids)` answers scenario 6's
"which clients moved, and which historical comparisons are now broken" without
running either cycle (demand #29).

### 3.7 Nineteen suppressions, individually attributable

[`modules/suppression/__init__.py`](modules/suppression/__init__.py) packs all
nineteen of `registries/suppressions.json`'s enumerated suppressions into one
`int64` bitmask column per scope:

```python
GlobalSuppressions = ruleset(
    name="suppressions_global", writes=["supp_mask_global"],
    interior="registries/suppressions.json",
    output_kind="flags", flag_dictionary="warehouse/suppression_bit.csv",
)
```

"a client suppressed by four rules shows four" costs one column, exploded
against a published bit dictionary the warehouse joins with an ordinary shift
and mask — the same pattern `route_node` already uses for path capture, on
purpose, "so the analytics team learns one pattern" (demand #20). The
absolute/measurement split — which suppressions skip evaluation entirely
(S01, S11, S14) versus which must still be evaluated and path-recorded because
"the analytics team reports the population that *would* have been targeted"
(S07, S23, S30) — is a single `param()`-bound bitmask compared against
`supp_mask_global`, so Compliance reclassifying a suppression is "no compile,
no staged swap, no engineer" (demand #21, and the file's own words: "the
cleanest win in this sketch"). Five suppressions too specific for a registry
row — `in_cooling_off`, `product_held_at_or_above_tier`, `fatigue_exceeded` —
are registered steps referenced from the registry by id, per doc 08 §3.2's own
rule that a derived value is a step, not an expression string.

### 3.8 Overlays on a published tree that must not change node identity

[`modules/overlays/sites.py`](modules/overlays/sites.py) names exactly three
places an overlay may change an answer — `ScoreShift`, the thresholds array of
`CampaignTreeStage`, and `AmountCapOverlay` — and all three are modules
present in the skeleton whether or not any overlay is currently in force
(demand #25). The mechanism that makes "an overlay is not an edit" true by
construction rather than by discipline is in `modules/tree/emit.py`: "No
literal is an immediate. Every threshold reads `thr[17]` from the thresholds
array" — a volume dial substitutes an array element at run time, and the
compiled code, and therefore `node_key`, never moves.

[`modules/overlays/resolve.py`](modules/overlays/resolve.py) resolves the
stack `as_at=cycle_date`, never today (demand #5), in the order
`overlays/stacking_order.json` declares as its own approved artefact (demand
#16), and fails the cycle by name on four conditions — `EXPIRED`,
`OUT_OF_SCOPE`, `MATCHES_NOTHING`, `DANGLING_TARGET` (demand #17) — with
approval separation checked on the overlay's declared *kind*, so a volume dial
relabelled as a risk overlay cannot launder its way past the wrong committee
(demand #18). `overlays/register/2026-10.json`'s `OV-2026-114` is the sketch's
worked failure: a volume dial targeting `s:n_a1f45e9c2b70d863:0`, withdrawn on
2026-09-24 because campaign 23's v12 folds the same move into the tree and
renames the node the slot depended on — `DANGLING_TARGET`, caught, not
silently reinterpreted. The unadjusted twin this all composes with is
`shadow(...)` (§2, demand #26): every evaluation, not a sample, so `§9.5`'s
"what would we have done without the overlays" is answerable from stored
columns rather than a special run.

---

## 4. What an analyst sees and edits

Four documents, and none of them requires reading Python.

**A campaign owner** edits `campaigns/023-flex-topup/campaign.yaml` — the
candidate population expression, permitted channels, measurement design,
publication bounds. The file states its own scope in a comment: adding
campaign 61 costs this file and a `trees/` directory; nothing else moves.

**Campaign Analytics and Credit Risk** both sign
`campaigns/023-flex-topup/signoff/v12.yaml` — one bundle, one hash
(`bundle_sha256`), two signatures, covering the tree, the identity map, the
population impact with and without overlays, and the disposition of every
overlay in force (spec §5.9's exact requirement: "not five separate approvals
of five separate files, because what the client experiences is the combined
effect"). The `combined_effect.statement` is prose a non-engineer wrote,
describing a tree tightening and an overlay withdrawal landing in the same
cycle as one number.

**Both of them read `identity_maps/v11_to_v12.json` before signing**, not
after — it is explicitly "seen BEFORE sign-off," and its
`comparability_note` is the artefact that tells a campaign owner, in plain
language, which node's month-on-month volume series is about to break and
why.

**Credit Risk Policy and the campaign forum** each own their half of
`overlays/register/2026-10.json` — every overlay carries a description, a
rationale in business terms, an approval reference, an owner and a mandatory
review date, and the file's own annotations narrate two real failures against
it: `OV-2026-114` withdrawn at publication, and `OV-2025-031` found still in
force eleven months after its review date lapsed, which the next cycle simply
refuses to run under (scenario 17).

---

## 5. Explaining one client's targeting

[`modules/pathcapture/render.py`](modules/pathcapture/render.py) renders
client 8 412 907, cycle 2026-09, campaign 23, tree version 11, from three
stored artefacts and nothing running:

```
node n_9f2c41a7b0e35d18   "L1 facility gate"
    has_active_flex_loan = true
    months_on_book_flex = 31 (>= 6)
    flex_term_remaining_months = 22 (>= 4)                  HELD     -> node 2
node n_4b71e0cd5a92f36c   "L2 settlement/arrears gate"
    settlement_ratio = 0.71 (<= 0.65 FAILS)
    payments_missed_12m = 0 (= 0)                           NOT HELD -> node 4
    ...
leaf l_cf2839a54e0b761d   TARGET, tier B, R84 000, SMS then in-app,
                          priority 0.58, "Standard top-up, SMS responsive"

overlay stack 4471100:  OV-2026-114 volume dial, node
                        n_a1f45e9c2b70d863 threshold 2 200 (unchanged this
                        cycle; the dial moved it to 2 600 from 2026-10-01)
control: no.  variant: champion.  unadjusted leaf: same.
```

Node 2 is printed even though its condition failed — the stored route holds
it (§3.3), so rendering is a lookup, not a re-walk of a tree that may no
longer exist in this shape. The file's closing comment is the point of the
whole section: "THE IMPORT LIST IS THE POINT — `import csv, json, datetime`...
No decider2, no numba, no polars, no database driver." The staff member
producing this answer within spec §9.1's five business days is not an
engineer, and eighteen months from now the decision system will not be the
same software; what must still exist is the CSV, the JSON and the two hundred
lines of Python that shipped inside the artefact at publication.

`plain_language_reason(leaf_key, node_meta, reason_labels)` is what the
contact centre actually reads — derived from the leaf's `reason_label` plus
the two or three highest-discrimination conditions on the stored route, so
"sixty campaigns x 190 leaves is 11 400 hand-written sentences nobody will
maintain" becomes "220 reason labels plus a path," generated, not authored per
campaign.

---

## 6. What changes when a value moves, versus when a tree's structure moves

| The change | Class | Owner | Costs | Reaches production by |
|---|---|---|---|---|
| Node 6 threshold R2 200 → R2 600, as an overlay | overlay value | Campaign owner + forum | nothing | `overlays/register/*.json`, expiring |
| The same move, folded into the tree | structure, cache **hit** | Campaign owner + Credit Risk | ~90 seconds, no compile | `trees/v12.json`; node 6's key changes, its siblings' do not |
| An analyst re-fits, inserts two levels, splits a node | structure, cache **miss** | Campaign owner + Credit Risk | ~3 minutes, compiles once | new `shape_fingerprint`; fourteen surviving nodes keep their keys regardless |
| Compliance reclassifies S23 absolute → measurement | value | Compliance | nothing | one `param()` bound, `absolute_mask` regenerated |
| Outbound call capacity 260k → 150k for 3 months | value | Channel Operations | nothing | one row in `channel_capacity.csv`; `Arbitration` reallocates |
| Fatigue cap 4 → 3 programme-wide | value | Channel Operations | nothing | one `param()` bound in `suppression/steps.py` |
| A band edge moves, touching 38 trees | **identity-bearing table** | Campaign Analytics | 38 identity maps, 38 sign-offs | blocked until every affected owner signs — see `registries/feature_bands.json` |
| A new campaign | nothing existing | New campaign owner | nothing | one directory, one registry row |
| A campaign retires | state only | Campaign owner | nothing | `state: retired`; every artefact kept forever, id never reused |
| A new overlay | overlay (its own class) | Kind-appropriate approver | nothing | `overlays/register/*.json`, scoped, dated, expiring |

The pattern worth naming: **every change scenario the spec lists as arriving
routinely (7, 8, 15, 16) is a values change**, and the framework already
serves it well — see demand #21's SAT callout. The two genuinely structural
rows (a re-fit, a band edit) are exactly the two cases this sketch spends the
most invented machinery on (§3.2, §3.8, demand #22), because they are the two
places where "just a number moved" would otherwise quietly break two years of
comparable history.

---

## Note on a fix

`campaigns/023-flex-topup/routes/v11.routes.csv`'s header states "16 feasible
routes, 74 rows exploded" (matching `validation/v12.report.json`'s "16
routes"), but the file as found showed only 7 of those routes (32 of the 74
rows) with no indication the listing was partial. That is an outright
inconsistency between the file's own header and its own body, not a
difference of opinion about the tree. The fix follows the sketch's own
established idiom for a deliberately shortened list —
`registries/campaigns.csv` ends with `# ... 53 more rows.` — and adds one
trailing comment to `v11.routes.csv` stating the elision explicitly and noting
that the seven shown routes already exercise every node and both `MULTI`
arrivals. No row was invented or altered.
