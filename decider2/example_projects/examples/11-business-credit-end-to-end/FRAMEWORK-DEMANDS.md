# FRAMEWORK-DEMANDS

Every numbered demand is forced by a section of
`11-business-credit-end-to-end.md` and answered by a specific construct in
this tree. Numbers match the `FRAMEWORK-DEMANDS D<n>` comments already
present in the code (`consumed/manifest.py`, `modules/covenants/
definition.py`, `modules/security/allocation.py`, `modules/request/
relationship.py`, `vocabulary.py`, `consumed/FORKS.toml`, `modules/
authority/routing.py`, `modules/covenants/schedule.py`, `lifecycle/exit/
handoff.py`, `time/comparability.py`, `modules/watchlist/signals.py`); the
rest fill gaps in that sequence. Numbering is by discovery order in the
tree, not by section, so D1 and D24 can sit in the same section below.

Status is one of three:

- **satisfied** — the existing mechanism, used as designed, answers this.
- **needs extension** — the right shape exists but not the specific
  mechanism this project requires.
- **doc 03 would make this ugly** — the existing mechanism technically
  applies but produces something nobody would trust, so the sketch invented
  a narrower construct instead.

Reuse and time demands are first: they are what this project exists to
surface (spec §2.2, §5.17).

---

## Reuse demands

### D1 — a consumed component must look different from local logic, on the page

*Spec §2.2 consequence 1; §4.9; §9.4 req. 2. Tree: `consumed/__init__.py`.
Status: doc 03 would make this ugly.*

Doc 03 has one node type, `Module`, used identically whether an engineer
wrote it yesterday or project 05 published it under a different Credit
Committee's approval. That is correct for authoring and wrong for
governance: a policy analyst who can open a consumed node's interior has
silently taken on an approval that isn't this project's to give. `Consumed`
is proposed as a wrapper with three properties `Module` lacks: no interior
document (`decider export --interiors` skips it), an owner and approval
reference in its header instead of this project's, and its pinned version
printed at the use site.

### D4 — a component's major version must be selectable per assessment, not per deployment

*Spec §5.17.4 scenario 1; §8.2; 10 acceptance 26; 13-Q16. Tree: `consumed/
manifest.py`. Status: needs extension.*

Doc 08 §4 gives a process one active generation and at most one staging,
selected by a pointer swap — right for a config change, wrong here, because
this project needs **three** component versions live simultaneously in one
process, selected per row: EP-1 on today's major, an in-flight EP-3 cohort
pinned to last quarter's, a 2027 replay on whichever major was current
then. The sketch reuses `Branch`, one arm per live major, all compiled at
build — but the demand is new: declaring "N majors, selected from an input
column" as a first-class fact about the binding, not an ad hoc `if`.

### D6 — a manifest-pinned branch's arms must be statically forbidden from fusing

*Spec §5.17.4; doc 02 §1.1. Tree: `consumed/manifest.py` (`fuse=False`).
Status: needs extension.*

Doc 03 §8.1's default is already "nothing fuses implicitly," so this looks
satisfied — it is not. The two arms of a manifest-pinned `Branch` are the
most tempting fusion candidate in the project, and doc 03 has no way to
declare "this `Branch` must never be fused," only "isn't currently
requested." A future engineer chasing a performance win can write
`fuse(Affordability)` and the build will not stop them, though fusing two
governed component majors is a correctness-of-budget violation, not a
performance tradeoff.

### D7 — a sixth scope, over role rather than instance, carrying vocabulary, parameters, nulls and disclosure together

*Spec §5.17.1 (31 role-dependent names, 12 materially misread if dropped);
13-Q13. Tree: `roles.py`. Status: doc 03 would make this ugly.*

The tree's central invention. Doc 03 §5.2's three-layer story — name
matching, project vocabulary, `.at()` — fails here because a library name's
*meaning*, not spelling, depends on which entity it is attached to.
`Vocabulary` is a function from name to name and cannot express "this name
means X for a director and Y for a surety." `.at()` could carry 120
per-instance relabels but hides them in the composition file and can only
rename — not carry a parameterisation (`criticality_class`) or a null
policy ("not applicable to a retired surety"). `role(...)` is proposed as a
scope alongside step, module, branch arm, loop body and pipeline.

### D8 — where lifecycle-only vocabulary lives when no prior project needed it

*Spec §4.8 (20 local names, 15 about time); 13-Q21. Tree: `vocabulary.py`
(`LOCAL`). Status: needs extension, at the library layer.*

Doc 03 §5.2's vocabulary mechanism handles this project's own local names
fine. The open question is where names like `covenant_instance_id`,
meaningless to projects 01–04 and mandatory for 07, 08 and 11, belong once
more than one project needs them. The sketch's answer — a second library,
`lifecycle-core`, distinct from `credit-core` — is a statement about
library boundaries the authoring API is silent on.

### D9 — a fourth null situation, bound to role rather than to the value

*Spec §5.17.3; doc 00 §7.4; 11 scenario 18. Tree: `roles.py` (`nulls=`,
`NOT_APPLICABLE`); `FORKS.toml` pressure 7. Status: AT RISK — the sketch's
own answer is flagged as a fork.*

Doc 03 §1's three null tiers don't distinguish "not collected" from "not
applicable to this role." Collapsing a corporate guarantor's non-existent
`applicant_age_years` into "could not be established" drags the 75%
coverage ratio down for a structural reason, not a data-quality one.
`NOT_APPLICABLE.when(...)` is the sketch's answer, but `FORKS.toml` records
it as pressure #7, `AT RISK`: "adding a fourth null situation to `core`
touches six consumers; a local convention touches none." This demand is
explicitly unresolved — the sketch is honest that its own answer might be
the fork.

### D20 — an undeclared reuse gap must be structurally impossible

*Spec §5.17.2; 10 acceptance 25; 13-Q14. Tree: `consumed/GAPS.toml` (checks
G1, G2). Status: needs extension.*

Doc 03 §5.1's `contract=` freezes what a *publisher* promises. Nothing
freezes the symmetrical case: a build-checked document declaring, for every
name this project writes that a consumed component also declares as an
output, a resolution among EXTEND/COMPOSE/PARAMETERISE — with an undeclared
one a hard build error. `GAPS.toml`'s own comment: the resolution "whoever
hits it first, at 2am, during a batch" cannot be reached without editing
the file, and editing the file is a PR.

### D25 — the fork register is a separate governed document from the gap register

*Spec §5.17.6; 10 acceptance 23. Tree: `consumed/FORKS.toml`. Status:
satisfied by doc 04 §2.1.*

Doc 04 §2.1 is explicit that enforcement of *where* a change happens is CI
and review policy, not framework machinery. `FORKS.toml` requiring a
different approval path (Credit Systems plus the forked component's owner)
than `GAPS.toml` is exactly that policy attaching to a boundary the
framework already makes visible. Listed to show the pattern is not always
a gap.

### D24 — reusing a shaped construct for a narrower scope than it was built for

*Spec §5.5.5. Tree: `modules/covenants/waiver.py` (`adjustment_shaped(...,
scope_by=)`). Status: satisfied by doc 03.*

A genuine reuse win: `core.adjustments`'s population-scoping generalises to
instance-scoping through the existing `scope_by=` parameter with no new
mechanism — "a reuse win nobody would have found by looking at the
component's name." Included to balance the list.

---

## Time demands

### D2 — a fourth change class: append-only, immutable once bound, superseded by nobody

*Spec §6.3 (H2). Tree: `modules/covenants/definition.py`. Status: needs
extension.*

Doc 08 §2's three classes — values, interiors, skeleton — have no row for
an artefact that is neither. Editing a covenant definition in place
retrospectively changes live contract terms, so it isn't a value. An
interior change applies to *everything* behind the interface, where this
must apply to nothing already bound, so it isn't an interior. Nothing about
pipeline shape changes, so it isn't skeleton. `append_only_artefact(...,
immutable_once_bound=True, closure=[...])` is the fourth class: bound,
never superseded, retired only when the last instance closes (up to twenty
years). Standardising it across a live book is "an amendment programme, not
a library edit" — 40,000 client consents, eighteen months.

### D16 — two version-resolution rules as two incompatible types, not one convention

*H2; spec §5.5.1, §6.3, §6.4; 13-Q3. Tree: `time/dating.py`. Status: needs
extension.*

Doc 03 and doc 08 share one dating story: everything resolves by
`decision_date`. Applied uniformly to a covenant definition this produces a
plausible, silent, legally consequential wrong answer, and it has to be
that the wrong call does not typecheck rather than a review convention that
gets missed. `dated_table(...)` returns `Dated[T]`, no `.bound_to()`.
`bound_table(...)` returns `Bound[T]`, with **no `.in_force_at(date)` at
all** — absent, not undocumented. Three build-time checks follow: a
`Bound[T]` read needs the binding key in scope; `contractual=True` forbids
reading `Dated[T]`; a bound version cannot be superseded while any live
instance's `closure` binds it.

### D10 — a bi-temporal fact store must refuse to resolve an unqualified read

*H3; spec §5.13.2, §5.13.4; 10 acceptance 17; 13-Q4. Tree: `time/
bitemporal.py` (`unqualified=None`). Status: needs extension.*

Doc 03 §5.2's `.at()` works when there's a right default and a rare
exception; it fails here because both `known` and `actual` are correct
answers to different questions, and neither errs when misapplied. The
knowledge-date view blinds a covenant to a change it exists to catch; the
effective-date view makes a correct replay appear to run on a structure the
Bank did not have. `bitemporal_source(..., unqualified=None)` makes the
*namespace itself* default-free — a bare `entities` read gets a build error
naming both candidates and the spec section that says which is wanted.

### D17 — "not comparable" as a first-class, arithmetic-refusing value

*Spec §5.10.4 req. 5; 10 acceptance 3, 4; 13-Q6. Tree: `time/
comparability.py` (`Compared[T]`, `delta()`). Status: needs extension.*

Doc 03 §1's three null tiers are about *missing* data. This is different:
the value exists on both sides, both individually correct, and they cannot
share a column. A `float | None` would let a report silently subtract
`None` as zero; the required behaviour is a **refusal with a reason code**
a downstream step cannot ignore. `compared_type(..., arithmetic=
"basis_checked")` carries `(value, basis, basis_ref)`; `delta()`, the only
defined subtraction, will not compile across two operands whose basis
cannot be shown equal — 10 acceptance 3 becomes something the report
builder has no path to construct, not a runtime check.

### D14 — the cost of making comparability universal, stated rather than hidden

*Spec §5.10.5. Tree: `time/comparability.py`. Status: needs extension.*

Distinct from D17: that asks for the type; this is the demand that using it
everywhere a grade-bearing value exists — roughly 40 columns — stays
affordable at the compiled tier. The sketch's own accounting: at the
compiled tier this "costs a branch," cheap per column but multiplied across
every grade-bearing output. Nothing in the docs quantifies what a compound
value kind (versus a primitive or `Optional`) costs through every step
signature and tap at this scale — this project is the first data point.

### D18 — one artefact resolved under both rules, depending on which direction it's read

*Spec §5.10.4 req. 1, 3. Tree: `time/master_scale.py`. Status: doc 03 would
make this ugly.*

The master scale registry is read by `decision_date` when a *new* grade is
assigned, and bound to `master_scale_version` — never re-resolved by
today's date — when an existing grade is restated. A single `dated_table`
would let a stale replay pick up today's scale; a single `bound_table` has
nothing to bind a new assessment to yet. The sketch's answer is two
declared bindings (`master_scale_current` as `POLICY`, `master_scale_
registry` as `CONTRACT`) over one conceptual registry — composing two
existing constructs rather than inventing a third, flagged in its own file
because the double-resolution is unusual enough to deserve one.

### D5 — a facility's decision history is an input the caller resolves, never a store the flow reads

*13-Q2; 09 §5.15 item 8. Tree: `modules/request/relationship.py`
(`facility_decision_history`). Status: satisfied by doc 03.*

Doc 03 §2's wiring rule already fits: a name matching nothing in the module
is a leaf input. The predecessor decision of record is exactly that — an
input column the caller resolves before the flow runs, not a store read
mid-execution, which 09 §5.15 item 8 names as a replay defect elsewhere. No
new mechanism needed; the cost — an 8-hour batch doing a 15,000-row
predecessor join before deciding anything — is paid explicitly at the frame
tier instead.

### D21 — a batch-wide version freeze wider than the generation pointer

*Spec §5.15.2 req. 2; §8.2; 5.15.1 collision 1. Tree: `consumed/
manifest.py` (`freeze_for_batch`); `pipelines/annual_review.py`
(`batch_pin`). Status: needs extension.*

Doc 08 §4 already guarantees `apply`/`score` reads the generation pointer
once per invocation, stopping a batch straddling a *skeleton* swap. It says
nothing about the component manifest or the 63 policy tables — separate
mechanisms, not generations. `batch_pin(..., covers=["generation",
"component_manifest", "dated_tables"])` freezes all three for a cohort,
closing the gap doc 08 §4 names itself: "two config objects touched at
different moments in one batch can be on different versions... with
nothing recording where the boundary fell."

---

## Cross-record, cascade and authority demands

### D3 — a cross-record operation over a set decided separately, months apart, by different people

*H4; spec §5.11; 10 acceptance 12, 13, 14; 13-Q8. Tree: `modules/security/
allocation.py` (`cross_record`). Status: doc 03 would make this ugly.*

Stated in the file itself: "every construct in doc 03 — step, module,
Branch, Loop — is about one record." Required together: a declared
invariant checkable both on the decision path and nightly over the whole
book (`invariant=sum_le(...)`); a declared consequence edge
(`reopens=every_member_sharing(...)`) from which the affected set, the
authority union and the client list are computed statically; and ordering
independence extended to the case spec §8.2 calls "most likely to violate
it." `cross_record(...)` is a new combinator alongside `Branch` and `Loop`,
scoped over a set rather than a record.

**Note — a tenth cascade trigger arises from the same file.**
`revaluation_consequences` treats a collateral revaluation as a
cascade-shaped event though it isn't among spec §5.12.1's nine. Handled by
routing through `Cascade` with `hops=1` rather than a parallel mechanism —
satisfied by composition, not a new primitive.

### D22 — a bounded propagation construct where every bound is a constructor argument

*H5; spec §5.12.2; 10 acceptance 15; 13-Q9. Tree: `pipelines/cascade.py`
(`Cascade(...)`). Status: doc 03 would make this ugly.*

No combinator in doc 03 expresses "a change to one subject produces
decisions about others, bounded by depth, materiality, fan-out and
re-entrancy." A `Loop` bounds by `max_iterations`; a cascade's bound is
graph depth, a materiality floor that stops most candidates before they
propagate at all, a fan-out cap that diverts rather than fails, and
re-entrancy de-duplication across cross-holdings. 13-Q9's answer here: all
four bounds are required constructor arguments, so a cascade with no
fan-out cap does not construct — 10 acceptance 15 ("enforced, not
monitored") becomes a property of the type.

### D23 — two disclosure-scoped outputs from one decision, structurally prevented from merging

*Spec §5.12.4, §9.3; 10 acceptance 16; 9.7. Tree: `pipelines/cascade.py`
(`dual_attribution`); `consumed/core_library.py` (`reason_codes.
contributes(requires_attribute=...)`). Status: needs extension.*

Doc 04 §5.2's audit record is one record. This needs two from one
decision — a complete internal attribution naming business A, and a
communicable explanation to business B naming none of it — with a renderer
that cannot reach the internal one by construction, not a "communicable"
flag per field a new field can default into unsafely.
`dual_attribution(internal=..., communicable=..., default="withheld")`
proposes two declared outputs with two disclosure classes. The dependent
library demand — a `communicable_to_connected_business` attribute on the
reason registry — is declared as gap `third_party_disclosure_class`,
EXTEND, owner Compliance, not this project's to resolve alone.

### D11 — an approval bound to a content fingerprint, invalidated automatically by any write inside it

*Spec §5.14.3 req. 2, 3; 10 acceptance 20. Tree: `modules/authority/
routing.py` (`approval_binds_to`). Status: needs extension.*

Static lineage already answers "what can affect z" (doc 02 §5, doc 04 §3).
What's missing is a declared *binding* that leans on that lineage to
invalidate an approval automatically the moment anything inside a named
fingerprint's lineage is rewritten. Without it, "a structure change that
would have required higher authority invalidates the approval,
automatically" is a rule written into every one of 1,900 decision points —
exactly what 14 teams collectively forget. `approval_binds_to(...)` composes
with lineage rather than duplicating it.

---

## Organisation and governance-as-data demands

### D19 — a governance-owned matrix is the control flow itself, not a description of it

*Spec §5.1, §8.1; 13-Q20, 13-Q22; 10 acceptance 28. Tree: `pipelines/
entry_points.py` (`scoped_by`); `lifecycle/amendment/reopen.py`
(`assert_matrix_total`); `modules/appetite/facility_types.py`
(`dispositioned_by`). Status: needs extension.*

Doc 07 §4 deliberately keeps composition out of config. This project's nine
entry points and fourteen amendment kinds are each a *scoped subset* of one
already-reviewed pipeline, determined by a governance-owned CSV —
composition-shaped without being composition in doc 07 §4's sense, because
only *which parts run* is governance data, not the pipeline itself.
`scoped_by(matrix, row)` resolves this: run level (FULL/PARTIAL/NOT_RUN) per
phase-part, `assert_matrix_total` checks bidirectionally at build (zero
orphans), and the framework computes the scoped pipeline from the declared
one. Distinct from `Admit.COMPOSITION` (doc 08 §7), which widens what
config may *build*; this narrows what an already-built pipeline *runs*.

### D13 — a consumer that never runs a flow can still freeze what that flow must produce

*Spec §5.15.1 collision 5; §5.15.2 req. 6. Tree: `lifecycle/exit/
handoff.py` (`consumer_contract`). Status: needs extension.*

Doc 03 §5.1's `contract=` freezes what a *publisher* promises. Nothing
freezes what a *read-only downstream consumer* — Recoveries, which never
writes a line into the flow it depends on — requires to keep receiving.
`consumer_contract(...)` is the mirror image: a build that would stop
producing a required field fails, naming the consumer and the field, in the
team that broke it. Spec §5.15.2 req. 6: "their requirements must be
expressible as constraints on the artefact, not as review comments."

### D12 — a covenant schedule as a frame-tier subject generator, not a new graph node kind

*13-Q5. Tree: `modules/covenants/schedule.py` (`Expand`). Status: needs
extension.*

13-Q5 asks whether a schedule is "a fourth thing, or a rule set with a
temporal scope." Neither: a schedule produces the *rows* an ordinary
record-tier pipeline then decides, in the frame tier as a declared
expansion — not a new node kind holding time and firing, which would put a
clock inside the decision engine and make a date unreplayable by
construction. Doc 02 §5 names join, aggregate and filter as stage-one frame
ops; `Expand`'s one-row-in, zero-to-many-rows-out transform is not among
them, and the file states this plainly.

### D15 — impact review must be declarable as mandatory for one named artefact

*10 acceptance 34; spec §5.15.1 collision 3. Tree: `modules/watchlist/
signals.py`. Status: needs extension — the gap is named, not closed.*

`decider2.impact(...)` already exists (doc 08 §5) and does the right
computation. What doc 08 §5 states plainly is that it does not enforce that
anyone runs it. For the watchlist grade boundaries specifically that gap is
not acceptable — Early Warning adding 14 signals against boundaries
calibrated on 172 is spec §5.15.1 collision 3, and "2,800 facilities change
grade, of which 2,640 are scale artefacts" is a report nobody is compelled
to read before activating. The demand is a way to mark one artefact as
requiring a *passing* impact review before its staged interior can
activate, which doc 08 §4's lifecycle does not currently gate on.

---

## What these demands add up to

Thirteen of the demands above ask for a genuinely new construct doc 03 does
not have: `Consumed`, per-assessment `Branch` pinning as a declared fact,
`role()`, `dated_table`/`bound_table` as incompatible types, a bi-temporal
namespace with no default, `Compared[T]`, `cross_record`, `Cascade`,
`dual_attribution`, `approval_binds_to`, `consumer_contract`, `Expand`,
`append_only_artefact`, and `scoped_by`. None is speculative — each is
load-bearing for at least one acceptance criterion in spec §10. Three
demands (D5, D24, the fork-register pattern) are satisfied by using an
existing mechanism deliberately, worth stating because a document of only
gaps would misrepresent how much of doc 03 already works unmodified for
1,280 of this project's 1,900 decision points. Two (D9, D15) are explicitly
unresolved even in the sketch's own terms — named as risks, not closed as
features.
