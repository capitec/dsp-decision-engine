# Cloud-vendor and enterprise decision services / BRMS vs `decider2`

**Date:** 2026-09-21 · **Scope:** §6 of `decider2/docs/research/decision-engine-landscape.md`, verified against
primary sources. **Focus axes:** rule lifecycle, parameterisation, governance artefacts, testing/simulation,
deployment, data model, explainability.

**Method.** Every factual claim about a product carries a first-party URL fetched on 2026-09-21. The landscape
note recorded IBM Docs as JS-only shells; that was only half true — the *product-code* static paths
(`ibm.com/docs/SSQP76_.../com.ibm.odm.*/topics/*.html`), the ODM-on-Cloud and ODM-on-Business-Automation
variants (`?topic=` on hosts `odmoc` / `dbaoc`) and versioned `?topic=` slugs all serve real content, so most
of §6's ODM "unverified" markers are now retired. Claims about `decider2` cite `decider2/<path>:<line>` against
the working tree (which has uncommitted changes in `src/` — line numbers are as of this read). Nothing under
`decider2/` was modified.

---

## 1. IBM Operational Decision Manager (ODM) and Automation Decision Services (ADS)

**Authoring and data model.** ODM's data model is a genuine three-layer stack that `decider2` has no analogue
for. The *execution object model* (XOM) "is the runtime model against which rules are run… the base
implementation of the BOM"; the *business object model* (BOM) is derived from the XOM and holds "the classes
and methods that rules act on"; and *verbalization* attaches a natural-language phrase to each BOM member so
rules read as Business Action Language sentences
([rule vocabulary](https://www.ibm.com/docs/en/odm/8.0.0?topic=server-rule-vocabulary),
[BOM](https://www.ibm.com/docs/en/odm/9.0.0?topic=bom-introducing-business-object-model)). Rules are authored
as BAL action rules, decision tables, decision trees and ruleflows
([product](https://www.ibm.com/products/operational-decision-manager)). The Business console is explicitly
"the environment for business users who manage and govern the lifecycle of decisions" and lets them edit
decision services, build rules in the Intellirule editor, manage ruleflows, "validate rules through test
suites and simulations", "create snapshots, manage branches", and follow streams with comments
([consoles](https://www.ibm.com/docs/SSQP76_8.10.x/com.ibm.odm.dcenter.consoles/topics/odm_dcenter_consoles.html)).

**Lifecycle.** The decision governance framework is the strongest lifecycle model in this group. "Decision
Center uses releases to capture and trace all the changes that are related to a purpose and period in time"
and "change activities to manage the work of participants who are collaborating toward a goal, in the larger
context of a release"; validation activities run *after* change activities are approved; "You complete a
release… and then deploy its content as a decision service"
([governance framework](https://www.ibm.com/docs/en/dbaoc?topic=services-managing-changes-decision-governance-framework)).
Approval is role-gated: "only users with the proper responsibilities can approve a release and deploy it", and
changes "need to be reviewed and approved, before the activity or release can be marked as complete and
deployed" ([using governance](https://www.ibm.com/docs/en/dbaoc?topic=services-using-governance-decision)).

**Effective dating.** ODM does not put valid-from/valid-to in the engine's dispatch; it exposes rule
properties and *runtime rule selection*: "You can define a selection filter on a rule task to specify
dynamically what rules of the rule task must run… For example, you can specify that the expiry date of the
rule is after the date of the loan"
([runtime rule selection](https://www.ibm.com/docs/SSFL4K/com.ibm.odm.dserver.rules.designer.dev/orchestrating_topics/con_orch_runtime_rule_selection.html)).
So effective dating is a *filter over rule metadata evaluated against a business date supplied on the
request* — which is exactly the shape `decider2` could adopt without a new concept.

**Rule analysis.** Consistency checking finds "Rules that are never selected", "Rules that never apply",
"Rules with range violation", "Rules with equivalent conditions", "Equivalent rules", "Redundant rules",
"Conflicting and self-conflicting rules", and checks "conflicts, redundancies, and equivalences" between
decision-table rows and decision-tree leaves, including across structures
([consistency checking](https://www.ibm.com/docs/en/odmoc?topic=analysis-consistency-checking)).

**Testing and simulation.** "You perform the tests and simulations in Decision Center, through test suite and
simulation artifacts." Scenarios are authored in Excel, each with a unique Scenario ID. "When you run a
simulation, the report that is returned provides some business-relevant interpretation of the results, based
on specified key performance indicators (KPIs)." Test reports carry a summary with "the precision level, and
the success rate, that is, the percentage of scenarios that were executed successfully", and classify each
scenario Successful / Failure / Error
([testing and simulation](https://www.ibm.com/docs/en/odm/8.9.1?topic=rulesets-overview-testing-simulation)).

**Decision-level audit.** Decision Warehouse "is a tool of the Rule Execution Server console for monitoring
ruleset execution. It stores execution traces in a database"
([Decision Warehouse](https://www.ibm.com/docs/en/odm/8.11.1?topic=server-decision-warehouse)). The per-ruleset
monitoring options are the most complete decision-record checklist anyone in this group publishes: Execution
Date, Execution Duration, Execution Output, "Total Number of Tasks Executed", "Total Number of Tasks Not
Executed", "Total Number of Rules Fired", "Total Number of Rules Not Fired", "Execution Events – Provides the
full execution tree for the executed ruleset", Ruleset Properties, List of All Tasks, List of Tasks Not
Executed, List of All Rules, List of Rules Not Fired, Bound Object by Rule, System Properties, and "Working
Memory with optional filter"
([monitoring options](https://www.ibm.com/docs/en/odm/8.11.1?topic=execution-setting-decision-warehouse-ruleset-monitoring-options)).
Note the two "not fired / not executed" rows — the negative space is recorded, not just what happened. The
same page warns "BOM serialization can result in poor performance", i.e. the verbose tiers cost money, which
is the same trade-off `decider2` doc 04 §6.5 item 3 names as configurable verbosity. Traces join back to
authoring: from the Decisions table you "View Decision details", expand Decision Trace, and open individual
rules in Decision Center from the Ruleflow Tasks tree
([viewing executed rules](https://www.ibm.com/docs/en/odm/9.0.0?topic=warehouse-viewing-executed-rules-in-decision-center)).

**ADS** is the newer, git-native sibling: decision models (diagram of input/decision nodes), decision tables
with preconditions and "error analysis", text rules, task models, predictive-model and generative-AI nodes; a
REST decision runtime and a Java execution API each with an execution trace; "Managing branches" covering
"creating, merging, and protecting branches"; and unit test scenarios that "you can run at build time"
([overview](https://www.ibm.com/docs/en/ads/25.0.0?topic=overview),
[testing](https://www.ibm.com/docs/en/SSGH5D_25.0.0/com.ibm.ads.test/topics/con_testing_intro.html)). The
trace is "an actual graph or tree of information that represents execution of a decision service", requested
by passing `executionTraceFilters`, "a JSON object that specifies which traces are activated in which format";
omit it and the trace is null
([execution trace](https://www.ibm.com/docs/en/SSGH5D_25.0.0/com.ibm.ads.execute/topics/con_execute_trace_restapi.html)).
Deployment is Kubernetes/OpenShift; ODM ships Apache-2.0 assets for EKS/AKS/GKE/ROKS
([odm-docker-kubernetes](https://github.com/DecisionsDev/odm-docker-kubernetes)). **Neither product publishes
a latency or throughput figure on a first-party page.**

**vs decider2.** decider2 has the *mechanism* for most of ODM's lifecycle and almost none of the *ceremony*.
`ServeHandle` is a real generation machine — `stage()`/`activate()`/`rollback()` with an explicit activation
step (`decider2/src/decider2/runtime/serve.py:253`, `:287`, `:305`), a structure fingerprint that excludes
every params value by construction (`:90`), and `sealed`/`live` modes enforced in code (`:82`). Doc 08 §4
property 4 states the design intent exactly: "Activation is explicit… That is the hook an approval workflow
attaches to; the framework does not implement the workflow"
(`decider2/docs/08-configuration-and-lifecycle.md:469-470`). Against ODM that is a defensible line, but three
gaps are real rather than stylistic. (1) **No releases, no activities, no snapshot/baseline concept.**
`ServeHandle._history` is an in-process LIFO of params dicts (`runtime/serve.py:153`, `:305`) — there is no
named, retained, re-activatable baseline, and doc 08 §4 property 5 concedes "Rollback across a *restart* is
the caller's problem". (2) **No approval field anywhere.** Doc 08 §9 lists this as open O17 — "Does activation
need per-rule approval, or is document-level enough?… it determines whether a rule carries an approval field"
(`:832-836`). (3) **Provenance is specified and then thrown away.** Doc 08 §6.2 makes `origin` "a **required,
non-empty, opaque token**… The framework stores it verbatim, never parses it, and refuses to run without one"
(`:707-711`), but both entry points accept it and delete it: `del origin` at
`decider2/src/decider2/runtime/invoke.py:465` (batch) and `:672` (single record), the docstring conceding
"accepted so the signature matches the spec, but not yet recorded anywhere" (`:452-454`). ODM's weakest axis
against decider2 is performance transparency: decider2 has measured figures for swap cost (0.177 µs),
rollback (3.36 µs, zero compile events), retained generations (2.4 MB for three) and stage-to-serving
(2.56 s at 10 rules, 7.34 s at 30) — doc 08 §4 — and ODM publishes none.

---

## 2. Progress Corticon

**Authoring and data model.** Rulesheets are condition/action grids over a declared Vocabulary, with
Ruleflows sequencing them and natural-language rule statements attached
([Rulesheets](https://docs.progress.com/bundle/corticon-quick-reference-71/page/Rulesheets.html)).

**Rule analysis — the strongest in the field.** Corticon Studio checks a Rulesheet for four classes of logical
problem: conflicts, incompleteness, hidden dependencies and unintended loops. The conflict checker "highlights
conflicting subrules that would execute under the same circumstances" and deliberately stops there: "Corticon.js
Studio does not instruct the rule writer how to resolve the conflict. It simply alerts the rule writer to its
presence"; resolution is either to make the actions agree or to declare an *override* so one rule suppresses
another ([conflict checker](https://docs.progress.com/bundle/corticon-js-rule-modeling/page/The-conflict-checker.html)).
The completeness checker "employs an algorithm which calculates all mathematical combinations of the
Conditions' values (the Cross Product), and compares them to the combinations defined by the rule writer"; the
missing combinations "are automatically added to the Rulesheet", though "the Action definitions of the new
rules are left to the rule writer"
([completeness checker](https://documentation.progress.com/output/Corticon/5.7.2/html/corticon/the-completeness-checker.html)).
That is a genuine *generative* gap analysis, not a warning list.

**Effective dating and versioning — also the strongest in the field.** A Ruleflow carries an effective date set
in Studio's Ruleflow Properties, and a request may address a Decision Service by either
`decisionServiceTargetVersion` (Major.Minor) or `decisionServiceEffectiveTimestamp`; the two "are mutually
exclusive and optional. If they both contain a null value then the server selects the latest effective version
of the Decision Service to process the request." With no version given, the server "will execute the Decision
Service with *highest* version number"; with only a Major given, the "live Decision Service with highest Minor
version number"; TEST services require a full Major.Minor
([version in a request](https://docs.progress.com/bundle/corticon-deployment/page/How-to-specify-a-version-in-a-SOAP-request-message.html)).
So multiple generations are resident simultaneously and the *caller's business date* selects one. This is the
single most transferable idea in this report.

**Explainability.** The Rule Trace Viewer shows "the sequence of actions that took place in a Ruletest",
sorts and exports to CSV, filters with regexes, and — the part that matters — "Double-click to open the
related Rulesheet at the specific rule applied", closing the loop from trace back to authored artefact
([trace](https://docs.progress.com/bundle/corticon-rule-modeling/page/Trace-rule-execution.html)). It requires
JSON: "if your Studio property uses XML instead, this functionality becomes inoperative."

**Deployment and numbers.** Corticon Server runs in-process in Java or .NET or as a REST/SOAP service
([REST API](https://docs.progress.com/bundle/corticon-server/page/Server-REST-API.html)); Corticon.js compiles
to JavaScript for serverless and client-side execution. The product page lists components including
"Versioning and Governance" and "Explainability and Audit Support", and carries the only concrete customer
throughput figure in this group: Commonwealth of Pennsylvania — "The Corticon platform processes 2.6 million
records in 43 minutes for a sustained throughput of over 1,000 decision sets per second"; plus Brocacef
"slashed time to implement rule changes from 1.5 weeks to 1.5 hours" ([corticon](https://www.progress.com/corticon)).

**vs decider2.** decider2's compiled path is ~1 µs per record (`decider2/docs/README.md:106-110`), four orders
of magnitude clear of Corticon's published 1,000 decision-sets/second, so the performance comparison is not
close and not the point. The two gaps are analysis and effective dating. On analysis, decider2's equivalent is
`assert_equivalent` (`decider2/src/decider2/testing/equivalence.py:218`) plus a boundary-value corpus
(`testing/corpus.py:88`) — both are *differential* checks, not *logical* ones. Nothing in `tables/schema.py` or
`trees/schema.py` computes a cross product over a decision table's condition columns to report uncovered
combinations, and doc 04 §7 positions correctness as "specification-based: rule-level assertions expressing
intent" (`decider2/docs/04-observability-and-governance.md:483-485`) — which is right, and is not a substitute
for "this table has a hole at income ∈ (8000, 8500]". On effective dating, `grep -ni 'effective|valid_from|
valid_to|as_of|decision_date'` over `decider2/src/decider2/` returns only `_effective()`, a params-merge
helper (`runtime/serve.py:188`) — there is no temporal dimension anywhere, while example project 09 makes
`decision_date` a hard contract item: "Every date-sensitive selection resolves against `decision_date`… A flow
that reads the current date anywhere cannot be replayed, and — worse — will replay *successfully* with the
wrong answer" (`decider2/example_projects/09-governance-and-replay-harness.md:1000-1004`). decider2 also has no
"multiple versions resident, selected per request": `ServeHandle` holds one compiled `Driver` and one active
params generation (`runtime/serve.py:167-176`).

---

## 3. InRule Decision Platform

**Authoring.** Three editors: Business Language Editor ("point-and-click natural language interface"),
Decision Table Editor, and Syntax Editor "with hundreds of built-in functions"
([intro](https://docs.inrule.com/docs/introducing-the-inrule-decision-platform)). Tooling is irAuthor
(desktop), Author Studio (web), irVerify and Catalog Manager, with irCatalog Service and irServer Rule
Execution Service as the runtime pair
([architecture](https://docs.inrule.com/docs/inrule-decisioning-architecture)).

**Lifecycle — the best versioning story of the five.** irCatalog is "the organization's single source of
truth". "At design time, authors will retrieve a rule application from the database, check out to make
modifications, and then check back in, and this check-out/check-in process creates an audit record in the
database. The audit records provide a means for ensuring organizational compliance." "Each revision of the
rule application that is checked into the Catalog increments the revision number for that application", the
save date and time are stored, and "optionally, new or existing text labels can be assigned to a revision
(e.g. Production, DEV)". Critically, *the caller selects the revision*: "The InRule API classes that are used
when calling the rule engine specify a revision label", or alternatively "can also specify a revision number".
Promotion between catalogs is a five-step operation in irCatalog Manager (pick revision, target catalog URI,
credentials, optional rename, Promote), and "When a revision is promoted, the changes to the rule application
are deployed seamlessly to the application utilizing it" with no restart, subject to a refresh interval.
DevOpsServices is "a reference implementation and starting point for automation and CI/CD integration around
irCatalog", listening for lifecycle events and triggering regression tests and inter-catalog promotion via
Azure DevOps, GitHub, Jenkins and Slack ([promoting](https://docs.inrule.com/docs/promoting-rule-applications)).

**Testing.** irVerify is the integrated unit- and regression-testing tool, capturing "summary statistics,
state changes, rule engine feedback, tracing information, and performance statistics"
([intro](https://docs.inrule.com/docs/introducing-the-inrule-decision-platform)).

**Deployment and numbers.** SaaS, self-hosted cloud, on-premises, Docker; .NET SDK in-process; irDistribution
compiles rules to JavaScript for client/offline execution; REST rule execution. InRule states "a long-running
rule execution would be anything over 150ms per transaction," with many customers "executing decisions in
single-digit milliseconds" (same page). That is the same order as decider2's 20–100 ms single-record budget
(`decider2/docs/README.md:106-107`) and 10³–10⁵× slower than its compiled kernel.

**vs decider2.** Two ideas transfer. First, **a label, not a document, is what a caller names.** decider2's
serving surface takes the whole params document by value: `POST /params` stages-and-activates a dict
(`decider2/src/decider2/serving/dispatch.py:80-99`, `ROUTES` at `:137-146`), which is precisely the hole doc
04 §2.1 admits — "That a realtime request's raw payload params ever touched a reviewed document at all… the
framework validates it… but has no opinion on where the caller got the values"
(`decider2/docs/04-observability-and-governance.md:95-99`), and which EXPERIMENTS §N3 has already established
is *not* a performance question (`:99-105`). InRule's label-or-revision selector is the concrete answer:
reference-by-id, resolved server-side against a retained set. Second, **check-in/check-out produces the audit
record as a side effect of the workflow**, whereas decider2's audit content is specified (doc 08 §8,
`:804-827`) but no module writes it — `observe/` does not exist (`decider2/docs/00-BUILD.md:217`, Layer 5, "the
top risk").

---

## 4. Oracle Intelligent Advisor (ex-Oracle Policy Automation)

**Authoring.** Rules are written in natural language in Microsoft Word and Excel, "enabling you to easily
configure advice without the need for any programming skills"
([Get Started with Policy Modeling](https://docs.oracle.com/en/cloud/saas/intelligent-advisor/using-policy-modeling/Content/Guides/Get_started_with_Policy_Modeling.htm)).
The Word/Excel choice is made on an explanatory criterion that is directly relevant to decider2's doc 04 §6:
"if the source material is in a text document, you would write the rules in Word. **Word should also be used
if your policy model requires an explanation of how a decision was reached. (Excel explanations just show the
values and the outcome, without detailed reasoning.)**" Excel is for decision tables and for "proving multiple
attributes for the same set of conditions"
([Word or Excel](https://docs.oracle.com/en/cloud/saas/intelligent-advisor/using-policy-modeling/Content/Guides/Use_Intelligent_Advisor/Use_Policy_Modeling/Work_with_rules/Create_rules/Decide_whether_to_write_rules_in_Word_or_Excel.htm)).

**Explainability.** "For every decision that is made by Policy Modeling rules, Intelligent Advisor can provide
detailed reasons showing how that decision was reached, ensuring auditability from the outset. The reasons are
directly generated from the rules and data involved in making the decision" (Get Started, above). The
Determinations API exposes four services — REST Batch Assess and REST Decision Service, SOAP Assess and
Answer — with "Decision reports with customizable level of detail of how each value was determined" and, for
Answer, an "Audit report with full detail of how decision made"
([headless decisioning](https://docs.oracle.com/en/cloud/saas/intelligent-advisor/config-intelligent-advisor/Content/Guides/Customize_extend/DeterminationsAPI/Introduction_to_headless_decisioning.htm)).
Deployment is Oracle Cloud SaaS only; no standards claimed; no latency figures published.

**vs decider2.** This is the product whose *central* claim is the one decider2 doc 04 §6 tested and failed.
Oracle's answer is that the rule source text *is* the explanation — a Word sentence is simultaneously the
executable artefact and the reviewer-facing prose, so they cannot diverge, which is exactly what example
project 09 §5.7 demands: "The generated artefact and the deployed logic must be incapable of disagreeing,
because they are the same thing rendered twice"
(`decider2/example_projects/09-governance-and-replay-harness.md:524-527`). decider2 chose the opposite
direction and is right to: doc 04 §6.6 records that both a Python and a ~1000-line serialised-AST rendering
fail a reviewer (`decider2/docs/04-observability-and-governance.md:446-464`), and doc 04 §6.5 settles that
"Trace is data; every rendering is replaceable" (`:415-444`). What decider2 *can* borrow is Oracle's
distinction, which its own two failed reviewer tests corroborate: Word (prose with derivation) vs Excel
(values and outcome only) is the same split as doc 04 §6.5's "configurable verbosity" (`:433-436`), and
decider2 already has the per-step prose channel to carry it — `Step.doc` and `Step.implements`, the latter
parsed from a docstring `Implements:` line (`decider2/src/decider2/types.py:51-52`,
`decider2/src/decider2/params.py:324-345`), which is the one artefact doc 04 §6.3b says survived both failed
tests: "a rule needs a **join key to the policy clause it implements** — `implements="§7.4.2"` as metadata,
with no `reads=` and no registry" (`:404-410`). That is implemented; nothing renders it.

---

## 5. Decisions (decisions.com)

**Authoring.** Nine rule kinds: Statement ("If/Then statements"), Truth Table ("If/And statements in a table
format. Produces a list of results"), Sequential, Expression ("Uses an algorithm to determine the result"),
Matrix ("Uses a table to allow different inputs to trigger user set outputs other than True or False"), Rule
Sets ("A grouping of Rules or Flows based on a common connection consolidated in a Flow"), Rule Chain ("a
special designer tool to create a path of rules that can branch depending on the rule results. Often used in
questionnaires"), Rule Table ("complex And statements that can return Data like a Matrix Rule"), and Tree Rule
("Similar to Matrix Rules, but with branching paths") ([rules](https://documentation.decisions.com/docs/rules)).

**Lifecycle.** The Deployment Tower is a promotion pipeline built on Deployment Packages: "a key component of
the Deployment Tower, enabling customers to move their projects efficiently and reliably between
environments", labelled `[MajorVersionNumber].[PackageNumber].[BuildNumber]`. Work flows from entities →
User Stories → Builds → Deployment Packages, and packages sit in one of three states: In Progress, Ready, In
Production ([deployment packages](https://documentation.decisions.com/docs/deployment-packages)).

**Testing at the gate.** "The Deployment Tower can run unit tests in the selected target environment during
deployment", either manually or automatically on every deployment (same page). Unit tests are "inputs with
assertions about what happens to them in addition to a Rule that evaluates the step results and the path that
the particular Flow follows", and "are declared either for the entire rule set or for the individual rule"
([unit tests](https://documentation.decisions.com/docs/unit-tests-advanced)). No rollback mechanism is
documented on the Deployment Packages page. Deployment is a Docker image for Enterprise v8 plus REST. African
adopter: GTBank Nigeria — workflow automation including term-loan booking and 15+ credit products, "7,500
users… over 4 million cases in six years" ([case study](https://decisions.com/case-studies/gtbank)).

**vs decider2.** Two contrasts. The **User Story → Build → Package → environment-state chain** is the
artefact decider2 lacks: a named, versioned, promotable unit that carries a state and a test result. decider2's
`StagePlan` is the right shape but has no identity or persistence — it is a frozen dataclass of
`(klass, recompiles, fingerprint, eta, doc)` created and discarded inside one process
(`decider2/src/decider2/runtime/serve.py:60-79`). Second, **running unit tests in the target environment
during deployment** is the CI gate example project 09 §5.6 specifies at much greater rigour (400,000 golden
records, a 40-minute CI budget, coverage thresholds of ≥98% rules and ≥95% tree nodes, and a no-effect rule
that blocks on *any* movement —
`decider2/example_projects/09-governance-and-replay-harness.md:428-490`); decider2 today ships the primitives
(`testing/equivalence.py:218`, `testing/recompile.py:51`, `testing/corpus.py:88`) and no gate that composes
them. On the other side, decider2's `decision_table` interior beats all nine of Decisions' kinds on one
governance property that matters more than kind-count: editing a table's rows is *free*, with no compile at
all, because rows live in `shared` arrays scanned by one generic kernel
(`decider2/src/decider2/tables/__init__.py:12-17`) — doc 08 §3.4's generic-kernel choice
(`decider2/docs/08-configuration-and-lifecycle.md:328-350`) delivered.

---

## 6. The rest, one paragraph each

**Sapiens Decision.** No-code modelling on The Decision Model — "Drag and drop visual modeling workbench
powered by The Decision Model. Quality, clarity, consistency" — over a "powerful glossary framework" of
business-friendly terms; Automated Logic Extraction converts legacy code "across programming languages into
technology-independent decision models without functionality loss"; Decision-as-a-Service "through OpenAPI or
roll your own with our generated POJOs"; "Approval workflows, traceability"; and auto-generated test cases,
"Auto-generate, create, import… No gaps, no conflicts" ([product](https://sapiensdecision.com/product)). No
performance figures; DMN support still appears only in third-party news — **unverified**. The transferable
item is the glossary as a first-class artefact: decider2's nearest equivalent is the project vocabulary map
referenced in doc 03 §5.2 and open question O19 ("Interface freezing and the vocabulary map",
`decider2/docs/06-open-questions-and-experiments.md:136`), plus the checked-in interface contract file at
`decider2/src/decider2/graph/module.py:152-176` — a real, working golden-file check on a module's
inputs/outputs/terminals/params that nothing in the docs advertises as a governance artefact.

**Microsoft RulesEngine (open source).** MIT-licensed .NET library, "A fast and reliable .NET Rules Engine
with extensive Dynamic expression support", rules as JSON (`WorkflowName`, `RuleName`, `RuleExpressionType`,
`Expression`, `SuccessEvent`, `ErrorMessage`) with C#-syntax lambda expressions, custom actions, ScopedParams
and nested rules ([repo](https://github.com/microsoft/RulesEngine)). No published benchmarks, no versioning,
no audit, no simulation, no approval model — it is a library, not a BRMS, and it is *below* decider2 on every
governance axis. Its one relevant lesson is negative: expression strings in config are exactly what decider2
doc 08 §1 forbids ("config is a *composition and parameterisation* language over a registry of code, never a
programming language", `decider2/docs/08-configuration-and-lifecycle.md:29-31`) and what `trees/` explicitly
removed from decider 1 (`ComputedFeatureRemoved`, `decider2/src/decider2/trees/__init__.py:14-20`).

**Azure Logic Apps Rules Engine.** The only first-party hyperscaler BRMS found. "a *decision management
inference engine* that lets you integrate declarative, semantically rich, and easily readable rules with your
Standard logic app workflows"; concepts are Facts ("XML and .NET objects are the native data sources available
today") and Rulesets ("small building blocks of business logic"); "The Rules Engine is based on the Rete
algorithm"; rules are authored in the Microsoft Rules Composer as XML rulesets against .NET Framework
assemblies
([overview](https://learn.microsoft.com/en-us/azure/logic-apps/rules-engine/rules-engine-overview),
[create rules](https://learn.microsoft.com/en-us/azure/logic-apps/rules-engine/create-rules)). It is the
BizTalk BRE lifted into Logic Apps Standard; the .NET-Framework-and-XML-only fact model rules it out for a
Python/polars stack. **AWS has no first-party BRMS**: searching `docs.aws.amazon.com` returns only Marketplace
listings (DecisionRules, OpenRules) and build-your-own blog posts using Flink or Step Functions + Drools
([Flink rules engine](https://aws.amazon.com/blogs/big-data/build-a-dynamic-rules-engine-with-amazon-managed-service-for-apache-flink/)),
with Amazon Fraud Detector closed to new customers as of 2025-11-07 (covered by the fraud agent). Google
Cloud's nearest first-party thing is the reCAPTCHA/Fraud Defense CEL policy engine, already in landscape §4.

**DecisionRules.io.** `POST /rule/solve/{ruleId}/{version}`, and "If the version parameter is omitted, the
last published version will be used automatically"; rules addressable by alias as well as id; four strategies
via `X-Strategy` (STANDARD / FIRST_MATCH / ARRAY / EVALUATE_ALL); and — the item worth copying — audit is a
*per-request header*: `X-Audit: true` "to create and save an audit of the solve", with `X-Audit-Ttl` setting
retention in days (default 14), plus `X-Correlation-Id` generated and echoed if not supplied
([solver API](https://docs.decisionrules.io/doc/api/rule-solver-api)). Per-request audit level and a
correlation id are two small, cheap things decider2's dispatcher does not have.

**Higson.** "Time Versioning - create multiple versions of the business logic and schedule them in a timeline";
versioning that lets users "track changes, revert to previous iterations, and compare differences across
versions"; "Granular Role-Based Permissions" scoped "by business domain, not just by user"; a Tester module for
mass testing with Excel data import that checks "the impact on other rules"; and rules provisioning via
snapshots exporting configuration to text files ([features](https://www.higson.io/features)). The features page
carries no numbers; the separate speed page claims "over 9,000 API calls per second" and "0.23 ms Response
Time" on an i7-13700H single thread ([speed](https://www.higson.io/speed-efficiency)) — the same order as
decider2's compiled path once Python glue is included, and the only vendor in this group whose published
latency is comparable. Scheduling logic versions on a timeline is Corticon's effective dating from the
authoring side.

**Red Hat Decision Manager / Apache KIE.** 7.13 "provides runtime support for DMN 1.1, 1.2, 1.3, and 1.4
models at conformance level 3, and design support for DMN 1.2 models at conformance level 3"
([DMN models](https://docs.redhat.com/en/documentation/red_hat_decision_manager/7.13/html/developing_decision_services_in_red_hat_decision_manager/dmn-con_dmn-models)),
where CL3 "supports Friendly Enough Expression Language (FEEL) expressions, the full set of boxed expressions,
and fully executable decision models". It overlaps ODM only as a lineage (both descend from ILOG/JBoss-era
engines and both ship Business Central/Decision Center-style web authoring); the governance framework is
ODM's, not Red Hat's. Relevant to decider2 only as the reference implementation of an interchange format
decider2 has chosen not to implement.

**SAP.** Signavio Process Manager models DMN DRDs and decision tables, but every `help.sap.com` page for it
returns an SPA shell and `sap.com` returns 403 — version and execution claims remain **unverified**. BRFplus
is "an ABAP-based framework" with "simulation, trace, transport, XML export and import"
([help](https://help.sap.com/doc/saphelp_ewm900/9.0/en-US/9a/6b67ce7c26446483af079719edf679/content.htm?no_cache=true)) —
note "transport", SAP's environment-promotion mechanism, which is the same concern as Decisions' Deployment
Packages. SAP Build Process Automation offers Decision Table and Text rules
([tutorial](https://developers.sap.com/tutorials/spa-create-decision.html)). Not a candidate for a Python bank
stack; listed for completeness.

---

## 7. (A) Feature inventory

| Feature | Which product(s) | What it does | decider2 has it? | Worth adopting? |
|---|---|---|---|---|
| Effective-dated logic selected per request | Corticon (`decisionServiceEffectiveTimestamp`); Higson time versioning; ODM (selection filter over rule expiry) | Caller supplies a business date; server picks the generation that was in force | **no** — no date concept in `src/` at all; `_effective()` at `runtime/serve.py:188` is a params merge | **yes** — 09 §5.15 item 4 makes it a replay precondition (`example_projects/09-...:1000-1004`) |
| Multiple versions resident, addressed by version/label | Corticon (Major.Minor); InRule (revision + label); DecisionRules (`/{version}`, alias) | Several generations live at once; caller names one | **partial** — one active + LIFO history of params dicts (`runtime/serve.py:153`, `:167`); no id, no addressing | **yes** — closes doc 04 §2.1's payload-params hole (`docs/04-...:95-105`) |
| Named, retained ruleset baseline / snapshot | ODM (snapshots, baselines); Higson (snapshots); Decisions (Deployment Packages) | A frozen, re-activatable, retained point | **no** — `_history` is in-process and unnamed | **yes** — 09 §5.7 "Every rendering that was approved… frozen and retained" |
| Branching and merging of rule changes | ADS (git branches, "creating, merging, and protecting"); ODM (branches + streams) | Parallel rule development, merged back | **partial** — skeleton is Python in git, so branching is git's; interiors have no branch model | maybe — free for skeleton; matters only if `Admit.COMPOSITION` ships (doc 08 §7) |
| Approval workflow gating activation | ODM ("only users with the proper responsibilities can approve a release"); Sapiens; Decisions | Four-eyes before a change serves traffic | **no** — activation is explicit (`runtime/serve.py:287`) but unauthenticated and unrecorded; O17 open (`docs/08-...:832`) | **yes** (must-have) — but as a *hook + recorded decision*, not a workflow engine |
| Business-user parameter editing with bounds | InRule; ODM Business console; Higson; all | Non-engineer moves a threshold safely | **yes** — `param()` forwards to pydantic `Field` (`params.py:250`), JSON Schema out (`runtime/serve.py:178`), bounds+`extra="forbid"` enforced (`runtime/serve.py:207-252`) | n/a — already ahead |
| Zero-recompile threshold retune | Higson; DecisionRules; (ODM/Corticon retune = redeploy) | Change a number without rebuilding | **yes** — thresholds are arguments, measured 0 compile events (`docs/08-...:129-155`); table rows free (`tables/__init__.py:15-17`) | n/a — best in class |
| Rule analysis: conflict / redundancy / never-fires | Corticon (conflict checker); ODM (consistency checking: never selected, never applies, equivalent, redundant, conflicting, range violation) | Static logical audit of a rule set | **no** — nothing in `tables/schema.py`, `trees/schema.py` | **yes** — cheap for a validated table document |
| Completeness / gap analysis with generated missing rows | Corticon (cross product, auto-adds missing combinations) | Proves a table covers its input space | **no** | **yes** — highest-value single import in this report |
| Scenario simulation with KPIs | ODM (simulation + KPI reports, Excel scenarios); Higson (mass Excel testing) | Business-readable "what would this change do" | **partial** — `preview()` scores one record two ways (`runtime/serve.py:332`); `decider2.impact()` specified (doc 08 §5) and absent | **yes** (must-have) — doc 08 §5 already specifies a better version (boundary solve, O24) |
| Back-testing on historical data / golden set at the gate | Decisions (unit tests during deployment); InRule (DevOpsServices regression on lifecycle events) | Block a promotion on a regression | **partial** — primitives exist (`testing/equivalence.py:218`, `testing/recompile.py:51`, `testing/corpus.py:88`); no gate | **yes** — 09 §5.6's no-effect rule is the one to build |
| Coverage measurement (rules exercised, nodes reached) | ODM Decision Warehouse ("List of Rules Not Fired", "Total Number of Rules Not Fired") | Finds dead logic | **no** — `<name>_path` gives the arm taken (`trees/__init__.py:41-46`), never the arms not taken | **yes** — 09 §5.8 requires it; cheap to derive from path codes |
| Decision-level audit record (inputs, outputs, rules fired, version) | ODM Decision Warehouse (17 options); ADS execution trace; Oracle audit report; DecisionRules `X-Audit` | The defensible per-decision artefact | **no** — contents fully specified (doc 08 §8 `:804-827`, doc 04 §5.2 `:239-260`); `observe/` unbuilt (`00-BUILD.md:217`) | **yes** (must-have) — the single largest gap |
| Per-request audit verbosity + retention | DecisionRules (`X-Audit`, `X-Audit-Ttl`); ADS (`executionTraceFilters`); ODM monitoring options | Caller pays for the detail it needs | **no** — doc 04 §6.5 item 3 specifies it (`:433-436`) | **yes** (S) |
| Correlation id echoed on the response | DecisionRules (`X-Correlation-Id`) | Joins a decision to a request across systems | **no** | **yes** (S) — 09 §5.15 item 1 needs a stable decision id |
| Provenance token on the config that ran | (none does this as cleanly as decider2 specifies) | Records where the values came from | **no, and it is specified** — `del origin` at `runtime/invoke.py:465`, `:672` vs "refuses to run without one" (`docs/08-...:707-711`) | **yes** (must-have) — the cheapest fix in the report |
| Rule execution report joinable back to the authored rule | Corticon (double-click trace line → Rulesheet at that rule); ODM (Decision Trace → Ruleflow Tasks tree → rule in Decision Center) | Closes trace → source | **partial** — version chains attribute each value to its producer (`docs/04-...:219-237`); nothing renders a link | **yes** (M) |
| Vocabulary / BOM / verbalization layer over the execution model | ODM (BOM/XOM/verbalization); Sapiens (glossary); Corticon (Vocabulary) | Business names decoupled from runtime types | **partial** — `.relabel()`/vocabulary (doc 03 §5.2), interface contract file (`graph/module.py:152`); no verbalization | maybe — doc 00 §6 warns against restating an inferred interface |
| Natural-language rule text that *is* the executable rule | Oracle IA (Word/Excel); InRule Business Language Editor; ODM BAL + Intellirule | Reviewer reads the artefact that runs | **no, deliberately** — two reviewer tests failed (`docs/04-...:277-410`); settled as "trace is data, renderings replaceable" (`:415-444`) | no — but render `Step.doc` + `Step.implements` (`types.py:51-52`), which exist and are unused |
| Policy-clause join key on a rule | (none) | "Which rules implement §7.4" answerable mechanically | **yes** — `Implements:` docstring line parsed into `Step.implements` (`params.py:324-345`, `types.py:52`) | n/a — decider2 is ahead; nothing consumes it yet |
| Hot deploy with no restart | InRule ("deployed seamlessly… does not require application or service restarts"); ODM; ADS | Change reaches a live process | **yes** — measured: swap 0.177 µs, worst in-flight call 1.37% of a 20 ms budget over 30 swaps (`docs/08-...:378-392`) | n/a — best in class, with numbers |
| Rollback | InRule (promote an earlier revision); Corticon (address the old version); Higson (revert) | Undo a bad change fast | **partial** — `rollback()` is free and measured (3.36 µs, zero compile events) but in-process only and lost on restart (`runtime/serve.py:305`; `docs/08-...:471-477`) | **yes** (M) — needs to survive a restart |
| Static lineage / impact without running | (ODM rule analysis is the nearest) | "What can affect output z" | **no, though documented** — `pipeline.lineage()` and `.render()` in doc 04 §3 (`:126-130`) do not exist; only `.schema()` (`graph/pipeline.py:69`) | **yes** (must-have) — doc 08 §5 item 2 depends on it |
| Equivalence of debug and production numerics | (none publishes this) | A trace you can trust | **yes, uniquely** — `interpreted ≡ stepped ≡ fused ≡ score` as one assertion (`testing/equivalence.py:218`, `runtime/modes.py:209`) | n/a — no product in §6 offers this |
| Reason codes | Equifax, ODM (rule ids), all | Machine-readable decline reasons | **yes** — "A reason code is a step output; 'which rule fired' is the branch's `_path`" (`docs/04-...:176-179`) | n/a |
| Compiled/embedded execution alongside REST | Corticon (in-process Java/.NET, Corticon.js); InRule (.NET SDK in-process, irDistribution→JS); ODM (embedded Java) | Same rules in a service and in a library | **yes** — `Pipeline.apply`/`.score` with no server (`graph/pipeline.py:188`, `:209`), serving optional (`runtime/serve.py:1-10`) | n/a |
| Containerised decision service, REST | all | Deployability | **yes** — SageMaker-convention ASGI app, `decider2 serve` (`cli.py:94`, `serving/dispatch.py:137-146`) | n/a |
| Typed schemas and null policy | ODM BOM types; Corticon Vocabulary | Data contract | **yes, stronger** — four declared null tiers incl. not-applicable≠missing (`params.py:278-306`, `boundary/nulls.py`) | n/a |
| Champion/challenger, deterministic assignment | Sparkling Logic, FlexRule, DecisionRules (landscape §4/§6) | A/B a policy safely | **no** | maybe — 09 §5.15 item 3 forbids RNG and requires a recorded seed |

---

## 8. Improvements for decider2 — ranked, de-duplicated

Ranked by cost of *not* having it in a regulated bank, not by effort. "Must-have" means example project 09's
contract or a South African credit regulator's evidence expectation fails without it.

| # | Gap | Demonstrated by | Maps onto | Effort | Verdict |
|---|---|---|---|---|---|
| 1 | **No decision record is emitted.** Contents are fully specified and nothing writes them. | ODM Decision Warehouse (17 monitoring options incl. rules *not* fired); ADS execution trace; Oracle audit report | new `observe/audit.py`; `docs/08-...:804-827`; `docs/04-...:239-260`; `00-BUILD.md:217` (Layer 5, "the top risk") | L | **must-have** |
| 2 | **Provenance is specified then discarded** — `del origin` twice, against a spec that says the framework "refuses to run without one". | InRule check-in audit records; DecisionRules `X-Correlation-Id` | `runtime/invoke.py:465`, `:672`; `docs/08-...:707-711` | S | **must-have** |
| 3 | **No impact/simulation report.** A params change is reviewable only as validator bounds, not as "1.8% of applications change decision". | ODM simulation with KPI reports and Excel scenarios | `decider2.impact()` in `docs/08-...:609-664`; extend `preview()` at `runtime/serve.py:332` | M | **must-have** |
| 4 | **No effective dating / business date.** Nothing in `src/` has a temporal dimension; replay of a dated decision is impossible. | Corticon `decisionServiceEffectiveTimestamp` + Ruleflow effective date; Higson time versioning; ODM rule-expiry selection filters | `tables/schema.py`, `trees/schema.py`, a note in `docs/08-...:104-127`; required by `example_projects/09-...:1000-1004` | M | **must-have** |
| 5 | **No named, retained, addressable generation.** `_history` is an in-process LIFO of dicts; rollback dies at restart. | InRule revisions + labels; Corticon Major.Minor resident versions; Higson snapshots | `runtime/serve.py:129-318`; `compile/cache.py:91-131`; `docs/08-...:471-477` | M | **must-have** |
| 6 | **No approval evidence on activation.** Explicit activation with nothing recorded; O17 open. | ODM ("only users with the proper responsibilities can approve a release and deploy it"); Sapiens approval workflows | `StagePlan` at `runtime/serve.py:60-79`; `activate()` at `:287`; `docs/08-...:832-836` | S | **must-have** |
| 7 | **`pipeline.lineage()` is documented and does not exist.** Blast radius cannot be computed. | ODM rule analysis (nearest equivalent) | `graph/pipeline.py:306-418`; `docs/04-...:126-130` | S | **must-have** |
| 8 | **No completeness / gap analysis on a decision table.** A rate card's uncovered combinations are invisible. | Corticon completeness checker (cross product, auto-generates the missing rows) | `tables/schema.py:1-590` | M | **must-have** |
| 9 | **No conflict / redundancy / never-selected analysis.** A new row can silently shadow an earlier one. | ODM consistency checking (never selected, never applies, equivalent, redundant, conflicting, range violation); Corticon conflict checker | `tables/schema.py`, `trees/schema.py:1-1165` | M | nice-to-have (must-have once a UI edits interiors) |
| 10 | **No CI gate composing the testing primitives.** `assert_equivalent` / `assert_no_recompile` / `corpus` exist; nothing blocks a promotion. | Decisions (unit tests run in the target environment during deployment); InRule DevOpsServices regression-on-promotion | `testing/` beside `equivalence.py:218`; `example_projects/09-...:465-472` (the no-effect rule) | S | **must-have** |
| 11 | **No coverage / dead-logic measurement.** `_path` records the arm taken, never the arms never taken. | ODM "List of Rules Not Fired" / "Total Number of Rules Not Fired" | `trees/codegen.py:1-622`; `example_projects/09-...:556+` | M | nice-to-have |
| 12 | **Audit verbosity is not selectable per request**, though doc 04 §6.5 item 3 specifies it. | DecisionRules `X-Audit` / `X-Audit-Ttl`; ADS `executionTraceFilters`; ODM monitoring options | `serving/dispatch.py:61-71`; `runtime/serve.py:319-331`; `docs/04-...:433-436` | S | nice-to-have |
| 13 | **Nothing renders `Step.doc` / `Step.implements`.** The one artefact that survived both failed reviewer tests is parsed and unused. | Oracle IA (the rule text *is* the explanation); ODM verbalization; InRule Business Language Editor | new `observe/render.py`; `types.py:51-52`; `params.py:324-345`; `docs/04-...:404-410`, `:415-444` | M | nice-to-have (unblocks E4) |
| 14 | **Trace does not link back to the authored rule.** Version chains attribute values; nothing produces a navigable link. | Corticon (double-click a trace line → the Rulesheet at that rule); ODM (Decision Trace → rule in Decision Center) | `observe/` + `docs/04-...:219-237` | M | nice-to-have |
| 15 | **No `decider2.diff(old, new)`**, though "a config diff is an audit record" is claimed. | Higson ("compare differences across versions") | `docs/08-...:703-704`; `docs/04-...:257-260` | S | nice-to-have |
| 16 | **The interface contract file is undocumented as governance.** `contract=` already gates an interface change and no doc mentions it. | ADS protected branches; Sapiens traceability | `graph/module.py:41-46`, `:152-176`; `docs/07-project-structure.md:207` | S | nice-to-have |
| 17 | **No champion/challenger or deterministic assignment.** 09 §5.15 item 3 forbids RNG and requires a recorded seed. | Sparkling Logic, FlexRule, DecisionRules (landscape §4/§6) | new; `example_projects/09-...:970-976` | M | nice-to-have |
| 18 | **No authoring UI for the surfaces doc 08 §3.3 declares editable.** JSON Schema is emitted; no client consumes it. | ODM Business console; Corticon Studio; InRule Author Studio; DecisionRules editor | `runtime/serve.py:178`; `docs/08-...:316-327` | L | nice-to-have (out of framework scope by doc 04 §6.5 item 4) |

**Deliberately not adopted.** Natural-language rules as the executable artefact (Oracle, InRule BLE, ODM BAL) —
doc 04 §6 tested a rendered artefact twice and both failed, and §6.3b settles that "A reviewer cannot
adjudicate a mismatch they are shown" (`docs/04-...:358-360`). Expression strings in config (Microsoft
RulesEngine, Azure/BizTalk) — forbidden by doc 08 §1 on evidence (`docs/08-...:23-31`). A `ConfigManager` /
promotion engine (ODM governance framework, Decisions Deployment Tower) — doc 08 §6 settles this as "not the
framework's business" (`:665-681`), correctly; the recommendations above make the *hook* attachable instead.

---

## 9. (B) Actionable recommendations for decider2

1. **Record `origin` instead of deleting it** in `runtime/invoke.py` — replace `del origin` at `:465` and
   `:672` with a returned/attached provenance field, and make it non-empty as doc 08 §6.2 already specifies,
   so that a decision record can answer "which config document produced these values". Touches
   `decider2/src/decider2/runtime/invoke.py`, `graph/pipeline.py:188-221`. **Effort S.** Motivated by InRule's
   check-in audit record and Oracle's audit report; and by decider2's own spec, which currently lies.
2. **Add an `origin`/`label` and a monotonic `generation_id` to `ServeHandle`** in `runtime/serve.py` so each
   staged document gets a name, and have `GET /health` report the active generation's id and origin alongside
   the fingerprint (`:383-393`), so that a caller can tell which named generation served a request. Touches
   `runtime/serve.py:129-318`, `serving/dispatch.py:125-135`. **Effort S.** Motivated by InRule revision
   labels and Corticon Major.Minor.
3. **Add `POST /params/by-ref {"generation": "<id>"}`** to `serving/dispatch.py:137-146` and have `stage()`
   accept a retained generation id as well as a raw document, so that a production endpoint can be configured
   to accept *only* references and doc 04 §2.1's "raw payload params never touched a reviewed document" hole
   becomes closable by deployment policy. Touches `serving/dispatch.py`, `runtime/serve.py:253-278`.
   **Effort M.** Motivated by InRule (`specify a revision label`) and DecisionRules (`/{version}`, alias).
   EXPERIMENTS §N3 already removed the performance objection (`docs/04-...:99-105`).
4. **Implement `pipeline.lineage(name)` on `Pipeline`** in `graph/pipeline.py` (the step DAG is already walked
   at `:306-394` and `_all_wired_names` at `:409` has the adjacency), so that doc 04 §3's documented API stops
   being fiction and doc 08 §5 item 2 ("which outputs can move") becomes computable. Touches
   `decider2/src/decider2/graph/pipeline.py`, `docs/04-observability-and-governance.md:126-130`. **Effort S.**
   Motivated by ODM rule analysis; required by 09 §5.4.
5. **Write `decider2/observe/audit.py` emitting the doc 08 §8 record as a dict** — skeleton identity, structure
   fingerprint (`runtime/serve.py:90`), compiled artefact id (`compile/cache.py:49` already content-addresses
   the source), params digest, params origin, generation, declared variants, fallback set, inputs, outputs,
   emitted values — so that a decision is defensible at all. Touches new `decider2/src/decider2/observe/`,
   `runtime/invoke.py`, `docs/00-BUILD.md:217`. **Effort L.** Motivated by ODM Decision Warehouse's 17
   monitoring options and ADS's execution trace. This is the must-have.
6. **Add an audit-verbosity parameter and a correlation id to the request surface**: `POST /invocations`
   accepts `{"record": ..., "audit": "off|fired|full", "correlation_id": ...}` and echoes the id, so that a
   realtime path can emit "only fired-rule ids and the values that moved" while a dispute investigation emits
   everything, exactly as doc 04 §6.5 item 3 specifies. Touches `serving/dispatch.py:61-71`,
   `runtime/serve.py:319-331`. **Effort S.** Motivated by DecisionRules `X-Audit`/`X-Audit-Ttl`/
   `X-Correlation-Id` and ADS `executionTraceFilters`.
7. **Add a completeness checker to `tables/`** — `DecisionTable.gaps()` computing the cross product of the
   condition columns' declared domains and returning the uncovered combinations — so that a reviewer is shown
   the holes in a rate card rather than asked to find them. Touches
   `decider2/src/decider2/tables/schema.py:1-590`, `tables/__init__.py`. **Effort M.** Motivated by Corticon's
   completeness checker; this is the highest-value single import in the report because the table document is
   already a validated pydantic object with typed bounds.
8. **Add a conflict/redundancy checker to `trees/` and `tables/`** — report rows or leaves whose conditions
   overlap with a different outcome, plus rows that can never be selected — so that a table edit cannot
   silently shadow an earlier row. Touches `tables/schema.py`, `trees/schema.py:1-1165`. **Effort M.**
   Motivated by ODM consistency checking ("never selected", "redundant", "conflicting") and Corticon's conflict
   checker.
9. **Implement `decider2.impact(active, candidate, sample) -> ImpactReport`** as `observe/blast_radius.py`,
   starting with the sampling half (fraction of records whose declared outputs changed, distribution of each
   change, which paths newly fired) and deferring the exact boundary solve behind O24, so that a params edit is
   reviewable as "1.8% of applications change decision, all in the declining direction". Touches new
   `decider2/src/decider2/observe/blast_radius.py`, `docs/08-configuration-and-lifecycle.md:609-664`,
   `docs/06-open-questions-and-experiments.md:187`. **Effort M.** Motivated by ODM simulation-with-KPIs;
   decider2's specified version is strictly better and does not exist.
10. **Extend `preview()` from one record to a frame** in `runtime/serve.py:332-352` (it already scores the same
    input against two documents through one implementation) so that `POST /params/preview` can take a sample
    and return the impact summary from item 9 rather than a single before/after. Touches `runtime/serve.py`,
    `serving/dispatch.py:100-124`. **Effort S.** Motivated by ODM simulation reports and Higson's mass Excel
    testing.
11. **Add `asserts_no_effect(pipeline_a, pipeline_b, corpus)` to `testing/`** — exact equality on every
    declared output, no tolerance band — so a refactor that moves any golden output fails CI, which example
    project 09 §5.6 calls the rule that "catches more real defects than the rest of the suite combined".
    Touches `decider2/src/decider2/testing/` (new module beside `equivalence.py:218`),
    `example_projects/09-...:465-472`. **Effort S.** Motivated by Decisions' unit tests at the deployment gate
    and InRule DevOpsServices regression-on-promotion.
12. **Emit path codes for arms *not* taken** — have `trees/codegen.py` also record the set of reachable leaf
    ids per tree so a batch run can report which leaves no record reached — so that dead logic is detectable
    without a second instrumentation pass. Touches `decider2/src/decider2/trees/codegen.py:1-622`,
    `trees/__init__.py:41-46`. **Effort M.** Motivated by ODM's "List of Rules Not Fired" / "Total Number of
    Rules Not Fired"; required by 09 §5.8.
13. **Add a business-date input convention and an `active_from`/`active_until` field to the tree and table
    schemas**, evaluated as an ordinary condition against a `decision_date` input rather than a new engine
    concept, so that a rate-card row or a tree leaf can be dated and a replay reproduces the version in force.
    Touches `tables/schema.py`, `trees/schema.py`, `docs/08-configuration-and-lifecycle.md:104-127` (a fourth
    change class or a note on values-with-dates). **Effort M.** Motivated by Corticon's Ruleflow effective date
    and `decisionServiceEffectiveTimestamp`, Higson time versioning, and ODM's rule-expiry selection filter.
14. **Persist a named generation to disk and reload it on start** — write the staged document plus its
    fingerprint and origin to the configured build directory under `compile/cache.py`'s content-addressed
    naming, and have `ServeHandle.__init__` offer `restore(generation_id)` — so that doc 08 §4's "Rollback
    across a *restart* is the caller's problem" stops being a governance hole. Touches
    `runtime/serve.py:144-157`, `compile/cache.py:91-131`, `docs/08-configuration-and-lifecycle.md:475-477`.
    **Effort M.** Motivated by InRule irCatalog (the catalog, not the process, is the source of truth) and
    Higson snapshots.
15. **Add an `approved_by`/`approval_ref` field to `StagePlan` and require it in `live` mode**, recorded
    verbatim and never parsed — the same discipline as `origin` — so that O17 is answered at document
    granularity and activation carries evidence. Touches `runtime/serve.py:60-79`, `:287-304`,
    `docs/08-configuration-and-lifecycle.md:832-836`. **Effort S.** Motivated by ODM ("only users with the
    proper responsibilities can approve a release and deploy it") and Sapiens approval workflows.
16. **Write a default renderer over the graph that emits one screen per step**: `Step.doc`, `Step.implements`,
    the declared inputs with their null tier, the params it reads with pydantic bounds, and the value-version
    chain — as `observe/render.py`, explicitly one replaceable renderer per doc 04 §6.5 item 2. Touches new
    `decider2/src/decider2/observe/render.py`, `types.py:44-58`, `docs/04-...:415-444`. **Effort M.** Motivated
    by Oracle IA (the rule text *is* the explanation) and ODM verbalization. `Step.implements` is already
    parsed (`params.py:324-345`) and nothing reads it — that is the cheapest half.
17. **Promote the interface contract file from an option to a documented governance artefact**: `contract=` at
    `graph/module.py:41-46` already writes and diffs `contracts/{name}.json` over inputs, outputs, terminals
    and params (`:152-176`), which is a working "interface changed" gate that no doc advertises. Document it in
    `docs/04-observability-and-governance.md` and `docs/07-project-structure.md:207` as the module-level
    equivalent of ADS's protected branch. **Effort S.** Motivated by ADS branch protection and Sapiens
    traceability.
18. **Add `decider2.diff(old, new) -> list[Change]`** as specified in doc 08 §6.2 (`:707-712`), over resolved
    params documents first and structure fingerprints second, so that "a config diff is an audit record" (doc
    04 §5.2, `:257-260`) is true in code. Touches new `decider2/src/decider2/observe/diff.py`. **Effort S.**
    Motivated by Higson ("compare differences across versions") and 09 §5.4.

---

## 10. Genuine alternatives to part of decider2

Honest reading: **nothing in §6 is an alternative to decider2's execution tier, and several are alternatives to
its unbuilt governance tier.**

- **Not an alternative: the record tier.** decider2's compiled path is ~1 µs/record with a 20–100 ms
  single-record budget (`decider2/docs/README.md:106-110`), an emitted diagnostic costing +0.11 ns/row
  (`docs/04-...:161-163`), and measured p99 flat at 4–12% of budget from 1 to 16 concurrent threads with
  `nogil=True` (`docs/08-...:414-430`). The best first-party figures in this group are Corticon's customer
  "over 1,000 decision sets per second" ([corticon](https://www.progress.com/corticon)), InRule's "single-digit
  milliseconds" ([intro](https://docs.inrule.com/docs/introducing-the-inrule-decision-platform)) and Higson's
  "0.23 ms" single-thread ([speed](https://www.higson.io/speed-efficiency)). Only Higson is within two orders
  of magnitude, and none of them runs inside a polars batch over 22 M decisions/year
  (`example_projects/09-...:83-107`).
- **Not an alternative: the equivalence ladder.** No product in §6 offers "the debugger runs the production
  code and their agreement is an automated test" (`testing/equivalence.py:218`, `runtime/modes.py:209`).
  Corticon's Ruletest, ODM's test suites and irVerify all test *a* build; none asserts that the traced
  semantics equal the deployed semantics.
- **Genuinely an alternative: the approval and promotion workflow.** decider2 has decided not to own this —
  "Banks have change-management systems, and four-eyes approval, RBAC, environment promotion and change tickets
  appear nowhere in those 553 lines anyway" (`docs/08-...:679-681`) — and ODM's decision governance framework,
  InRule's irCatalog + DevOpsServices and Decisions' Deployment Tower are all mature implementations of exactly
  that. The right conclusion is not to buy one but to confirm that decider2's activation hook is genuinely
  attachable: today `activate()` (`runtime/serve.py:287`) carries no approver, no plan identity and no
  persistence, so there is nothing for an external workflow to bind to. Items 2, 14 and 15 above are what
  close that.
- **Genuinely an alternative: a reviewer-facing authoring surface for a *self-contained* rate card or
  eligibility table.** Where a business team owns a grid and nothing else, ODM's Business console, Corticon
  Studio, InRule's Author Studio and DecisionRules' editor are all better today than anything decider2 has,
  because decider2 has no authoring UI and doc 08 §3.3's editable-surfaces table (`:316-327`) is a JSON Schema
  contract (`runtime/serve.py:178`) with no client. decider2's counter-argument is real and measured — row
  edits are free, no compile at all (`tables/__init__.py:15-17`) — but a schema is not a screen.
- **Genuinely an alternative: rule analysis as a service.** Corticon's completeness and conflict checkers and
  ODM's consistency checking are years of work decider2 has none of. Items 7 and 8 replicate the tractable
  part for validated table/tree documents; the full Rete-era analysis over arbitrary rule interaction is not
  worth rebuilding and is not needed, because decider2 steps are pure functions in a declared DAG rather than
  rules competing in working memory.
- **Not an alternative, and a warning: Microsoft RulesEngine and Azure Logic Apps Rules Engine.** The first is
  a library with no governance; the second is BizTalk's Rete engine over "XML and .NET objects… the native data
  sources available today"
  ([overview](https://learn.microsoft.com/en-us/azure/logic-apps/rules-engine/rules-engine-overview)). Both
  make config a programming language, which doc 08 §1 rejects on evidence
  (`docs/08-...:23-31`, and `trees/__init__.py:14-20` on what `simpleeval` expression strings cost decider 1).
- **AWS and GCP are not alternatives at all** for general credit decisioning: no first-party BRMS exists on
  either, and Amazon Fraud Detector closed to new customers on 2025-11-07.

---

## 11. Unverified / unreachable

| Item | Status |
|---|---|
| `ibm.com/docs/en/odm/9.5.0` and `.../odm/9.5.0?topic=...` (topic slugs) | JS shell / 404. **Worked around**: static `SSQP76_8.10.x/com.ibm.odm.*` paths, host `odmoc`, host `dbaoc`, and versioned `?topic=` slugs for 8.9.x–8.11.x all serve real content. Facts above are from 8.9.1–8.11.1 and ODM-on-Cloud; **9.5 parity not verified.** |
| `SSQP76_8.10.x/com.ibm.odm.dcenter.prepare/topics/con_cmg_dgf_intro.html` | 404 for 8.10.x. Governance-framework facts taken from host `dbaoc` instead. |
| ODM Decision Center governance **role names and per-role permissions** | **unverified** — both fetched pages reference "governance roles" without enumerating them. |
| ODM/ADS **baselines vs snapshots** as distinct concepts | **partial** — "create snapshots" verified on the consoles page; a retained *baseline* appears only as the ruleset property `ilog.rules.teamserver.baseline`. |
| ODM and ADS **latency/throughput** | **none published** on any first-party page fetched. IBM's ODM 8.9.x Tuning Guide PDF was found but not fetched. |
| IBM Redpaper `redp5333.pdf` (ODM) | found, **not fetched** — could firm up governance and sizing. |
| Oracle **Intelligent Advisor versioning, deployment activation and migration** | **unverified** — the index page carries no detail and the guide pages were not reached. |
| Oracle **Banking Origination** product page | 403 — **unverified** (unchanged from landscape §10). |
| Corticon **Server REST API** verbatim contents | cited from landscape §6's fetch; **not re-fetched** this round. Effective-dating facts are from the deployment bundle, verified. |
| Corticon **completeness-checker limitations** | referenced as a separate page, **not fetched**. |
| InRule **irCatalog audit record fields** | **partial** — "creates an audit record in the database… for ensuring organizational compliance" verified; the fields are not published on the pages fetched. |
| Decisions **rollback** | **not documented** on the Deployment Packages page; absence, not denial. |
| Sapiens Decision **DMN support** and any performance figure | **unverified** (unchanged). |
| SAP **Signavio** DMN version and execution claims | **unverified** — `help.sap.com` SPA shells, `sap.com` 403 (unchanged). |
| Higson **audit trail** specifics | **unverified** — the features page implies it via versioning and private-session change registers; `docs.higson.io/4.1` not fetched. |
| Microsoft RulesEngine **maintenance status** and benchmarks | **unverified** — 201 commits with recent activity, no published status statement or benchmark. |
| Azure Logic Apps Rules Engine **ruleset versioning and audit** | **unverified** — the overview claims "a centralized, auditable repository" without specifying the mechanism. |
