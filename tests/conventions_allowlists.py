"""Today's violations of each convention rule; every entry is a to-do for a later phase."""

NO_ALL: dict[str, str] = {}  # {module path relative to decider/: reason}

BIG_ALL: dict[str, str] = {  # {package path relative to decider/: reason}
    ".": "the authoring vocabulary is one namespace by design",
    "steps/trees/schema": "permanent: the document grammar is public by design",
}

PRIVATE_IMPORT: dict[str, str] = {  # {"importer -> imported": reason}
    "_engine.debug.edit -> steps._base": "as_step is internal; a session edit converts a function to a step",
    "_engine.ir.context -> steps._base": "as_step is internal; step_map converts a function to a step",
    "_tooling.bridge.building -> serving._parse": "request parsing helpers have no public home",
    "_tooling.bridge.datasets -> lab.data._formats": "format_of has no public home",
    "_tooling.bridge.datasets -> lab.data._providers": "dataset_documents and datasets_folder are bridge helpers with no public home",
    "_tooling.bridge.loading -> lab.data._providers": "project_root is a bridge helper with no public home",
    "_tooling.bridge.rows -> lab.data._keys": "suggest_key has no public home",
    "_engine.compile.njit -> steps._helpers": "allows_fallback and helper_signatures have no public home",
    "lab.diff._worker -> steps._entry": "import_flow has no public home",
    "lab.experiments._experiments -> steps._entry": "import_flow has no public home",
    "lab.experiments._runner -> steps._entry": "import_flow has no public home",
    "lab.experiments._worker -> steps._entry": "import_flow has no public home",
    "lab.diff._hash -> _engine.compile.fingerprint": "fingerprint has no public home",
    "lab.diff._smt -> steps.trees._encode": "feature_kinds and kind_of have no public home",
    "lab.diff._smt -> steps.tables._encode": "variables, dtypes, kind and typed have no public home",
    "_tooling.cli -> serving._handler": "construct_handler_from_settings has no public home",
    "_tooling.cli -> serving._servers.sanic": "the sanic app factory has no public home",
    "_tooling.bridge.experiment -> lab.experiments._experiments": "InputSpec is not public; the bridge overrides a definition's input",
    "_tooling.bridge.experiment -> lab.experiments._runner._run": "runs with the bridge's own job handle over a live step",
    "_tooling.bridge.experiment -> lab.experiments._runner._load": "loads a definition's flow and input for that run",
    "_tooling.bridge.experiment -> steps._entry": "import_flow has no public home",
    "_tooling.mcp._tools -> steps._entry": "import_flow has no public home",
    "lab.experiments._cost -> _engine.wiring.resolve": "the cost estimate counts the resolved plan's calls, which is internal",
    "lab.checks._base -> _engine.wiring.resolve": "ctx.plan is the resolved plan, which is internal",
    "lab.checks._builtin.structure -> _engine.wiring.plan": "the version-use check walks the resolved plan, which is internal",
    "_tooling.cli._check -> steps._entry": "import_flow has no public home",
    "_tooling.cli -> _engine.compile": "cpu_target has no public home",
    "_tooling.bridge.protocol -> _engine.run.engine": "capabilities lists the runner names, which have no public home",
    "lab._describe -> _engine.wiring.Plan": "describe reads data edges off the resolved plan, which is internal",
    "lab._describe -> _engine.wiring.resolve": "describe reads data edges off the resolved plan, which is internal",
    "_tooling.bridge.bridge -> _engine.debug": "wire constants are internal to the bridge protocol",
    "_tooling.bridge.bridge -> _engine.debug.edit": "hot-edit swap is internal to the bridge",
    "_tooling.bridge.bridge -> _engine.params": "param merge has no public home",
    "_tooling.bridge.building -> _engine.params": "param merge has no public home",
    "_tooling.bridge.lineage -> _engine.wiring.plan": "lineage walks the resolved plan, which is internal",
    "_tooling.bridge.forks -> _engine.params": "param merge and document keys have no public home",
    "_tooling.bridge.run -> _engine.run.runners.interpreted": "the bridge drives the interpreted runner directly",
    "lab.experiments._runner -> _engine.params": "param merge has no public home",
    "lab.experiments._runner -> _engine.run.state": "batch state and dtype mapping are internal",
    "lab.experiments._runner -> _engine.wiring.plan": "walks the resolved plan, which is internal",
    "_tooling.cli.ids -> steps._composites": "the constructor names are shared with flow files and have no public home",
    "lab.testing._corpus -> _engine.wiring.resolve": "resolve is internal",
    "lab.experiments._runner -> _engine.types": "cast_values, the value-coercion helper, has no public home",
    "steps.expr._postfix -> _engine.boundary._arrow.intrinsics": "permanent: steps may use the Arrow intrinsics",
    "steps.tables._matcher -> _engine.boundary._arrow.intrinsics": "permanent: steps may use the Arrow intrinsics",
    "steps.trees._walker -> _engine.boundary._arrow.intrinsics": "permanent: steps may use the Arrow intrinsics",
}

UPWARD_IMPORT: dict[str, str] = {  # {"importer module -> imported module": reason}
    "__init__ -> lab": "permanent: describe is the one deliberate second path, for users",
    "_settings -> config": "the settings build a config store from its type tag",
}

LAYER_CYCLE: dict[str, str] = {  # {"package -> package": reason}
    "_engine -> steps": "engine reaches steps.base, helpers and ParamRef lazily; removed when that goes through the IR",
    "steps -> _engine": "steps bind through the engine; goes with the engine -> steps edge",
}

SHARED_STATE: dict[str, str] = {  # {"path::Class.method": reason}
    "_engine/run/engine.py::Executable._params": "writes a threading.local, so the report is per thread",
}

KERNEL_DECORATOR: dict[str, str] = {  # {path relative to decider/: reason}
    "_engine/boundary/_arrow/intrinsics.py": "permanent: the Arrow intrinsics stay with the boundary",
    "_engine/compile/kernel.py": "the generated kernel body is an intrinsic; to be confirmed as an internal kernel",
    "steps/scorecard/_kernels.py": "permanent: internal kernels may use register_jitable",
    "steps/tables/_matcher.py": "internal kernel registers inlining overloads; rewrite as register_jitable",
    "steps/trees/_walker.py": "internal kernel registers inlining overloads and an intrinsic; rewrite as register_jitable",
}
