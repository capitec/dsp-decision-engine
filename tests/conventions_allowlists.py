"""Today's violations of each convention rule; every entry is a to-do for a later phase."""

NO_ALL: dict[str, str] = {}  # {module path relative to decider/: reason}

BIG_ALL: dict[str, str] = {  # {package path relative to decider/: reason}
    ".": "the authoring vocabulary is one namespace by design",
    "contract": "the integration contract exports every versioned model an integrator receives",
    "steps/trees/schema": "permanent: the document grammar is public by design",
}

PRIVATE_IMPORT: dict[str, str] = {  # {"importer -> imported": reason}
    "_engine.debug.edit -> steps._base": "as_step is internal; a session edit converts a function to a step",
    "_engine.ir.context -> steps._base": "as_step is internal; step_map converts a function to a step",
    "_engine.ir.context -> steps._values": "TableRef is internal; ctx.table tells a reference from inline rows",
    "_bridge.building -> serving._parse": "request parsing helpers have no public home",
    "_bridge.datasets -> data._formats": "format_of has no public home",
    "_bridge.datasets -> data._providers": "dataset_documents and datasets_folder are bridge helpers with no public home",
    "_bridge.loading -> data._providers": "project_root is a bridge helper with no public home",
    "_bridge.rows -> data._keys": "suggest_key has no public home",
    "_engine.compile.njit -> steps._helpers": "allows_fallback and helper_signatures have no public home",
    "diff._hash -> _engine.compile.fingerprint": "fingerprint has no public home",
    "experiments._runner -> diff._revisions": "revision checkout helpers have no public home",
    "experiments.metrics._compare -> diff._outputs": "the changed-value primitive is shared with Divergence and has no public home",
    "cli -> serving._handler": "construct_handler_from_settings has no public home",
    "cli -> serving._servers.sanic": "the sanic app factory has no public home",
    "_bridge.experiment -> experiments._experiments": "InputSpec is not public; the bridge overrides a definition's input",
    "_bridge.experiment -> experiments._runner._run": "runs with the bridge's own job handle over a live step",
    "_bridge.experiment -> experiments._runner._load": "loads a definition's flow and input for that run",
    "_bridge.experiment -> experiments._entry": "import_flow has no public home",
    "mcp._tools -> experiments._entry": "import_flow has no public home",
    "experiments._runner -> contract._manifest": "new_manifest_id is minting, not part of the contract surface",
    "_bridge.bridge -> contract._messages": "the reply models are the bridge protocol, kept out of the 25-name contract surface",
    "checks._common -> _engine.ir.nodes": "FiniteStateMachineNode and ScatterGatherNode are not in decider.ir",
    "cli -> _engine.compile": "cpu_target has no public home",
    "data._preflight -> _engine.run.params": "param bundles and namespace checks; goes with data/preflight",
    "data._preflight -> _engine.wiring": "plans and resolve are internal; goes with data/preflight",
    "data._preflight -> _engine.wiring.resolve": "plans and resolve are internal; goes with data/preflight",
    "_bridge.bridge -> _engine.debug": "wire constants are internal to the bridge protocol",
    "_bridge.bridge -> _engine.debug.edit": "hot-edit swap is internal to the bridge",
    "_bridge.bridge -> _engine.params": "param merge has no public home",
    "_bridge.building -> _engine.params": "param merge has no public home",
    "_bridge.lineage -> _engine.wiring.plan": "lineage walks the resolved plan, which is internal",
    "_bridge.forks -> _engine.params": "param merge and document keys have no public home",
    "_bridge.run -> _engine.run.runners.interpreted": "the bridge drives the interpreted runner directly",
    "experiments._runner -> _engine.ir.decls": "base_annotation is not in decider.ir",
    "experiments._runner -> _engine.params": "param merge has no public home",
    "experiments._runner -> _engine.run.state": "batch state and dtype mapping are internal",
    "experiments._runner -> _engine.wiring.plan": "walks the resolved plan, which is internal",
    "_tools.ids -> steps._composites": "the constructor names are shared with flow files and have no public home",
    "_tools.ids -> _engine.ir.origin": "check_id and check_name are not in decider.ir",
    "serving._handler -> _engine.types": "struct_item has no public home",
    "serving._parse -> _engine.types": "struct_item has no public home",
    "serving._parse -> _engine.ir.decls": "base_annotation is not in decider.ir",
    "testing._corpus -> _engine.ir.decls": "base_annotation is not in decider.ir",
    "testing._corpus -> _engine.wiring.resolve": "resolve is internal",
    "experiments._runner -> _engine.debug.session._cast": "wants a public value-coercion helper",
    "steps.expr._postfix -> _engine.boundary._arrow.intrinsics": "permanent: steps may use the Arrow intrinsics",
    "steps.tables._matcher -> _engine.boundary._arrow.intrinsics": "permanent: steps may use the Arrow intrinsics",
    "steps.trees._walker -> _engine.boundary._arrow.intrinsics": "permanent: steps may use the Arrow intrinsics",
}

UPWARD_IMPORT: dict[str, str] = {  # {"importer module -> imported module": reason}
    "__init__ -> contract": "permanent: describe is the one deliberate second path, for users",
    "_settings -> config": "the settings build a config store from its type tag",
}

LAYER_CYCLE: dict[str, str] = {  # {"package -> package": reason}
    "_engine -> steps": "engine reaches steps.base, helpers and values lazily; removed when that goes through the IR",
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
