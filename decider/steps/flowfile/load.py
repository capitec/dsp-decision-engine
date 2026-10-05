from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import pkgutil
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from pydantic import ValidationError

from decider.exceptions import WiringError, wrap_import_errors
from decider.registry.resolve import hint
from decider.steps.base import Step, as_step
from decider.steps.branch import _column_condition, branch
from decider.steps.configurable import ConfigurableStep
from decider.steps.loop import loop
from decider.steps.sequential import flow

from decider.steps.flowfile.schema import NODE_MODELS, BranchNode, ExprCondition, FlowNode, LoopNode, RootDoc, StepNode

_EXTS = ("yaml", "yml", "json")


@dataclasses.dataclass
class _Context:
    project_dir: Path
    root_components: str | None
    components_index: dict[str, Any]
    documents: Mapping[str, Any]
    documents_index: set
    cache: dict
    file_cache: dict
    chain: tuple = ()

    def push(self, name: str) -> "_Context":
        return dataclasses.replace(self, chain=(*self.chain, name))


def load_flow(path: str | Path, *, components: str | ModuleType | None = None,
              documents: Mapping[str, Any] | None = None) -> Step:
    """Load a flow file into the same `Step` tree `flow`/`branch`/`loop` would build by hand.

    `components` names (or is) the module or package whose public functions
    and `Step` instances a `step:` node can reference by name; it defaults to
    the file's own `components:` key. `documents` maps a name to a parsed
    tree, table or scorecard document that `step:` can reference instead.

    Example::

        # flow.yaml: {flow: credit, steps: [{step: approved}]}
        load_flow("flow.yaml", components="credit_risk.steps")
    """
    path = Path(path)
    root = _validate(RootDoc, read_flow(path), str(path))
    project_dir = path.parent.parent if path.parent.name == "flows" else path.parent
    documents = documents or {}
    components = components if components is not None else root.components
    if isinstance(components, str):
        components = importlib.import_module(components)
    ctx = _Context(
        project_dir=project_dir,
        root_components=components and components.__name__,
        components_index=_index(components),
        documents=documents,
        documents_index={n for n, d in documents.items() if n != "params" and isinstance(d, Mapping) and "type" in d},
        cache={},
        file_cache={},
        chain=(root.flow,),
    )
    built = _build_list(root.steps, f"{path}: steps", ctx)
    return flow(*built, name=root.flow, id=root.id)


def _validate(model: type, raw: Any, loc: str) -> Any:
    try:
        return model.model_validate(raw)
    except ValidationError as e:
        raise WiringError(f"{loc}: {e}") from e


def read_flow(path: Path) -> Any:
    """Read a flow file (YAML or JSON) into its raw dict, without building steps.

    Example::

        read_flow(Path("flows/credit.yaml"))["flow"]  # "credit"
    """
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    with wrap_import_errors("yaml"):
        import yaml
    return yaml.safe_load(text)


def _build_list(nodes: Sequence[dict], loc: str, ctx: _Context) -> list:
    return [_build_node(n, f"{loc}[{i}]", ctx) for i, n in enumerate(nodes)]


def _as_flow_or_single(nodes: Sequence[dict], loc: str, ctx: _Context) -> Step:
    built = _build_list(nodes, loc, ctx)
    return built[0] if len(built) == 1 else flow(*built)


def _build_node(raw: Any, loc: str, ctx: _Context) -> Step:
    if not isinstance(raw, Mapping):
        raise WiringError(f"{loc}: a node must be a mapping, got {raw!r}")
    present = [k for k in NODE_MODELS if k in raw]
    if len(present) != 1:
        raise WiringError(f"{loc}: a node needs exactly one of step/flow/branch/loop, got {present or 'none'}")
    kind = present[0]
    node = _validate(NODE_MODELS[kind], raw, loc)
    if isinstance(node, StepNode):
        return _build_step(node, loc, ctx)
    if isinstance(node, FlowNode):
        return _build_flow(node, loc, ctx)
    if isinstance(node, BranchNode):
        return _build_branch(node, loc, ctx)
    return _build_loop(node, loc, ctx)


def _resolve_name(name: str, loc: str, ctx: _Context) -> Step:
    if name in ctx.cache:
        return ctx.cache[name]
    in_components, in_documents = name in ctx.components_index, name in ctx.documents_index
    if in_components and in_documents:
        raise WiringError(f"{loc}: step {name!r} is defined in both components and documents")
    if in_components:
        built = as_step(ctx.components_index[name])
    elif in_documents:
        built = ConfigurableStep.load(ctx.documents[name])
    else:
        names = [*ctx.components_index, *ctx.documents_index]
        raise WiringError(f"{loc}: unknown step {name!r}.{hint(name, names)}")
    ctx.cache[name] = built
    return built


def _build_step(node: StepNode, loc: str, ctx: _Context) -> Step:
    built = _resolve_name(node.step, loc, ctx)
    return built.named(node.as_) if node.as_ else built


def _flow_files(project_dir: Path, name: str) -> list[Path]:
    return [p for ext in _EXTS if (p := project_dir / "flows" / f"{name}.{ext}").exists()]


def _build_flow(node: FlowNode, loc: str, ctx: _Context) -> Step:
    matches = _flow_files(ctx.project_dir, node.flow)
    if node.steps is not None:
        if matches:
            raise WiringError(f"{loc}: flow {node.flow!r} has inline steps, but {matches[0]} also exists")
        built = _build_list(node.steps, loc, ctx)
        return flow(*built, name=node.flow, id=node.id)
    return _build_flow_reference(node.flow, matches, loc, ctx)


def _build_flow_reference(name: str, matches: list, loc: str, ctx: _Context) -> Step:
    if not matches:
        raise WiringError(f"{loc}: flow {name!r} has no inline steps and no flows/{name}.yaml|.yml|.json")
    if len(matches) > 1:
        raise WiringError(f"{loc}: flow {name!r} matches more than one file: {matches}")
    if name in ctx.chain:
        raise WiringError(f"{loc}: flow reference cycle: {' -> '.join((*ctx.chain, name))}")
    file_path = matches[0]
    if file_path not in ctx.file_cache:
        ctx.file_cache[file_path] = read_flow(file_path)
    sub = _validate(RootDoc, ctx.file_cache[file_path], str(file_path))
    if sub.flow != name:
        raise WiringError(f"{file_path}: declares flow {sub.flow!r}, expected {name!r}")
    if sub.components is not None and sub.components != ctx.root_components:
        raise WiringError(
            f"{file_path}: components {sub.components!r} does not match the root's {ctx.root_components!r}"
        )
    built = _build_list(sub.steps, f"{file_path}: steps", ctx.push(name))
    return flow(*built, name=name, id=sub.id)


def _resolve_condition(raw: Any, node_name: str, suffix: str, loc: str, ctx: _Context) -> Any:
    if isinstance(raw, ExprCondition):
        from decider.steps.trees import expr_condition

        return expr_condition(raw.expr, name=f"{node_name}_{suffix}", feature_types=raw.feature_types)
    if raw in ctx.components_index:
        return _resolve_name(raw, loc, ctx)
    return raw


def _build_branch(node: BranchNode, loc: str, ctx: _Context) -> Step:
    condition = _resolve_condition(node.when, node.branch, "when", loc, ctx)
    arms = [_as_flow_or_single(node.then, f"{loc}.then", ctx)]
    if node.else_ is not None:
        arms.append(_as_flow_or_single(node.else_, f"{loc}.else", ctx))
    return branch(condition, *arms, modifies=node.modifies, name=node.branch, id=node.id)


def _build_loop(node: LoopNode, loc: str, ctx: _Context) -> Step:
    condition = _resolve_condition(node.while_, node.loop, "while", loc, ctx)
    if isinstance(condition, str):
        condition = _column_condition(condition)
    body = _as_flow_or_single(node.steps, f"{loc}.steps", ctx)
    return loop(condition, body, carries=node.carries, max_iterations=node.max, name=node.loop, id=node.id)


def components(module: ModuleType | None) -> dict[str, Any]:
    """The steps and step functions a components module (and its submodules) defines, by name.

    Example::

        import my_project.components
        components(my_project.components)["full_limit"]
    """
    if module is None:
        return {}
    modules = [module]
    if hasattr(module, "__path__"):
        modules += [importlib.import_module(info.name)
                    for info in pkgutil.walk_packages(module.__path__, module.__name__ + ".")]
    found: dict[str, tuple] = {}
    for mod in modules:
        for key, obj in vars(mod).items():
            if key.startswith("_"):
                continue
            if isinstance(obj, Step):
                name = obj.name
            elif inspect.isfunction(obj) and obj.__module__ == mod.__name__:
                name = obj.__name__
            else:
                continue
            if name is None:
                continue
            if name in found and found[name][0] is not obj:
                raise WiringError(f"component {name!r} is defined in both {found[name][1]} and {mod.__name__}")
            found[name] = (obj, mod.__name__)
    return {name: obj for name, (obj, _) in found.items()}


_index = components  # load_flow's `components` parameter shadows the function
