from __future__ import annotations

import weakref
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from decider.engine.ir.decls import ParamDecl
from decider.engine.ir.nodes import BranchNode, CallNode, IRNode, LoopNode, SequenceNode, iter_nodes
from decider.engine.ir.origin import Origin, check_name
from decider.engine.params.models import record_shared_type
from decider.registry import import_path

if TYPE_CHECKING:
    from decider.steps.base import Step

# Built IR and the path -> Step entries of its members, per (step object,
# parent path). Keyed by id() because pydantic steps compare by value; the
# finalizer drops the entry before the id can be reused.
_BUILT: dict[int, dict[str, tuple[IRNode, dict[str, Step]]]] = {}


def _join(parent: str, name: str) -> str:
    check_name(name)
    return f"{parent}/{name}" if parent else name


@dataclass(frozen=True, slots=True)
class IRContext:
    """What a step's `to_ir` knows about where it is placed: the path of its parent.

    Example::

        class Scaled(ConfigurableStep):
            factor: Value[float]

            def to_ir(self, ctx):
                factor = ctx.value(self.factor, float)
                ...
                return CallNode(ctx.origin(self), "row", ...)
    """

    path: str = ""
    # Every step built under this context, by the path of its node.
    steps: dict[str, Step] = field(default_factory=dict, compare=False, repr=False)

    def child(self, name: str | None) -> IRContext:
        """The context for members of a step called `name`; `None` (anonymous) is transparent."""
        return self if name is None else IRContext(_join(self.path, name), self.steps)

    def origin(self, step: Step, name: str | None = None, locator: str | None = None) -> Origin:
        """The origin of a node `step` produces here, named `name` (default: the step's name)."""
        source = type(step) if isinstance(step, BaseModel) else getattr(step, "fn", type(step))
        own = step.name if name is None else name
        return Origin(self.path if own is None else _join(self.path, own), import_path(source), locator)

    def build(self, step: Step) -> IRNode:
        """`step`'s IR placed under this context, with its relabels applied.

        Composite steps call this for each member. The result is cached per
        step object and path, so an unchanged step returns the same IR.
        """
        key = id(step)
        built = _BUILT.get(key)
        if built is None:
            built = _BUILT[key] = {}
            weakref.finalize(step, _BUILT.pop, key, None)
        entry = built.get(self.path)
        if entry is None:
            # A fresh map per build, so a cached subtree brings its own entries along.
            scope = IRContext(self.path)
            node = step.to_ir(scope)
            reads, writes = dict(getattr(step, "reads", ())), dict(getattr(step, "writes", ()))
            if reads or writes:
                node = _relabel(node, reads, writes, set())
            entry = built[self.path] = (node, scope.steps)
        node, members = entry
        self.steps.update(members)
        # Kept out of the cache entry, which would otherwise keep the step alive.
        # Set last, so a step owns its path over an anonymous member or helper sharing it.
        self.steps[self.child(step.name).path] = step
        return node

    def expand(self, owner: Step, helper: Step) -> IRNode:
        """Build `helper` as the body of `owner`: its nodes sit under `owner`'s name, its root takes `owner`'s origin.

        For a `ConfigurableStep` whose `to_ir` assembles other steps.

        Example::

            class Affordability(ConfigurableStep):
                def to_ir(self, ctx):
                    return ctx.expand(self, flow(disposable_income, ratio))
        """
        node = self.child(owner.name).build(helper)
        return replace(node, origin=self.origin(owner))

    def value(self, value: Any, annotation: Any = None, arg: str | None = None) -> Any:
        """A `Value[T]`: the literal itself, or a `ParamDecl` for a `ParamRef`.

        A `ParamRef` without a default is a required param. `annotation`
        defaults to the type of the ref's default. A literal goes in the
        node's `consts` as `(arg, literal)`, a `ParamDecl` in its `params`.

        Args:
            arg: the function argument a scalar call feeds, whatever the
                ref's param is called in the params document.

        Example::

            ctx.value(0.7)                                            # 0.7
            ctx.value(ParamRef(param="hi_thresh", default=0.7), float, arg="threshold")
            # ParamDecl("hi_thresh", float, 0.7, arg="threshold")
        """
        from decider.steps.values import ParamRef

        if not isinstance(value, ParamRef):
            return value
        return ParamDecl(
            value.param, annotation or type(value.default), value.default, required=value.default is None,
            shared_key=value.param if value.shared else None, arg=arg,
        )

    def table(self, value: Any, schema: dict[str, str]) -> Any:
        """A `TableValue`: inline rows as given, or a required table `ParamDecl` for a `TableRef`.

        `schema` maps each column to its dtype; the param carries it so rows
        are checked when they arrive.

        Example::

            ctx.table(TableRef(table="prices"), {"product": "str", "rate": "float"})
        """
        from decider.serializable.dataframe import DataFrame
        from decider.steps.values import TableRef

        if not isinstance(value, TableRef):
            return value
        return ParamDecl(
            value.table, DataFrame, required=True, shared_key=value.table if value.shared else None,
            schema=tuple(schema.items()),
        )


def to_ir(step: Any) -> IRNode:
    """Build the IR of a step (or plain function) and check it.

    Checks that every node has an origin, paths are unique, no step is named
    `shared`, and shared params agree on their type. Unchanged steps return
    the IR built before.

    Example::

        from decider import engine
        ir = engine.to_ir(pipeline)
    """
    return _build(step)[0]


def step_map(step: Any) -> dict[str, Step]:
    """The step object behind each node path of `to_ir(step)`, conditions, arms and loop bodies included.

    Example::

        engine.step_map(pipeline)["term/by_sector/cap_private"]  # the FunctionStep
    """
    return _build(step)[1]


def _build(step: Any) -> tuple[IRNode, dict[str, Step]]:
    from decider.steps.base import as_step

    ctx = IRContext()
    root = ctx.build(as_step(step))
    _check(root)
    return root, ctx.steps


def _check(root: IRNode) -> None:
    seen: dict[str, Origin] = {}
    shared: dict[str, tuple[Any, str]] = {}
    for node in iter_nodes(root):
        origin = getattr(node, "origin", None)
        if not isinstance(origin, Origin) or not origin.source:
            raise TypeError(f"{type(node).__name__} {origin!r} has no origin; build it with ctx.origin(step)")
        path = origin.path
        if "shared" in path.split("/"):
            raise ValueError(f"{path}: no step may be named 'shared'; that key holds the shared params")
        if path in seen:
            name = path.rpartition("/")[2]
            raise ValueError(
                f"two nodes have path {path!r} ({seen[path].source} and {origin.source}); "
                f"did you mean to name one of them, e.g. .named({name + '_2'!r})?"
            )
        seen[path] = origin
        if isinstance(node, CallNode):
            for decl in node.params:
                if not isinstance(decl, ParamDecl):
                    raise TypeError(
                        f"{path}: params holds {decl!r}, not a ParamDecl; pass literals as consts=((name, value),)"
                    )
                if decl.shared_key is not None:
                    record_shared_type(shared, decl, path)


def _relabel(node: IRNode, reads: dict, writes: dict, produced: set) -> IRNode:
    # A name read after the relabelled step produced it is one of its own
    # writes; before that it comes from outside, so `reads` applies.
    def read(name: str) -> str:
        return writes.get(name, name) if name in produced else reads.get(name, name)

    if isinstance(node, CallNode):
        names = [i.name for i in node.inputs or ()] + [o.name for o in node.outputs or ()]
        if node.kind == "frame" and any(n in reads or n in writes for n in names):
            raise TypeError(f"{node.origin.path}: relabel can't rename a frame step's columns; rename them in the function")
        inputs = None if node.inputs is None else tuple(replace(i, name=read(i.name)) for i in node.inputs)
        outputs = None if node.outputs is None else tuple(replace(o, name=writes.get(o.name, o.name)) for o in node.outputs)
        produced.update(o.name for o in node.outputs or ())
        return replace(node, inputs=inputs, outputs=outputs)
    if isinstance(node, SequenceNode):
        children = tuple(_relabel(c, reads, writes, produced) for c in node.children_)
        emits = tuple(writes.get(n, n) + at + where for n, at, where in (e.partition("@") for e in node.emits))
        drops = tuple(writes.get(d, reads.get(d, d)) for d in node.drops)
        return replace(node, children_=children, emits=emits, drops=drops)
    if isinstance(node, BranchNode):
        condition = _relabel(node.condition, reads, writes, produced)
        before, arms = set(produced), []
        for arm in node.arms:
            seen = set(before)
            arms.append(_relabel(arm, reads, writes, seen))
            produced |= seen
        return replace(node, condition=condition, arms=tuple(arms), modifies=tuple(writes.get(m, m) for m in node.modifies))
    if isinstance(node, LoopNode):
        # ponytail: a carry renamed differently by reads and writes gets two names; rejecting that waits for loop execution.
        condition = _relabel(node.condition, reads, writes, produced)
        body = _relabel(node.body, reads, writes, produced)
        return replace(node, condition=condition, body=body, carries=tuple(writes.get(c, c) for c in node.carries))
    raise TypeError(f"can't relabel {type(node).__name__}")
