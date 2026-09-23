from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from decider.engine.ir.decls import Input
from decider.engine.ir.nodes import BranchNode, CallNode, IRNode, LoopNode, SequenceNode, iter_nodes
from decider.engine.wiring.plan import Branch, Call, Carry, Loop, Merge, Plan, Resolved, Sequence, Version
from decider.exceptions import WiringError
from decider.registry.resolve import suggest_name, suggest_names


def resolve(ir: Any) -> Plan:
    """Bind every name in an IR to the version it reads, and check the wiring.

    Written order is execution order: a node reads the latest version of each
    name, and every write is kept as a new version. A name nothing has
    produced yet is a pipeline input column, unless it is close to a name
    that has been produced, which is taken as a typo and raised with a
    suggestion. Accepts an IR node or anything `engine.to_ir` accepts.

    Raises `WiringError` (a `ValueError`), naming the node path, for: a likely
    typo, a node reading an input column that a later node overwrites, a
    read or bare emit of a name written only inside a branch arm, branch
    condition or loop body that doesn't pass it on, an emit or drop of a
    name that doesn't exist, a drop of a value only used internally, a
    `name@path` whose path produces no such name, a `modifies` name no arm
    writes, and a `carries` name the loop body never writes.

    Example::

        plan = resolve(pipeline)
        plan.outputs["term_cap"].producer   # "term/by_sector"
    """
    if not isinstance(ir, IRNode):
        from decider.engine.ir.context import to_ir

        ir = to_ir(ir)
    resolver = _Resolver({o.name: o.annotation for n in iter_nodes(ir) if isinstance(n, CallNode) for o in n.outputs or ()})
    scope = _Scope()
    root = resolver.walk(ir, scope)
    return resolver.finish(root, scope)


@dataclass(slots=True)
class _Scope:
    names: dict[str, Version] = field(default_factory=dict)
    barrier: Call | None = None
    # name -> path of the first node that read it as an input column
    leaf_readers: dict[str, str] = field(default_factory=dict)

    def child(self) -> _Scope:
        return _Scope(dict(self.names), self.barrier, dict(self.leaf_readers))

    def absorb(self, other: _Scope) -> None:
        for name, reader in other.leaf_readers.items():
            self.leaf_readers.setdefault(name, reader)


class _Resolver:
    def __init__(self, annotations: dict[str, Any]) -> None:
        self.annotations = annotations
        self.versions: list[Version] = []
        self.calls: list[Call] = []
        self.leaves: dict[str, Version] = {}
        self.inputs: dict[str, Input] = {}
        self.chains: dict[str, list[Version]] = {}
        self.read: set[int] = set()
        self.passthrough: set[int] = set()
        self.emitted: dict[str, Version] = {}
        self.drops: list[tuple[str, str]] = []
        # Finished branch/loop path -> the names it passes on, for errors about the ones it hides.
        self.closed: dict[str, str] = {}

    def new(self, name: str, producer: str | None, annotation: Any) -> Version:
        v = Version(len(self.versions), name, producer, annotation)
        self.versions.append(v)
        if producer is not None:
            self.chains.setdefault(name, []).append(v)
        return v

    def walk(self, node: IRNode, scope: _Scope) -> Resolved:
        if isinstance(node, CallNode):
            return self.call(node, scope)
        if isinstance(node, SequenceNode):
            children = tuple(self.walk(c, scope) for c in node.children_)
            for spec in node.emits:
                self.emit(node.origin.path, spec, scope)
            self.drops += [(node.origin.path, d) for d in node.drops]
            return Sequence(node, children)
        if isinstance(node, BranchNode):
            return self.branch(node, scope)
        if isinstance(node, LoopNode):
            return self.loop(node, scope)
        raise TypeError(f"can't resolve {type(node).__name__}")

    def call(self, node: CallNode, scope: _Scope) -> Call:
        path = node.origin.path
        own = {o.name for o in node.outputs or ()}
        reads = None
        if node.inputs is not None:
            reads = tuple(self.lookup(i.name, scope, path, own, i) for i in node.inputs)
            for v in reads:
                if v.producer is None:
                    scope.leaf_readers.setdefault(v.name, path)
        call = Call(len(self.calls), node, reads, ())
        self.calls.append(call)
        if node.outputs is None:
            # Unknown lineage: the frame may have rewritten anything, so every
            # later read that isn't produced after it comes from it.
            scope.names.clear()
            scope.barrier = call
        else:
            call.writes = tuple(self.write(o.name, o.annotation, path, scope) for o in node.outputs)
        return call

    def lookup(self, name: str, scope: _Scope, reader: str, own: set[str] = frozenset(), decl: Input | None = None) -> Version:
        v = scope.names.get(name)
        if v is None:
            v = scope.names[name] = self.unbound(name, scope, reader, own, decl)
        self.read.add(v.id)
        return v

    def unbound(self, name: str, scope: _Scope, reader: str, own: set[str], decl: Input | None) -> Version:
        annotation = decl.annotation if decl is not None else self.annotations.get(name)
        if scope.barrier is not None:
            return self.from_barrier(name, scope.barrier, annotation)
        hidden = self.hidden(name)
        if hidden is not None:
            v, owner = hidden
            raise WiringError(
                f"{reader} reads {name!r}, but the only step writing it, {v.producer}, is inside {owner}, "
                f"so {name!r} is not visible after it. List {name!r} there to pass it on, or rename one of "
                f"them if {reader} means an input column."
            )
        # A node's own outputs are no typo: `term_cap_a` may read `term_cap`.
        produced = {n: v for n, v in scope.names.items() if v.producer is not None and n not in own}
        near = suggest_name(name, produced)
        if near is not None:
            raise WiringError(
                f"{reader}: input {name!r} is not produced by any earlier step and is not a declared input "
                f"column. Did you mean {near!r} (produced by {produced[near].producer!r})? Rename it, or "
                f"relabel(reads={{{name!r}: {near!r}}})."
            )
        if name not in self.leaves:
            self.leaves[name] = self.new(name, None, annotation)
            # ponytail: the first reader's declaration describes the column; each Call still pairs its own Input.
            self.inputs[name] = decl or Input(name, annotation)
            self.passthrough.add(self.leaves[name].id)
        return self.leaves[name]

    def from_barrier(self, name: str, barrier: Call, annotation: Any) -> Version:
        for v in barrier.writes:
            if v.name == name:
                return v
        v = self.new(name, barrier.node.origin.path, annotation)
        barrier.writes += (v,)
        self.passthrough.add(v.id)
        return v

    def write(self, name: str, annotation: Any, writer: str, scope: _Scope) -> Version:
        old = scope.names.get(name)
        reader = scope.leaf_readers.get(name)
        if (old is None or old.producer is None) and reader not in (None, writer):
            raise WiringError(
                f"{reader} reads {name!r} as an input column, but {writer}, which runs later, writes "
                f"{name!r}. Order is execution order: {reader} saw the input column, while anything after "
                f"{writer} sees {writer}'s value. Move {writer} before {reader}, or rename one of the two."
            )
        v = scope.names[name] = self.new(name, writer, annotation)
        return v

    def hidden(self, name: str) -> tuple[Version, str] | None:
        """The last version of `name` written inside a finished branch or loop, and that branch or loop."""
        for v in reversed(self.chains.get(name, ())):
            owners = [p for p in self.closed if v.producer.startswith(p + "/")]
            if owners:
                return v, self.closed[max(owners, key=len)]
        return None

    def final(self, name: str, inner: _Scope, owner: str) -> Version | None:
        """The version of `name` that nodes under `owner` left in `inner`, or `None` if they left it alone."""
        prefix = owner + "/"
        v = inner.names.get(name)
        if v is None and inner.barrier is not None and inner.barrier.node.origin.path.startswith(prefix):
            v = self.from_barrier(name, inner.barrier, self.annotations.get(name))
        return v if v is not None and v.producer is not None and v.producer.startswith(prefix) else None

    def branch(self, node: BranchNode, scope: _Scope) -> Branch:
        path = node.origin.path
        # The condition's own output is only for routing, not visible after the branch.
        cond = scope.child()
        condition = self.call(node.condition, cond)
        arm_scopes = [cond.child() for _ in node.arms]
        arms = tuple(self.walk(arm, s) for arm, s in zip(node.arms, arm_scopes))
        merges = []
        for name in node.modifies:
            finals = tuple(self.final(name, s, path) for s in arm_scopes)
            written = [v for v in finals if v is not None]
            if not written:
                raise WiringError(f"branch {path}: modifies {name!r}, but no arm writes it")
            prior = self.lookup(name, cond, path) if None in finals else None
            merges.append(Merge(self.new(name, path, written[0].annotation), prior, finals))
        for s in (cond, *arm_scopes):
            scope.absorb(s)
        self.closed[path] = f"branch {path}, which passes on only modifies={list(node.modifies)}"
        for m in merges:
            scope.names[m.version.name] = m.version
        return Branch(node, condition, arms, tuple(merges))

    def loop(self, node: LoopNode, scope: _Scope) -> Loop:
        path = node.origin.path
        initials = [self.lookup(c, scope, path) for c in node.carries]
        carried = [self.new(c, path, v.annotation or self.annotations.get(c)) for c, v in zip(node.carries, initials)]
        inner = scope.child()
        inner.names.update(zip(node.carries, carried))
        cond = inner.child()
        condition = self.call(node.condition, cond)
        inner.absorb(cond)
        body = self.walk(node.body, inner)
        carries = []
        for c, initial, v in zip(node.carries, initials, carried):
            last = self.final(c, inner, path)
            if last is None:
                raise WiringError(f"loop {path}: carries {c!r}, but the body never writes it")
            carries.append(Carry(v, initial, last))
        scope.absorb(inner)
        self.closed[path] = f"loop {path}, which passes on only carries={list(node.carries)}"
        # Reads inside the loop don't consume the value the loop hands on.
        self.read -= {v.id for v in carried}
        scope.names.update(zip(node.carries, carried))
        return Loop(node, condition, body, tuple(carries))

    def emit(self, where: str, spec: str, scope: _Scope) -> None:
        label = where or "<root>"
        name, at, producer = spec.partition("@")
        if not at:
            v = scope.names.get(name)
            if v is None and scope.barrier is not None:
                v = self.from_barrier(name, scope.barrier, self.annotations.get(name))
            hidden = self.hidden(name) if v is None else None
            if hidden is not None:
                raise WiringError(
                    f"{label}: emit({spec!r}): {name!r} is only written inside {hidden[1]}. "
                    f"Emit that version as '{name}@{hidden[0].producer}', or list {name!r} there to pass it on."
                )
            if v is None:
                raise WiringError(
                    f"{label}: emit({spec!r}): no step produces {name!r} and it is not a declared input "
                    f"column.{_hint(name, [*scope.names, *self.chains])}"
                )
            self.emitted[name] = v
            return
        prefix = f"{where}/" if where else ""
        mine = [v for v in self.chains.get(name, ()) if f"{v.producer}/".startswith(prefix)]
        if not mine:
            raise WiringError(
                f"{label}: emit({spec!r}): no step {'under ' + label + ' ' if where else ''}produces {name!r}."
                f"{_hint(name, self.chains)}"
            )
        relative = [v.producer[len(prefix):] for v in mine]
        if producer == "*":
            self.emitted.update((f"{name}@{r}", v) for r, v in zip(relative, mine))
            return
        # A path relative to this flow, or absolute as in step_map and run reports.
        for paths in (relative, [v.producer for v in mine]):
            if producer in paths:
                self.emitted[spec] = mine[paths.index(producer)]
                return
        raise WiringError(
            f"{label}: emit({spec!r}): {name!r} is never produced by {producer!r}. "
            f"Producers, in order: {relative}.{_hint(producer, [*relative, *(v.producer for v in mine)])}"
        )

    def finish(self, root: Resolved, scope: _Scope) -> Plan:
        # Input columns read only inside a branch or loop still pass through.
        outputs = dict(self.leaves)
        outputs.update(
            (name, v) for name, v in scope.names.items()
            if name in self.leaves or v.id in self.passthrough or v.id not in self.read
        )
        outputs.update(self.emitted)
        drops = []
        for where, name in self.drops:
            if name not in outputs and name in self.chains:
                raise WiringError(
                    f"{where or '<root>'}: drop({name!r}): {name!r} is an internal value, not an input "
                    "column, a final value or an emitted one, so there is nothing in the output to drop."
                )
            if name not in outputs and scope.barrier is None:
                near = suggest_name(name, [*outputs, *self.chains])
                if near is not None:
                    raise WiringError(f"{where or '<root>'}: drop({name!r}): no such value. Did you mean {near!r}?")
            outputs.pop(name, None)
            drops.append(name)
        return Plan(
            root, tuple(self.calls), tuple(self.versions), tuple(self.inputs.values()), outputs,
            tuple(dict.fromkeys(drops)), {name: tuple(chain) for name, chain in self.chains.items()},
        )


def _hint(name: str, candidates) -> str:
    near = suggest_names(name, candidates)
    return f" Did you mean {near[0]!r}?" if near else ""
