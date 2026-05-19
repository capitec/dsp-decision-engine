"""Executor infrastructure for compiling and executing computation graphs.

This module provides:
- Executor ABC: Strategy pattern for different execution approaches
- CompiledDag: Cached, executable artifact
- SimpleExecutor: One-by-one expression application
- (Future) WaveExecutor: Batched independent expressions
"""

import typing as t
import polars as pl
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

if t.TYPE_CHECKING:
    from decider.modules.core import Node


@dataclass
class FrameOperation:
    """A frame-level operation (join, filter, etc.) to execute."""
    name: str
    """Output frame name"""

    callable: t.Callable[[t.Dict[str, pl.LazyFrame]], pl.LazyFrame]
    """Function that takes dict of frames and returns a new LazyFrame"""

    depends_on: t.List[str]
    """Names of frames this operation depends on"""


@dataclass
class CompiledDag:
    """Executable artifact from compiling a module's nodes.

    This is the cached result of compilation - applying it to data is the hot path.
    """

    expression_groups: t.List[t.Tuple[str, t.List[pl.Expr]]]
    """List of (frame_name, expressions) to apply. Ordered by dependencies."""

    frame_operations: t.List[FrameOperation] = field(default_factory=list)
    """Frame-level operations (joins, etc.) interleaved with expressions."""

    allow_overrides: bool = False
    """If True, input columns that share a name with a computed expression silently
    win (the expression is skipped for that column). This enables what-if injection:
    pass a pre-computed column in the input frame to bypass the module's own logic.

    If False (default), a name collision raises a ValueError — because silent
    shadowing is almost always a wiring mistake.
    """

    output_column_names: t.List[str] = field(default_factory=list)
    """Column names this DAG will produce. Used to detect input-shadows-function
    collisions before applying expressions."""

    def execute(
        self,
        frames: t.Dict[str, t.Union[pl.LazyFrame, pl.DataFrame]],
        output_frames: t.Optional[t.List[str]] = None,
        debug: bool = False,
        lazy: bool = True,
    ) -> t.Union[pl.LazyFrame, pl.DataFrame, t.Dict[str, pl.LazyFrame], t.Dict[str, pl.DataFrame]]:
        """Execute the compiled plan on input dataframes.

        Return semantics match Pipeline.execute():
        - Default (output_frames=None) → LazyFrame of the primary result frame
        - output_frames=[...] → Dict[str, LazyFrame] of named frames
        - lazy=False → collect eagerly (DataFrame instead of LazyFrame)

        The primary result frame is "input" for expression modules, or the
        last frame operation's output name for frame modules (e.g. JoinModule).

        Args:
            frames: Input dataframes (will be converted to LazyFrame)
            output_frames: Named frames to return. If None, returns primary frame.
            debug: If True, print debug information
            lazy: If False, collect results eagerly.
        """
        if debug:
            normalized = {
                k: (v.lazy() if isinstance(v, pl.DataFrame) else v)
                for k, v in frames.items()
            }
            print(f"\n=== Executing CompiledDag ===")
            print(f"Input frames: {list(normalized.keys())}")
            print(f"Expression groups: {len(self.expression_groups)}")
            print(f"Frame operations: {len(self.frame_operations)}")

        result = self._execute_raw(frames, debug=debug)

        if debug:
            print(f"Output frames: {list(result.keys())}")
            print("=" * 50)

        if output_frames is not None:
            missing = [k for k in output_frames if k not in result]
            if missing:
                raise KeyError(
                    f"Requested output_frames {missing} not found. "
                    f"Available frames: {sorted(result.keys())}."
                )
            out = {k: result[k] for k in output_frames}
            return {k: v.collect() for k, v in out.items()} if not lazy else out

        # Default: return the primary frame (last frame op output, or "input")
        primary = self.frame_operations[-1].name if self.frame_operations else "input"
        frame = result[primary]
        return frame.collect() if not lazy else frame

    def _execute_raw(
        self,
        frames: t.Dict[str, t.Union[pl.LazyFrame, pl.DataFrame]],
        debug: bool = False,
    ) -> t.Dict[str, pl.LazyFrame]:
        """Internal: execute and return the full frame dict. Used by Pipeline."""
        # Re-run execute internals but always return full dict.
        # We duplicate the normalization + application logic rather than
        # calling execute() so Pipeline can read any frame by name.
        result: t.Dict[str, pl.LazyFrame] = {
            k: (v.lazy() if isinstance(v, pl.DataFrame) else v)
            for k, v in frames.items()
        }

        # Detect input columns that collide with this DAG's computed column names.
        # Pipeline steps legitimately pass prior-step columns through (that's how
        # frame threading works), so we only check the target frame for each group.
        # We build a local, non-mutating copy of expression_groups to use below.
        active_expression_groups = self.expression_groups
        if self.output_column_names and not self.allow_overrides:
            for target_frame, _ in self.expression_groups:
                if target_frame in result:
                    try:
                        existing_cols = set(result[target_frame].collect_schema().names())
                    except Exception:
                        existing_cols = set()
                    collisions = existing_cols & set(self.output_column_names)
                    if collisions:
                        raise ValueError(
                            f"Input frame '{target_frame}' already contains column(s) "
                            f"{sorted(collisions)} that this module is about to compute. "
                            f"This is usually a wiring mistake — a parameter name in this "
                            f"module matches an existing column.\n"
                            f"To allow input columns to take precedence over computed ones "
                            f"(what-if injection), pass allow_overrides=True to "
                            f"module.compile()."
                        )
        elif self.output_column_names and self.allow_overrides:
            # Drop expressions whose output column is already present in the target
            # frame — the existing column wins (what-if injection).
            overridden: t.Set[str] = set()
            for target_frame, _ in self.expression_groups:
                if target_frame in result:
                    try:
                        existing_cols = set(result[target_frame].collect_schema().names())
                    except Exception:
                        existing_cols = set()
                    overridden |= existing_cols & set(self.output_column_names)
            if overridden:
                active_expression_groups = [
                    (tf, [e for e in es if e.meta.output_name() not in overridden])
                    for tf, es in self.expression_groups
                ]

        for target_frame, exprs in active_expression_groups:
            if target_frame not in result:
                hint = (
                    " Pass your dataframe as {\"input\": df} to use the default frame."
                    if target_frame == "input" else ""
                )
                raise ValueError(
                    f"Target frame '{target_frame}' not found. "
                    f"Available frames: {list(result.keys())}.{hint}"
                )
            if debug:
                print(f"  Applying {len(exprs)} expressions to '{target_frame}'")
            frame = result[target_frame]
            for expr in exprs:
                frame = frame.with_columns(expr)
            result[target_frame] = frame

        for frame_op in self.frame_operations:
            if debug:
                print(f"  Executing frame operation: {frame_op.name}")
            for dep in frame_op.depends_on:
                if dep not in result:
                    raise ValueError(
                        f"Frame operation '{frame_op.name}' depends on '{dep}' "
                        f"which doesn't exist. Available: {list(result.keys())}"
                    )
            result[frame_op.name] = frame_op.callable(result)

        return result


class Executor(ABC):
    """Abstract base class for execution strategies.

    An executor knows how to:
    1. Compile a list of Nodes into a CompiledDag
    2. Determine expression ordering and batching strategy
    """

    @abstractmethod
    def compile(self, nodes: t.List["Node"], allow_overrides: bool = False) -> CompiledDag:
        """Compile nodes into an executable CompiledDag.

        Args:
            nodes: List of nodes from a module
            allow_overrides: If True, input columns that share a name with a
                computed column take precedence (what-if injection). If False
                (default), such collisions raise a ValueError.

        Returns:
            CompiledDag ready for execution
        """
        ...


class SimpleExecutor(Executor):
    """Simple executor that applies expressions one-by-one and handles frame operations.

    This is the default executor. It:
    - Topologically sorts nodes by dependencies
    - Separates expression nodes from frame nodes
    - Creates ordered expression groups and frame operations
    """

    def compile(self, nodes: t.List["Node"], allow_overrides: bool = False) -> CompiledDag:
        """Compile nodes into ordered expressions and frame operations.

        Strategy:
        1. Separate expression nodes from frame nodes
        2. Topological sort by dependencies
        3. Group expressions by target frame
        4. Create frame operations for frame nodes
        """
        from decider.modules.core import ExternalInputNode

        # Separate by type
        expr_nodes = [n for n in nodes if n.node_type == "expression"]
        frame_nodes = [n for n in nodes if n.node_type == "frame"]

        # Topological sort both groups
        sorted_expr_nodes = self._topological_sort(expr_nodes)
        sorted_frame_nodes = self._topological_sort(frame_nodes)

        # Build expressions for expression nodes
        expressions_by_frame: t.Dict[str, t.List[pl.Expr]] = {}

        for node in sorted_expr_nodes:
            # Call the function to get the expression
            try:
                expr = node.callable(**node.get_input_expressions())
            except Exception as e:
                missing = [
                    f"'{k}' (looked up as column '{v.input_name}')"
                    for k, v in node.input_map.items()
                    if isinstance(v, ExternalInputNode)
                ]
                hint = (
                    f"\nHint: node '{node.name}' expected these input columns: "
                    + ", ".join(missing)
                    + ". Make sure the column names match your function parameter names."
                ) if missing else ""
                raise ValueError(
                    f"Error building expression for '{node.name}': {e}{hint}"
                ) from e

            # Add alias
            expr = expr.alias(node.name)

            # Group by target frame
            target = node.target_frame
            if target not in expressions_by_frame:
                expressions_by_frame[target] = []
            expressions_by_frame[target].append(expr)

        # Convert to ordered list of (frame, expressions) tuples
        expression_groups = list(expressions_by_frame.items())

        # Build frame operations
        frame_operations: t.List[FrameOperation] = []

        for node in sorted_frame_nodes:
            def make_frame_callable(node_ref: "Node") -> t.Callable:
                return lambda frames: node_ref.callable(**node_ref.get_frame_kwargs(frames))

            frame_operations.append(FrameOperation(
                name=node.target_frame,
                callable=make_frame_callable(node),
                depends_on=node.frame_dependencies,
            ))

        output_column_names = [node.name for node in sorted_expr_nodes]

        return CompiledDag(
            expression_groups=expression_groups,
            frame_operations=frame_operations,
            allow_overrides=allow_overrides,
            output_column_names=output_column_names,
        )

    def _topological_sort(self, nodes: t.List["Node"]) -> t.List["Node"]:
        """Sort nodes in dependency order using Kahn's algorithm."""
        from decider.modules.core import Node as NodeClass
        from collections import deque

        # Build adjacency list and in-degree count
        node_map = {n.name: n for n in nodes}
        in_degree = {n.name: 0 for n in nodes}
        adjacency = {n.name: [] for n in nodes}

        for node in nodes:
            for input_ref in node.input_map.values():
                if isinstance(input_ref, NodeClass):
                    # This node depends on input_ref
                    dep_name = input_ref.name
                    if dep_name in node_map:
                        adjacency[dep_name].append(node.name)
                        in_degree[node.name] += 1

        # Kahn's algorithm
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        sorted_names = []

        while queue:
            current = queue.popleft()
            sorted_names.append(current)

            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_names) != len(nodes):
            # Circular dependency
            remaining = [name for name, degree in in_degree.items() if degree > 0]
            raise ValueError(f"Circular dependency detected involving: {remaining}")

        return [node_map[name] for name in sorted_names]