"""Pipeline composition for the decider framework.

Pipeline  - ordered sequence of modules threading frames forward
ForkPipeline - parallel branches with dict output
"""

import typing as t
import polars as pl
from dataclasses import dataclass, field

if t.TYPE_CHECKING:
    from decider.modules.core import BaseModule
    from decider.executor import Executor


@dataclass
class Pipeline:
    """Ordered sequence of modules. Each module reads from the previous one's output.

    Create via:
        mod_a | mod_b | mod_c
        Pipeline.chain(mod_a, mod_b, mod_c)

    Execute:
        result = pipeline.execute({"input": df})          # → LazyFrame (last frame)
        result = pipeline.execute({"input": df},
            output_frames=["engagement", "score"])        # → dict of frames
    """

    steps: t.List[t.Union["BaseModule", "ForkPipeline"]]

    # ── operators ────────────────────────────────────────────────────────────

    def __or__(self, other: t.Union["BaseModule", "Pipeline", "ForkPipeline"]) -> "Pipeline":
        if isinstance(other, Pipeline):
            return Pipeline(self.steps + other.steps)
        return Pipeline(self.steps + [other])

    def __and__(self, other: t.Union["BaseModule", "Pipeline", "ForkPipeline"]) -> "ForkPipeline":
        right = other if isinstance(other, (Pipeline, ForkPipeline)) else Pipeline([other])
        return ForkPipeline(branches=[self, right])

    # ── helpers ───────────────────────────────────────────────────────────────

    @classmethod
    def chain(cls, *modules: "BaseModule") -> "Pipeline":
        """Create a sequential pipeline from multiple modules.

        Equivalent to: mod_a | mod_b | mod_c
        """
        return cls(list(modules))

    # ── execution ─────────────────────────────────────────────────────────────

    def execute(
        self,
        frames: t.Dict[str, t.Union[pl.LazyFrame, pl.DataFrame]],
        output_frames: t.Optional[t.List[str]] = None,
        executor: t.Optional["Executor"] = None,
        debug: bool = False,
        lazy: bool = True,
        allow_overrides: bool = False,
    ) -> t.Union[pl.LazyFrame, pl.DataFrame, t.Dict[str, pl.LazyFrame], t.Dict[str, pl.DataFrame]]:
        """Execute the pipeline, threading frames through each step.

        Args:
            frames: Input frames. Convention: use "input" as the main frame.
            output_frames: Names of intermediate frames to return.
                           If None, returns the final frame.
            executor: Optional executor override.
            debug: Print debug info.
            lazy: If True (default), return LazyFrame(s). If False, collect
                  eagerly and return DataFrame(s).
            allow_overrides: If True, input columns whose names match a module's
                computed columns take precedence (what-if injection). If False
                (default), such collisions raise a ValueError.

        Returns:
            * ``lazy=True, output_frames=None``  → ``pl.LazyFrame``
            * ``lazy=False, output_frames=None`` → ``pl.DataFrame``
            * ``lazy=True, output_frames=[...]`` → ``Dict[str, pl.LazyFrame]``
            * ``lazy=False, output_frames=[...]`` → ``Dict[str, pl.DataFrame]``

        Frame names in ``output_frames`` correspond to module names.  Each
        module stores its output under ``module.name`` after execution.
        """
        result: t.Dict[str, pl.LazyFrame] = {
            k: (v.lazy() if isinstance(v, pl.DataFrame) else v)
            for k, v in frames.items()
        }

        # Start from "input" if present, otherwise first available frame
        prev_frame_name = "input" if "input" in result else next(iter(result))

        for step in self.steps:
            if isinstance(step, ForkPipeline):
                # Fork step: run all branches, merge their outputs into result
                fork_result = step.execute(
                    {**result, "input": result[prev_frame_name]},
                    executor=executor,
                    debug=debug,
                    allow_overrides=allow_overrides,
                )
                result.update(fork_result)
                # prev_frame_name intentionally unchanged — the module after the
                # fork (typically a JoinModule) references frames by name itself

            else:
                compiled = step.compile(executor, allow_overrides=allow_overrides)

                if debug:
                    print(f"\n--- Step: {step.name} (input_frame='{prev_frame_name}') ---")

                if compiled.expression_groups:
                    # Expression module: alias prev output as "input" so the
                    # module's target_frame="input" nodes find it correctly.
                    # Note: explicit "input" must come LAST to override any
                    # existing "input" key carried forward from earlier steps.
                    frames_for_step = {**result, "input": result[prev_frame_name]}
                else:
                    # Frame module (e.g. JoinModule): references frames by name,
                    # doesn't need "input" aliasing
                    frames_for_step = dict(result)

                step_result = compiled._execute_raw(frames_for_step, debug=debug)

                if compiled.frame_operations:
                    # Frame module created a new named frame
                    new_frame_name = compiled.frame_operations[-1].name
                    result[new_frame_name] = step_result[new_frame_name]
                    prev_frame_name = new_frame_name
                else:
                    # Expression module modified "input" — store under module name
                    result[step.name] = step_result["input"]
                    prev_frame_name = step.name

        if output_frames is not None:
            missing = [k for k in output_frames if k not in result]
            if missing:
                raise KeyError(
                    f"Requested output_frames {missing} not found. "
                    f"Available frames after execution: {sorted(result.keys())}. "
                    f"Frame names come from module names (module.name) or "
                    f"JoinModule.output_frame."
                )
            out = {k: result[k] for k in output_frames}
            if not lazy:
                return {k: v.collect() for k, v in out.items()}
            return out

        final = result[prev_frame_name]
        return final.collect() if not lazy else final


@dataclass
class ForkPipeline:
    """Parallel branches that each produce an output frame.

    Create via:
        pipeline_a & pipeline_b
        some_module.include(mod_b, mod_c)   # branches from a shared predecessor

    Execute returns a dict keyed by each branch's terminal module name:
        result = fork.execute({"input": df})
        # → {"branch_a_terminal": df, "branch_b_terminal": df}
    """

    branches: t.List[t.Union[Pipeline, "ForkPipeline"]]
    prefix: t.Optional[Pipeline] = None
    """Shared prefix executed once before branching (used by .include())."""

    # ── operators ────────────────────────────────────────────────────────────

    def __or__(self, other: t.Union["BaseModule", Pipeline]) -> Pipeline:
        """Rejoin: ForkPipeline | join_module → Pipeline."""
        tail = other if isinstance(other, Pipeline) else Pipeline([other])
        return Pipeline([self] + tail.steps)

    def __and__(self, other: t.Union["BaseModule", Pipeline, "ForkPipeline"]) -> "ForkPipeline":
        right = other if isinstance(other, (Pipeline, ForkPipeline)) else Pipeline([other])
        return ForkPipeline(branches=self.branches + [right])

    # ── execution ─────────────────────────────────────────────────────────────

    def execute(
        self,
        frames: t.Dict[str, t.Union[pl.LazyFrame, pl.DataFrame]],
        output_frames: t.Optional[t.List[str]] = None,
        executor: t.Optional["Executor"] = None,
        debug: bool = False,
        lazy: bool = True,
        allow_overrides: bool = False,
    ) -> t.Union[t.Dict[str, pl.LazyFrame], t.Dict[str, pl.DataFrame]]:
        """Execute all branches and return their outputs as a dict.

        ForkPipeline always returns a dict (one entry per branch terminal).

        Args:
            frames: Input frames.
            output_frames: Specific branch names to include. If None, all branches.
            executor: Optional executor override.
            debug: Print debug info.
            lazy: If False, collect each branch frame eagerly.
            allow_overrides: Passed through to each branch's execution.

        Returns:
            Dict of {branch_terminal_name: LazyFrame | DataFrame}.
        """
        result: t.Dict[str, pl.LazyFrame] = {
            k: (v.lazy() if isinstance(v, pl.DataFrame) else v)
            for k, v in frames.items()
        }

        # Run shared prefix once (used by .include())
        if self.prefix is not None:
            prefix_output = self.prefix.execute(result, executor=executor, debug=debug,
                                                allow_overrides=allow_overrides)
            last_name = self.prefix.steps[-1].name  # type: ignore[union-attr]
            result[last_name] = prefix_output
            result["input"] = prefix_output  # branches see prefix output as "input"

        # Run each branch independently
        branch_results: t.Dict[str, pl.LazyFrame] = {}

        for branch in self.branches:
            branch_output = branch.execute(result, executor=executor, debug=debug,
                                           allow_overrides=allow_overrides)

            if isinstance(branch_output, dict):
                branch_results.update(branch_output)
            else:
                # Determine terminal name
                terminal = branch.steps[-1] if isinstance(branch, Pipeline) else branch
                terminal_name = getattr(terminal, "name", "output")
                branch_results[terminal_name] = branch_output

        if output_frames is not None:
            missing = [k for k in output_frames if k not in branch_results]
            if missing:
                raise KeyError(
                    f"Requested output_frames {missing} not found. "
                    f"Available branch frames: {sorted(branch_results.keys())}."
                )
            branch_results = {k: branch_results[k] for k in output_frames}

        if not lazy:
            return {k: v.collect() for k, v in branch_results.items()}
        return branch_results
