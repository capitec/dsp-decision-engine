"""The step constructors by name, read by `decider ids` and by flow files."""
from __future__ import annotations

from decider.steps._branch import branch
from decider.steps._dag import dag
from decider.steps._each import each
from decider.steps._frame import frame_step
from decider.steps._function import step
from decider.steps._loop import loop
from decider.steps._optimise import optimise
from decider.steps._sequential import flow

SCALARS = {"step": step, "frame_step": frame_step}
COMPOSITES = {"flow": flow, "dag": dag, "branch": branch, "loop": loop, "each": each, "optimise": optimise}
