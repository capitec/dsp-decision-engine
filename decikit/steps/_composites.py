"""The step constructors by name, read by `decikit ids` and by flow files."""
from __future__ import annotations

from decikit.steps._branch import branch
from decikit.steps._dag import dag
from decikit.steps._each import each
from decikit.steps._frame import frame_step
from decikit.steps._function import step
from decikit.steps._loop import loop
from decikit.steps._optimise import optimise
from decikit.steps._sequential import flow

SCALARS = {"step": step, "frame_step": frame_step}
COMPOSITES = {"flow": flow, "dag": dag, "branch": branch, "loop": loop, "each": each, "optimise": optimise}
